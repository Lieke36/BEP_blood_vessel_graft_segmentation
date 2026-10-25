### Evaluate ###

# What does Evaluate.py do?
# This script creates the segmentation prediction masks of a set of specified patches using model 1 architecture (U-Net resnet34 + ImageNet initialization)

# How do use Evaluate.py?
# 1. Change the input, output and model paths to the correct location. The input path should be a folder containing tissue and masks image patches.
# The output path should be the path where the prediction masks should be stored. The model path should be the path to the model parameters which will be used to create the segmentation.
# 2. Set the slide_id_list and patch_nr_list variables to the patch id's you want to segment.
 
## Libraries ##

from pathlib import Path
import numpy as np
from PIL import Image
import torch
import segmentation_models_pytorch as smp
from Losses_metrics import calculate_metrics


## Configuration ##

ENCODER = "resnet34"
IN_CHANNELS = 3
CLASSES = 1
THRESHOLD = 0.5

## settings ##

INPUT_PATH = Path(
    r"/home/20234257/BEP_Project/External_data/validation"
)

OUTPUT_PATH = Path(
    r"/home/20234257/BEP_Project/BEP_blood_vessel_graft_segmentation-main/results/evaluation/n10000_bs8_ep50_5_70_25"
)

MODEL_PATH = Path(
    r"/home/20234257/BEP_Project/BEP_blood_vessel_graft_segmentation-main/results/models/n10000_bs8_ep50_5_70_25_best.pth"
)

# Note:
# train_slide_id_list = [
#     9943, 10010, 10210, 10143, 10315, 9972, 10174,
#     10955, 10925, 11014, 10858, 11057, 11133, 10765, 
#     10782, 10535, 10361, 10404, 10449, 10284, 10503, 
#     11525, 11418, 11629, 7522, 80490, 80501, 80589
# ]
# test_slide_id_list = [2152, 11193, 10648, 11743, 80975, 80600]
# val_slide_id_list = [10251, 11084, 10886, 11712, 11273, 80655]

slide_id_list = [9943]
patch_nr_list = [0, 104, 113, 126, 127]


## ImageNet normalization ##

IMAGE_MEAN = np.array(
    [
        0.485,
        0.456,
        0.406,
    ],
    dtype=np.float32,
).reshape(
    3,
    1,
    1,
)

IMAGE_STD = np.array(
    [
        0.229,
        0.224,
        0.225,
    ],
    dtype=np.float32,
).reshape(
    3,
    1,
    1,
)


## Device ##

def get_device():
    """
    Automatically selects CUDA if available,
    otherwise CPU.
    """

    if torch.cuda.is_available():

        device = torch.device("cuda")

        print(
            f"Using GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    else:

        device = torch.device("cpu")

        print(
            "CUDA not available. "
            "Using CPU."
        )

    return device


## Create model ##

def create_model():
    """
    Creates the same U-Net architecture used during training.

    The model outputs raw logits because activation=None.

    This must match the architecture used in Model.py.
    """

    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=None,
        in_channels=IN_CHANNELS,
        classes=CLASSES,
        activation=None,
    )

    return model


## Load trained model ##

def load_model(
    model_path,
    device,
):
    """
    Loads a trained model from a .pth checkpoint.

    The checkpoints produced by Model.py contain:

        model_state_dict
        optimizer_state_dict
        epoch
        validation_dice
        etc.

    Only model_state_dict is needed for prediction.
    """

    model_path = Path(
        model_path
    )

    if not model_path.exists():

        raise FileNotFoundError(
            f"Model checkpoint not found:\n"
            f"{model_path}"
        )

    print(
        f"Loading model:\n"
        f"{model_path}"
    )

    model = create_model()

    checkpoint = torch.load(
        model_path,
        map_location=device,
    )

    # --------------------------------------------------------
    # Checkpoint created by Model.py
    # --------------------------------------------------------

    if isinstance(checkpoint, dict) and \
       "model_state_dict" in checkpoint:

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        if "epoch" in checkpoint:

            print(
                f"Checkpoint epoch: "
                f"{checkpoint['epoch']}"
            )

        if "validation_dice" in checkpoint:

            print(
                f"Checkpoint validation Dice: "
                f"{checkpoint['validation_dice']:.4f}"
            )

    else:

        # ----------------------------------------------------
        # Also support a .pth containing only state_dict.
        # ----------------------------------------------------

        model.load_state_dict(
            checkpoint
        )

    model = model.to(
        device
    )

    model.eval()

    return model


## Load and preprocess tissue image ##

def preprocess_image(
    image_path,
):
    """
    Loads a tissue PNG and applies the exact preprocessing
    used during training in Model.py.

    Returns:
        torch.Tensor with shape:
            1 x 3 x H x W
    """

    image = Image.open(
        image_path
    ).convert("RGB")

    image = np.asarray(
        image,
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # H x W x C
    #
    # ->
    #
    # C x H x W
    # --------------------------------------------------------

    image = image.transpose(
        2,
        0,
        1,
    )

    # --------------------------------------------------------
    # Scale to [0, 1]
    # --------------------------------------------------------

    image = (
        image / 255.0
    )

    # --------------------------------------------------------
    # ImageNet normalization
    # --------------------------------------------------------

    image = (
        image - IMAGE_MEAN
    ) / IMAGE_STD

    # --------------------------------------------------------
    # NumPy -> PyTorch
    # --------------------------------------------------------

    image = torch.from_numpy(
        image
    ).float()

    # --------------------------------------------------------
    # Add batch dimension
    #
    # C x H x W
    #
    # ->
    #
    # 1 x C x H x W
    # --------------------------------------------------------

    image = image.unsqueeze(
        0
    )

    return image


## Load ground truth mask ##

def load_ground_truth(
    mask_path,
):
    """
    Loads a ground-truth mask.

    Any nonzero pixel is treated as foreground.

    Returns:
        torch.Tensor with shape:
            1 x 1 x H x W
    """

    mask = Image.open(
        mask_path
    ).convert("L")

    mask = np.asarray(
        mask,
        dtype=np.uint8,
    )

    mask = (
        mask > 0
    ).astype(
        np.float32
    )

    mask = torch.from_numpy(
        mask
    ).float()

    # H x W
    #
    # ->
    #
    # 1 x 1 x H x W
    mask = mask.unsqueeze(
        0
    ).unsqueeze(
        0
    )

    return mask


## Download predicted mask ##

def download_predicted_mask(
    slide_id,
    patch_nr,
    input_path,
    output_path,
    model_path,
    threshold=THRESHOLD,
):
    """
    Predicts and saves the segmentation mask for one patch.

    Also saves the sigmoid probability map before the
    threshold is applied.

    Parameters
    ----------
    slide_id : str or int
        Slide identifier.

    patch_nr : str or int
        Patch number.

    input_path : str or Path
        Directory containing the validation/testing patches.

        Expected files:

            image_{slide_id}_patch_{patch_nr}_tissue.png
            image_{slide_id}_patch_{patch_nr}_mask.png

    output_path : str or Path
        Directory where the predicted mask and sigmoid
        prediction map will be saved.

    model_path : str or Path
        Path to the trained .pth model.

    threshold : float
        Probability threshold used to convert the model's
        sigmoid output into a binary mask.

    Returns
    -------
    dict
        Dictionary containing:

            slide_id
            patch_nr
            dice
            sensitivity
            precision
            predicted_mask_path
            sigmoid_prediction_path
    """

    # --------------------------------------------------------
    # Convert paths
    # --------------------------------------------------------

    input_path = Path(
        input_path
    )

    output_path = Path(
        output_path
    )

    model_path = Path(
        model_path
    )

    # --------------------------------------------------------
    # Construct filenames
    # --------------------------------------------------------

    tissue_filename = (
        f"image_{slide_id}"
        f"_patch_{patch_nr}_tissue.png"
    )

    mask_filename = (
        f"image_{slide_id}"
        f"_patch_{patch_nr}_mask.png"
    )

    predicted_filename = (
        f"image_{slide_id}"
        f"_patch_{patch_nr}_predicted.png"
    )

    # NEW: sigmoid prediction filename
    sigmoid_filename = (
        f"image_{slide_id}"
        f"_patch_{patch_nr}_sigmoid.png"
    )

    tissue_path = (
        input_path
        / tissue_filename
    )

    mask_path = (
        input_path
        / mask_filename
    )

    predicted_path = (
        output_path
        / predicted_filename
    )

    # NEW: sigmoid prediction path
    sigmoid_path = (
        output_path
        / sigmoid_filename
    )

    # --------------------------------------------------------
    # Check input files
    # --------------------------------------------------------

    if not tissue_path.exists():

        raise FileNotFoundError(
            f"Tissue patch not found:\n"
            f"{tissue_path}"
        )

    if not mask_path.exists():

        raise FileNotFoundError(
            f"Ground-truth mask not found:\n"
            f"{mask_path}"
        )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = get_device()

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model(
        model_path,
        device,
    )

    # --------------------------------------------------------
    # Load tissue image
    # --------------------------------------------------------

    image = preprocess_image(
        tissue_path
    )

    image = image.to(
        device
    )

    # --------------------------------------------------------
    # Load ground truth
    # --------------------------------------------------------

    ground_truth = load_ground_truth(
        mask_path
    )

    ground_truth = ground_truth.to(
        device
    )

    # --------------------------------------------------------
    # Model prediction
    # --------------------------------------------------------

    with torch.no_grad():

        logits = model(
            image
        )

        probabilities = torch.sigmoid(
            logits
        )

        # ----------------------------------------------------
        # Save sigmoid prediction map
        #
        # The sigmoid output contains probabilities from
        # 0.0 to 1.0.
        #
        # Convert:
        #
        # 0.0 -> 0
        # 0.5 -> approximately 127
        # 1.0 -> 255
        #
        # This is saved BEFORE applying the threshold.
        # ----------------------------------------------------

        sigmoid_map = (
            probabilities[0, 0]
            .cpu()
            .numpy()
        )

        sigmoid_map = (
            sigmoid_map * 255
        ).clip(
            0,
            255,
        ).astype(
            np.uint8
        )

        sigmoid_image = Image.fromarray(
            sigmoid_map,
            mode="L",
        )

        sigmoid_image.save(
            sigmoid_path
        )

        # ----------------------------------------------------
        # Apply threshold to create binary prediction
        # ----------------------------------------------------

        predictions = (
            probabilities >= threshold
        ).float()

    # --------------------------------------------------------
    # Calculate metrics
    #
    # Reuses Losses_metrics.py
    # --------------------------------------------------------

    metrics = calculate_metrics(
        logits,
        ground_truth,
        threshold=threshold,
    )

    # --------------------------------------------------------
    # Convert prediction to PNG
    #
    # 0      -> black
    # 1      -> white
    #
    # Using 0 / 255 makes the output easy to inspect.
    # --------------------------------------------------------

    predicted_mask = (
        predictions[0, 0]
        .cpu()
        .numpy()
    )

    predicted_mask = (
        predicted_mask * 255
    ).astype(
        np.uint8
    )

    predicted_image = Image.fromarray(
        predicted_mask,
        mode="L",
    )

    predicted_image.save(
        predicted_path
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 60
    )

    print("PATCH EVALUATION")

    print("=" * 60)

    print(f"Slide ID:       {slide_id}")

    print(f"Patch number:   {patch_nr}")

    print(f"Dice:           {metrics['dice']:.4f}")

    print(f"Sensitivity:    {metrics['sensitivity']:.4f}")

    print(f"Precision:      {metrics['precision']:.4f}")

    print(f"Saved prediction:")

    print(predicted_path)

    print(f"Saved sigmoid prediction:")

    print(sigmoid_path)

    print(
        "=" * 60
    )

    # --------------------------------------------------------
    # Return everything useful for future evaluation code
    # --------------------------------------------------------

    results = {

        "slide_id":
            slide_id,

        "patch_nr":
            patch_nr,

        "dice":
            metrics["dice"],

        "sensitivity":
            metrics["sensitivity"],

        "precision":
            metrics["precision"],

        "predicted_mask_path":
            str(predicted_path),

        "sigmoid_prediction_path":
            str(sigmoid_path),
    }

    return results


## Main ##

if __name__ == "__main__":
    for slide in slide_id_list:
        for patch in patch_nr_list:
            download_predicted_mask(
                slide_id=slide,
                patch_nr=patch,
                input_path=INPUT_PATH,
                output_path=OUTPUT_PATH,
                model_path=MODEL_PATH,
            )
