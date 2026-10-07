### full slide evaluate ###
# (for model 3 - DINO ResNet-50)

from pathlib import Path
import csv
import numpy as np
from PIL import Image
import torch
import segmentation_models_pytorch as smp

from PIL import Image
Image.MAX_IMAGE_PIXELS = None


ENCODER = "resnet50"
IN_CHANNELS = 3
CLASSES = 1
PATCH_SIZE = 512
THRESHOLD = 0.5
BATCH_SIZE = 8

IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("CUDA not available. Using CPU.")
    return device


def create_model():
    print("Loading pretrained DINO v1 ResNet-50 weights from Torch Hub...")

    dino_backbone = torch.hub.load(
        "facebookresearch/dino:main",
        "dino_resnet50",
    )

    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=None,
        in_channels=IN_CHANNELS,
        classes=CLASSES,
        activation=None,
    )

    model.encoder.load_state_dict(
        dino_backbone.state_dict(),
        strict=False,
    )

    return model


def load_model(model_path, device):
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found:\n{model_path}"
        )

    model = create_model()

    checkpoint = torch.load(
        model_path,
        map_location=device,
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
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
        model.load_state_dict(
            checkpoint
        )

    model.to(device)
    model.eval()

    return model


def preprocess_image(image):
    image = np.asarray(
        image.convert("RGB"),
        dtype=np.float32,
    )

    image = image.transpose(
        2,
        0,
        1,
    ) / 255.0

    image = (
        image - IMAGE_MEAN
    ) / IMAGE_STD

    return torch.from_numpy(
        image
    ).float()


def read_patch_csv(csv_path, slide_id):
    """Reads patch coordinates belonging to one slide."""

    patches = []

    with open(
        csv_path,
        "r",
        newline="",
        encoding="utf-8",
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            if int(row["slide_id"]) == int(slide_id):

                patches.append(
                    {
                        "patch_id": row["patch_id"],
                        "x": int(row["x"]),
                        "y": int(row["y"]),
                        "patch_path": row["tissue_path"],
                    }
                )

    if not patches:
        raise ValueError(
            f"No patches for slide {slide_id} found in {csv_path}"
        )

    # Sort by patch number so reconstruction follows the extraction order.
    patches.sort(
        key=lambda p: int(
            p["patch_id"].split("_")[-1]
        )
    )

    return patches


def predict_full_slide(
    slide_id,
    slide_path,
    patch_dir,
    patch_csv,
    output_path,
    model_path,
    threshold=THRESHOLD,
    batch_size=BATCH_SIZE,
):
    """
    Predicts a complete slide from 512x512 patches and reconstructs
    the thresholded predictions into one full-slide binary mask.

    The patch coordinates stored by full_patch_extraction() are used
    to place every prediction back at its original slide location.
    """

    slide_path = Path(slide_path)
    patch_dir = Path(patch_dir)
    patch_csv = Path(patch_csv)
    output_path = Path(output_path)
    model_path = Path(model_path)

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not slide_path.exists():
        raise FileNotFoundError(
            f"Slide not found:\n{slide_path}"
        )

    # Read original slide dimensions.
    with Image.open(slide_path) as slide_image:
        slide_width, slide_height = slide_image.size

    print(f"\nPredicting slide {slide_id}")
    print(f"Slide size: {slide_width} x {slide_height}")

    # Load patch locations from full_patch_extraction().
    patches = read_patch_csv(
        patch_csv,
        slide_id,
    )

    print(f"Number of patches: {len(patches)}")

    device = get_device()
    model = load_model(
        model_path,
        device,
    )

    # Create the final binary prediction map.
    # 0 = background, 255 = predicted foreground.
    prediction_map = np.zeros(
        (
            slide_height,
            slide_width,
        ),
        dtype=np.uint8,
    )

    # Process patches in batches.
    for batch_start in range(
        0,
        len(patches),
        batch_size,
    ):

        batch = patches[
            batch_start:batch_start + batch_size
        ]

        images = []
        valid_patches = []

        for patch in batch:

            patch_path = (
                patch_dir
                / patch["patch_path"]
            )

            if not patch_path.exists():
                raise FileNotFoundError(
                    f"Patch not found:\n{patch_path}"
                )

            with Image.open(patch_path) as image:
                image_tensor = preprocess_image(
                    image
                )

            images.append(
                image_tensor
            )

            valid_patches.append(
                patch
            )

        images = torch.stack(
            images
        ).to(device)

        # Model prediction.
        with torch.no_grad():

            logits = model(images)

            probabilities = torch.sigmoid(
                logits
            )

            predictions = (
                probabilities >= threshold
            ).squeeze(
                1
            ).cpu().numpy()

        # Place each binary prediction back into the full-slide map.
        for patch, prediction in zip(
            valid_patches,
            predictions,
        ):

            x = patch["x"]
            y = patch["y"]

            prediction_map[
                y:y + PATCH_SIZE,
                x:x + PATCH_SIZE,
            ] = (
                prediction.astype(
                    np.uint8
                ) * 255
            )

        processed = min(
            batch_start + batch_size,
            len(patches),
        )

        print(
            f"Processed {processed}/{len(patches)} patches",
            end="\r",
        )

    print()

    # Save the complete binary prediction map.
    output_file = (
        output_path
        / f"image_{slide_id}_model_3_random_predicted.png" # CHANGE #
    )

    Image.fromarray(
        prediction_map,
        mode="L",
    ).save(
        output_file
    )

    print(
        f"Saved full-slide prediction:\n"
        f"{output_file}"
    )

    return output_file


if __name__ == "__main__":

    # ---------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------

    SLIDE_ID_LIST = [10251, 11084, 11273, 11712, 80655]

    SLIDES_DIR = Path(
        r"/home/20234257/BEP_Project/External_data/slides"
    )

    PATCH_DIR = Path(
        r"/home/20234257/BEP_Project/External_data/validation"
    )

    PATCH_CSV = Path(
        r"data/validation/validation_tissue_patches.csv"
    )

    OUTPUT_PATH = Path(
        r"/home/20234257/BEP_Project/BEP_blood_vessel_graft_segmentation-main/results_full/full_slide_predictions"
    )

    MODEL_PATH = Path(
        r"/home/20234257/BEP_Project/BEP_blood_vessel_graft_segmentation-main/results_full/models/dino_resnet50_n28000_bs8_ep50_random_DINO_best.pth" # CHANGE #
    )

    # ---------------------------------------------------------
    # Predict each complete slide
    # ---------------------------------------------------------

    for slide_id in SLIDE_ID_LIST:

        predict_full_slide(
            slide_id=slide_id,
            slide_path=(
                SLIDES_DIR
                / f"image_{slide_id}_tissue.png"
            ),
            patch_dir=PATCH_DIR,
            patch_csv=PATCH_CSV,
            output_path=OUTPUT_PATH,
            model_path=MODEL_PATH,
            threshold=THRESHOLD,
            batch_size=BATCH_SIZE,
        )
