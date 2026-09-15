### Model ###
# make sure it is applicable on the HPC GPU/CPU


## create initial model ##



## function that trains 1 model (1 trial) for all possible parameters ##
# Batch_size
# epochs
# encoder
# learning rate
# weight decay
# stain augmentation (true/false)
# threshold (use threshold function from train.py)
# -> summary results

## create initial model ##


## function to check runtime for different epochs, batchsize and sample size ##


## function to check performance across validation slides ##
# see if validation performance varies a lot across slides
# check which slides are difficult to segment (low performance)


# (after intuition model building parameters: epochs, batchsize and sample size are decided)

# ============================================================
# Libraries
# ============================================================

from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

import segmentation_models_pytorch as smp

from Losses_metrics import (
    BCEDiceLoss,
    calculate_metrics,
)


# ============================================================
# Configuration
# ============================================================

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

TRAIN_DIR = Path(
    r"/home/20234257/BEP_Project/External_data/training_10_80_10"
)

VALIDATION_DIR = Path(
    r"/home/20234257/BEP_Project/External_data/validation"
)

RESULTS_DIR = Path(
    r"/home/20234257/BEP_Project/BEP_blood_vessel_graft_segmentation-main/results" 
)

MODEL_DIR = (
    RESULTS_DIR / "models"
)

METRICS_DIR = (
    RESULTS_DIR / "metrics"
)

VISUALIZATION_DIR = (
    RESULTS_DIR / "visualizations"
)


# ------------------------------------------------------------
# Model parameters
# ------------------------------------------------------------

ENCODER = "resnet34"

ENCODER_WEIGHTS = "imagenet"

IN_CHANNELS = 3

CLASSES = 1


# ------------------------------------------------------------
# Training parameters
# ------------------------------------------------------------

BATCH_SIZE = 8  # use 4 for initial

EPOCHS = 50 # use 20 for initial

SAMPLE_SIZE = 10000 # use 1000 for initial

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

THRESHOLD = 0.5


# ------------------------------------------------------------
# DataLoader parameters
# ------------------------------------------------------------

NUM_WORKERS = 4 # use 0 for initial

PIN_MEMORY = True


# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------

RANDOM_SEED = 42


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed=42):
    """
    Sets random seeds for reproducibility.
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)

        # These settings improve reproducibility.
        # They can slightly reduce maximum performance.

        torch.backends.cudnn.deterministic = True

        torch.backends.cudnn.benchmark = False


# ============================================================
# Device
# ============================================================

def get_device():
    """
    Automatically selects:

        CUDA GPU
        otherwise CPU

    Returns
    -------
    torch.device
    """

    if torch.cuda.is_available():

        device = torch.device("cuda")

        print(
            "\nGPU detected:"
        )

        print(
            torch.cuda.get_device_name(0)
        )

        print(
            f"CUDA version: "
            f"{torch.version.cuda}"
        )

    else:

        device = torch.device("cpu")

        print(
            "\nNo CUDA GPU detected."
        )

        print(
            "Using CPU."
        )

    return device


# ============================================================
# Dataset
# ============================================================

class SegmentationDataset(Dataset):
    """
    Dataset for paired tissue and mask PNG files.

    Expected naming:

        image_XXXX_patch_0_tissue.png
        image_XXXX_patch_0_mask.png

    The dataset automatically searches for tissue images
    and constructs the corresponding mask path.

    Masks are converted robustly:

        black / zero      -> 0
        anything nonzero -> 1
    """

    def __init__(
        self,
        image_paths,
        mask_paths,
    ):

        if len(image_paths) != len(mask_paths):

            raise ValueError(
                "Number of images and masks do not match."
            )

        self.image_paths = list(
            image_paths
        )

        self.mask_paths = list(
            mask_paths
        )

        if len(self.image_paths) == 0:

            raise ValueError(
                "Dataset contains no images."
            )


    def __len__(self):

        return len(
            self.image_paths
        )


    def __getitem__(self, index):

        image_path = (
            self.image_paths[index]
        )

        mask_path = (
            self.mask_paths[index]
        )

        # ----------------------------------------------------
        # Load tissue image
        # ----------------------------------------------------

        image = Image.open(
            image_path
        ).convert("RGB")

        # ----------------------------------------------------
        # Load mask
        # ----------------------------------------------------

        mask = Image.open(
            mask_path
        ).convert("L")

        # ----------------------------------------------------
        # Convert image to NumPy
        # ----------------------------------------------------

        image = np.asarray(
            image,
            dtype=np.float32
        )

        # ----------------------------------------------------
        # Convert mask to NumPy
        #
        # Any nonzero value becomes foreground.
        #
        # This handles:
        #
        #     0 / 1
        #     0 / 255
        #     other nonzero grayscale values
        # ----------------------------------------------------

        mask = np.asarray(
            mask,
            dtype=np.uint8
        )

        mask = (
            mask > 0
        ).astype(
            np.float32
        )

        # ----------------------------------------------------
        # Convert image:
        #
        # H x W x C
        #
        # into:
        #
        # C x H x W
        # ----------------------------------------------------

        image = image.transpose(
            2,
            0,
            1
        )

        # ----------------------------------------------------
        # Scale image to [0, 1]
        # ----------------------------------------------------

        image = (
            image / 255.0
        )

        # ----------------------------------------------------
        # ImageNet normalization
        #
        # This is appropriate because the ResNet34 encoder
        # was pretrained on ImageNet.
        # ----------------------------------------------------

        mean = np.array(
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

        std = np.array(
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

        image = (
            image - mean
        ) / std

        # ----------------------------------------------------
        # Convert to PyTorch tensors
        # ----------------------------------------------------

        image = torch.from_numpy(
            image
        ).float()

        mask = torch.from_numpy(
            mask
        ).float()

        # ----------------------------------------------------
        # Add channel dimension to mask.
        #
        # H x W
        #
        # becomes:
        #
        # 1 x H x W
        # ----------------------------------------------------

        mask = mask.unsqueeze(
            0
        )

        return image, mask


# ============================================================
# Find training image/mask pairs
# ============================================================

def find_training_pairs(
    directory,
    sample_size=None,
    seed=42,
):
    """
    Finds tissue/mask pairs in a directory.

    If sample_size is None:
        use all available patches.

    If sample_size is smaller than the available number
    of patches:
        randomly select sample_size patches.

    The random selection is reproducible.
    """

    directory = Path(
        directory
    )

    if not directory.exists():

        raise FileNotFoundError(
            f"Training directory does not exist:\n"
            f"{directory}"
        )

    tissue_paths = sorted(
        directory.glob(
            "*_tissue.png"
        )
    )

    if len(tissue_paths) == 0:

        raise FileNotFoundError(
            f"No *_tissue.png files found in:\n"
            f"{directory}"
        )

    pairs = []

    for tissue_path in tissue_paths:

        mask_path = Path(
            str(tissue_path).replace(
                "_tissue.png",
                "_mask.png",
            )
        )

        if not mask_path.exists():

            print(
                "WARNING: mask missing for:"
            )

            print(
                tissue_path
            )

            continue

        pairs.append(
            (
                tissue_path,
                mask_path,
            )
        )

    if len(pairs) == 0:

        raise RuntimeError(
            "No valid tissue/mask pairs found."
        )

    print(
        f"Found {len(pairs)} "
        f"training image/mask pairs."
    )

    # --------------------------------------------------------
    # Select sample size
    # --------------------------------------------------------

    if sample_size is None:

        selected_pairs = pairs

    else:

        if sample_size <= 0:

            raise ValueError(
                "sample_size must be > 0 "
                "or None."
            )

        if sample_size >= len(pairs):

            selected_pairs = pairs

            print(
                f"Requested sample size "
                f"{sample_size}, but only "
                f"{len(pairs)} patches exist."
            )

            print(
                "Using all available patches."
            )

        else:

            rng = random.Random(
                seed
            )

            selected_pairs = rng.sample(
                pairs,
                sample_size
            )

    # --------------------------------------------------------
    # Split tuples into lists
    # --------------------------------------------------------

    image_paths = [
        pair[0]
        for pair in selected_pairs
    ]

    mask_paths = [
        pair[1]
        for pair in selected_pairs
    ]

    print(
        f"Using {len(image_paths)} "
        f"training patches."
    )

    return (
        image_paths,
        mask_paths,
    )


# ============================================================
# Find validation image/mask pairs
# ============================================================

def find_validation_pairs(
    directory
):
    """
    Finds all validation tissue/mask pairs.

    Validation data is NOT subsampled.

    Validation slides are separate from training slides,
    so all available validation patches are used.
    """

    directory = Path(
        directory
    )

    if not directory.exists():

        raise FileNotFoundError(
            f"Validation directory does not exist:\n"
            f"{directory}"
        )

    tissue_paths = sorted(
        directory.glob(
            "*_tissue.png"
        )
    )

    if len(tissue_paths) == 0:

        raise FileNotFoundError(
            f"No validation tissue patches found in:\n"
            f"{directory}"
        )

    image_paths = []

    mask_paths = []

    for tissue_path in tissue_paths:

        mask_path = Path(
            str(tissue_path).replace(
                "_tissue.png",
                "_mask.png",
            )
        )

        if not mask_path.exists():

            print(
                "WARNING: validation mask missing:"
            )

            print(
                tissue_path
            )

            continue

        image_paths.append(
            tissue_path
        )

        mask_paths.append(
            mask_path
        )

    if len(image_paths) == 0:

        raise RuntimeError(
            "No validation image/mask pairs found."
        )

    print(
        f"Found {len(image_paths)} "
        f"validation patches."
    )

    return (
        image_paths,
        mask_paths,
    )


# ============================================================
# Create model
# ============================================================

def create_model(
    encoder_name="resnet34",
    encoder_weights="imagenet",
):
    """
    Creates the initial U-Net model.

    Binary segmentation:
        classes = 1

    Input:
        RGB = 3 channels

    No sigmoid activation is used here.

    The model therefore returns raw logits.
    """

    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=1,
        activation=None,
    )

    return model


# ============================================================
# Create DataLoaders
# ============================================================

def create_dataloaders(
    train_dataset,
    validation_dataset,
    batch_size,
    num_workers=0,
):
    """
    Creates PyTorch DataLoaders.
    """

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(
            torch.cuda.is_available()
            and PIN_MEMORY
        ),
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(
            torch.cuda.is_available()
            and PIN_MEMORY
        ),
    )

    return (
        train_loader,
        validation_loader,
    )


# ============================================================
# Training epoch
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
):
    """
    Runs one complete training epoch.

    Returns:
        average training loss
    """

    model.train()

    running_loss = 0.0

    total_samples = 0

    for images, masks in loader:

        images = images.to(
            device,
            non_blocking=True,
        )

        masks = masks.to(
            device,
            non_blocking=True,
        )

        # ----------------------------------------------------
        # Clear gradients
        # ----------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        logits = model(
            images
        )

        # ----------------------------------------------------
        # Calculate loss
        # ----------------------------------------------------

        loss = criterion(
            logits,
            masks
        )

        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        loss.backward()

        optimizer.step()

        # ----------------------------------------------------
        # Record loss
        # ----------------------------------------------------

        current_batch_size = (
            images.size(0)
        )

        running_loss += (
            loss.item()
            * current_batch_size
        )

        total_samples += (
            current_batch_size
        )

    epoch_loss = (
        running_loss
        / total_samples
    )

    return epoch_loss


# ============================================================
# Validation epoch
# ============================================================

def validate_one_epoch(
    model,
    loader,
    criterion,
    device,
    threshold=0.5,
):
    """
    Runs validation.

    Calculates:

        validation loss
        Dice
        sensitivity
        precision

    Returns a dictionary containing these values.
    """

    model.eval()

    running_loss = 0.0

    total_samples = 0

    dice_values = []

    sensitivity_values = []

    precision_values = []

    with torch.no_grad():

        for images, masks in loader:

            images = images.to(
                device,
                non_blocking=True,
            )

            masks = masks.to(
                device,
                non_blocking=True,
            )

            # ------------------------------------------------
            # Forward pass
            # ------------------------------------------------

            logits = model(
                images
            )

            # ------------------------------------------------
            # Validation loss
            # ------------------------------------------------

            loss = criterion(
                logits,
                masks
            )

            current_batch_size = (
                images.size(0)
            )

            running_loss += (
                loss.item()
                * current_batch_size
            )

            total_samples += (
                current_batch_size
            )

            # ------------------------------------------------
            # Metrics
            # ------------------------------------------------

            metrics = calculate_metrics(
                logits,
                masks,
                threshold=threshold,
            )

            dice_values.append(
                metrics["dice"]
            )

            sensitivity_values.append(
                metrics["sensitivity"]
            )

            precision_values.append(
                metrics["precision"]
            )

    validation_loss = (
        running_loss
        / total_samples
    )

    results = {

        "loss": validation_loss,

        "dice": float(
            np.mean(
                dice_values
            )
        ),

        "sensitivity": float(
            np.mean(
                sensitivity_values
            )
        ),

        "precision": float(
            np.mean(
                precision_values
            )
        ),
    }

    return results


# ============================================================
# Evaluate training Dice
# ============================================================

def evaluate_training_dice(
    model,
    loader,
    device,
    threshold=0.5,
):
    """
    Calculates Dice on the training set after an epoch.

    No gradients are calculated.
    """

    model.eval()

    dice_values = []

    with torch.no_grad():

        for images, masks in loader:

            images = images.to(
                device,
                non_blocking=True,
            )

            masks = masks.to(
                device,
                non_blocking=True,
            )

            logits = model(
                images
            )

            metrics = calculate_metrics(
                logits,
                masks,
                threshold=threshold,
            )

            dice_values.append(
                metrics["dice"]
            )

    return float(
        np.mean(
            dice_values
        )
    )


# ============================================================
# Save training curves
# ============================================================

def plot_training_history(
    history,
    output_path,
):
    """
    Creates a two-panel training graph.

    Panel 1:
        training vs validation loss

    Panel 2:
        training vs validation Dice
    """

    epochs = history["epoch"]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5),
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    axes[0].plot(
        epochs,
        history["train_loss"],
        marker="o",
        label="Training loss",
    )

    axes[0].plot(
        epochs,
        history["val_loss"],
        marker="o",
        label="Validation loss",
    )

    axes[0].set_xlabel(
        "Epoch"
    )

    axes[0].set_ylabel(
        "Loss"
    )

    axes[0].set_title(
        "Training and Validation Loss"
    )

    axes[0].legend()

    axes[0].grid(
        alpha=0.3
    )

    # --------------------------------------------------------
    # Dice
    # --------------------------------------------------------

    axes[1].plot(
        epochs,
        history["train_dice"],
        marker="o",
        label="Training Dice",
    )

    axes[1].plot(
        epochs,
        history["val_dice"],
        marker="o",
        label="Validation Dice",
    )

    axes[1].set_xlabel(
        "Epoch"
    )

    axes[1].set_ylabel(
        "Dice"
    )

    axes[1].set_title(
        "Training and Validation Dice"
    )

    axes[1].set_ylim(
        0,
        1
    )

    axes[1].legend()

    axes[1].grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# Main training function
# ============================================================

def run_initial_experiment(
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    sample_size=SAMPLE_SIZE,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    threshold=THRESHOLD,
):
    """
    Runs one complete baseline experiment.
    """
    # --------------------------------------------------------
    # Experiment name
    # --------------------------------------------------------

    experiment_name = (
        f"n{sample_size}"
        f"_bs{batch_size}"
        f"_ep{epochs}"
    )
    
    # --------------------------------------------------------
    # Create output directories
    # --------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    VISUALIZATION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    set_seed(
        RANDOM_SEED
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = get_device()

    print(
        f"\nUsing device: {device}"
    )

    # --------------------------------------------------------
    # Training data
    # --------------------------------------------------------

    (
        train_image_paths,
        train_mask_paths,
    ) = find_training_pairs(
        TRAIN_DIR,
        sample_size=sample_size,
        seed=RANDOM_SEED,
    )

    # --------------------------------------------------------
    # Validation data
    # --------------------------------------------------------

    (
        val_image_paths,
        val_mask_paths,
    ) = find_validation_pairs(
        VALIDATION_DIR
    )

    # --------------------------------------------------------
    # Create datasets
    # --------------------------------------------------------

    train_dataset = SegmentationDataset(
        train_image_paths,
        train_mask_paths,
    )

    validation_dataset = SegmentationDataset(
        val_image_paths,
        val_mask_paths,
    )

    # --------------------------------------------------------
    # Create DataLoaders
    # --------------------------------------------------------

    (
        train_loader,
        validation_loader,
    ) = create_dataloaders(
        train_dataset,
        validation_dataset,
        batch_size=batch_size,
        num_workers=NUM_WORKERS,
    )

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    print(
        "\nCreating model..."
    )

    model = create_model(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
    )

    model = model.to(
        device
    )

    # --------------------------------------------------------
    # Number of parameters
    # --------------------------------------------------------

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"Total parameters: "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_parameters:,}"
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = BCEDiceLoss(
        bce_weight=0.5,
        dice_weight=0.5,
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # --------------------------------------------------------
    # Training history
    # --------------------------------------------------------

    history = {

        "epoch": [],

        "train_loss": [],

        "val_loss": [],

        "train_dice": [],

        "val_dice": [],

        "val_sensitivity": [],

        "val_precision": [],
    }

    # --------------------------------------------------------
    # Best model tracking
    # --------------------------------------------------------

    best_validation_dice = -1.0

    best_epoch = None

    best_model_path = (
        MODEL_DIR
        / f"{experiment_name}_10_80_10_best.pth" 
    )

    # --------------------------------------------------------
    # Start experiment timer
    # --------------------------------------------------------

    experiment_start = time.time()

    print(
        "\n"
        + "=" * 70
    )

    print(
        "STARTING INITIAL EXPERIMENT"
    )

    print(
        "=" * 70
    )

    print(
        f"Encoder:          {ENCODER}"
    )

    print(
        f"Pretrained:       {ENCODER_WEIGHTS}"
    )

    print(
        f"Training patches: {len(train_dataset)}"
    )

    print(
        f"Validation patches: "
        f"{len(validation_dataset)}"
    )

    print(
        f"Batch size:       {batch_size}"
    )

    print(
        f"Epochs:            {epochs}"
    )

    print(
        f"Learning rate:     {learning_rate}"
    )

    print(
        f"Weight decay:      {weight_decay}"
    )

    print(
        f"Threshold:         {threshold}"
    )

    print(
        "=" * 70
    )

    # ========================================================
    # Epoch loop
    # ========================================================

    for epoch in range(
        1,
        epochs + 1
    ):

        epoch_start = time.time()

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )

        # ----------------------------------------------------
        # Training Dice
        # ----------------------------------------------------

        train_dice = evaluate_training_dice(
            model,
            train_loader,
            device,
            threshold=threshold,
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        validation_results = (
            validate_one_epoch(
                model,
                validation_loader,
                criterion,
                device,
                threshold=threshold,
            )
        )

        val_loss = (
            validation_results["loss"]
        )

        val_dice = (
            validation_results["dice"]
        )

        val_sensitivity = (
            validation_results["sensitivity"]
        )

        val_precision = (
            validation_results["precision"]
        )

        # ----------------------------------------------------
        # Epoch runtime
        # ----------------------------------------------------

        epoch_time = (
            time.time()
            - epoch_start
        )

        # ----------------------------------------------------
        # Store results
        # ----------------------------------------------------

        history["epoch"].append(
            epoch
        )

        history["train_loss"].append(
            train_loss
        )

        history["val_loss"].append(
            val_loss
        )

        history["train_dice"].append(
            train_dice
        )

        history["val_dice"].append(
            val_dice
        )

        history["val_sensitivity"].append(
            val_sensitivity
        )

        history["val_precision"].append(
            val_precision
        )

        # ----------------------------------------------------
        # Print epoch results
        # ----------------------------------------------------

        print(
            f"\nEpoch "
            f"{epoch:03d}/{epochs:03d}"
        )

        print(
            f"  Train loss: "
            f"{train_loss:.4f}"
        )

        print(
            f"  Val loss:   "
            f"{val_loss:.4f}"
        )

        print(
            f"  Train Dice: "
            f"{train_dice:.4f}"
        )

        print(
            f"  Val Dice:   "
            f"{val_dice:.4f}"
        )

        print(
            f"  Val sens.:  "
            f"{val_sensitivity:.4f}"
        )

        print(
            f"  Val prec.:  "
            f"{val_precision:.4f}"
        )

        print(
            f"  Runtime:    "
            f"{epoch_time:.2f} s"
        )

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if val_dice > best_validation_dice:

            best_validation_dice = (
                val_dice
            )

            best_epoch = epoch

            torch.save(
                {
                    "epoch": epoch,

                    "model_state_dict":
                        model.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "validation_dice":
                        val_dice,

                    "encoder":
                        ENCODER,

                    "encoder_weights":
                        ENCODER_WEIGHTS,

                    "batch_size":
                        batch_size,

                    "learning_rate":
                        learning_rate,

                    "weight_decay":
                        weight_decay,

                    "threshold":
                        threshold,
                },
                best_model_path,
            )

            print(
                "  -> New best model saved."
            )

    # ========================================================
    # Experiment finished
    # ========================================================

    total_runtime = (
        time.time()
        - experiment_start
    )

    # --------------------------------------------------------
    # Save training history
    # --------------------------------------------------------

    history_df = pd.DataFrame(
        history
    )

    history_path = (
        METRICS_DIR
        / f"{experiment_name}_10_80_10_training_history.csv"
    )

    history_df.to_csv(
        history_path,
        index=False,
    )

    # --------------------------------------------------------
    # Save training curves
    # --------------------------------------------------------

    plot_path = (
        VISUALIZATION_DIR
        / f"{experiment_name}_10_80_10_training_curves.png"
    )

    plot_training_history(
        history,
        plot_path,
    )

    # --------------------------------------------------------
    # Find best epoch
    # --------------------------------------------------------

    best_index = (
        history_df["val_dice"].idxmax()
    )

    best_row = (
        history_df.loc[best_index]
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    summary = {

        "encoder":
            ENCODER,

        "encoder_weights":
            ENCODER_WEIGHTS,

        "sample_size":
            len(train_dataset),

        "validation_size":
            len(validation_dataset),

        "batch_size":
            batch_size,

        "epochs":
            epochs,

        "learning_rate":
            learning_rate,

        "weight_decay":
            weight_decay,

        "threshold":
            threshold,

        "best_epoch":
            int(
                best_row["epoch"]
            ),

        "best_val_dice":
            float(
                best_row["val_dice"]
            ),

        "best_val_loss":
            float(
                best_row["val_loss"]
            ),

        "best_val_sensitivity":
            float(
                best_row[
                    "val_sensitivity"
                ]
            ),

        "best_val_precision":
            float(
                best_row[
                    "val_precision"
                ]
            ),

        "total_runtime_seconds":
            total_runtime,
    }

    summary_df = pd.DataFrame(
        [summary]
    )

    summary_path = (
        METRICS_DIR
        / f"{experiment_name}_10_80_10_training_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    # --------------------------------------------------------
    # Print final summary
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print("EXPERIMENT FINISHED")

    print("=" * 70)

    print(
        f"Best epoch:           "
        f"{best_epoch}"
    )

    print(
        f"Best validation Dice: "
        f"{best_validation_dice:.4f}"
    )

    print(
        f"Total runtime:        "
        f"{total_runtime / 60:.2f} minutes"
    )

    print("\nBest model:")
    print(best_model_path)

    print("\nTraining history:")
    print(history_path)

    print("\nTraining curves:")
    print(plot_path)

    print("\nSummary:")
    print(summary_path)

    print("=" * 70)

    return {
        "model": model,

        "history": history,

        "summary": summary,

        "best_model_path":
            best_model_path,
    }


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    run_initial_experiment()
