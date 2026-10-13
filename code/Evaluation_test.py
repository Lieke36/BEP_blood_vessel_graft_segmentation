### Evaluation test ###
# ============================================================
# Evaluate a trained segmentation model on the test set
# Standalone: does not import Model.py
# ============================================================

from pathlib import Path
import re
import time

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

import segmentation_models_pytorch as smp

from Losses_metrics import BCEDiceLoss, calculate_metrics


# ============================================================
# Configuration: PATHS ONLY
# ============================================================

MODEL_PATH = Path(
    "/home/20234257/BEP_Project/BEP_blood_vessel_graft_segmentation-main/results_full/models/n28000_bs8_ep50_10_80_10_best.pth"
)

TRAINING_SUMMARY_PATH = Path(
    "/home/20234257/BEP_Project/BEP_blood_vessel_graft_segmentation-main/results_full/metricsn28000_bs8_ep50_10_80_10_training_summary.csv"
)

TEST_DIR = Path(
    "/home/20234257/BEP_Project/External_data/testing"
)

SUMMARY_PATH = Path(
    "/home/20234257/BEP_Project/BEP_blood_vessel_graft_segmentation-main/results_full/metrics/testing_summary.csv"
)

SLIDE_SUMMARY_PATH = Path(
    "/home/20234257/BEP_Project/BEP_blood_vessel_graft_segmentation-main/results_full/metrics/testing_summary_slides.csv"
)


# ============================================================
# Evaluation settings
# ============================================================

BATCH_SIZE = 8
NUM_WORKERS = 4
PIN_MEMORY = True
DEFAULT_THRESHOLD = 0.5


# ============================================================
# Dataset
# ============================================================

class TestDataset(Dataset):
    """Load paired tissue images and binary masks."""

    def __init__(self, image_paths, mask_paths):
        if len(image_paths) != len(mask_paths):
            raise ValueError("Image/mask counts do not match.")

        if not image_paths:
            raise ValueError("The test dataset is empty.")

        self.image_paths = list(image_paths)
        self.mask_paths = list(mask_paths)

        self.mean = np.array(
            [0.485, 0.456, 0.406],
            dtype=np.float32,
        ).reshape(3, 1, 1)

        self.std = np.array(
            [0.229, 0.224, 0.225],
            dtype=np.float32,
        ).reshape(3, 1, 1)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(
            self.image_paths[index]
        ).convert("RGB")

        mask = Image.open(
            self.mask_paths[index]
        ).convert("L")

        image = np.asarray(
            image, dtype=np.float32
        ).transpose(2, 0, 1) / 255.0

        image = (image - self.mean) / self.std

        mask = np.asarray(
            mask, dtype=np.uint8
        )

        mask = (mask > 0).astype(np.float32)

        image = torch.from_numpy(
            np.ascontiguousarray(image)
        ).float()

        mask = torch.from_numpy(
            np.ascontiguousarray(mask)
        ).float().unsqueeze(0)

        return image, mask


# ============================================================
# Find test image/mask pairs
# ============================================================

def find_test_pairs(directory):
    directory = Path(directory)

    if not directory.is_dir():
        raise FileNotFoundError(
            f"Test directory not found: {directory}"
        )

    tissue_paths = sorted(
        directory.glob("*_tissue.png")
    )

    if not tissue_paths:
        raise FileNotFoundError(
            f"No *_tissue.png files found in {directory}"
        )

    image_paths = []
    mask_paths = []
    missing_masks = []

    for image_path in tissue_paths:
        mask_path = image_path.with_name(
            image_path.name.replace(
                "_tissue.png",
                "_mask.png",
            )
        )

        if not mask_path.exists():
            missing_masks.append(image_path.name)
            continue

        image_paths.append(image_path)
        mask_paths.append(mask_path)

    if missing_masks:
        print(
            f"WARNING: {len(missing_masks)} images "
            "have no matching mask and will be skipped."
        )

        for name in missing_masks[:10]:
            print(f"  {name}")

    if not image_paths:
        raise RuntimeError(
            "No valid test image/mask pairs found."
        )

    print(f"Test patches: {len(image_paths)}")

    return image_paths, mask_paths


# ============================================================
# Extract slide ID
# ============================================================

def extract_slide_id(image_path):
    filename = Path(image_path).name

    match = re.match(
        r"^image_(.+?)_patch_\d+_tissue\.png$",
        filename,
    )

    if match is None:
        raise ValueError(
            f"Cannot extract slide ID from {filename}"
        )

    return match.group(1)


# ============================================================
# Load configuration from checkpoint and training CSV
# ============================================================

def load_configuration(model_path, summary_path):
    """
    The checkpoint is the primary configuration source.
    The training summary provides fallback values.
    """

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {model_path}"
        )

    if not summary_path.is_file():
        raise FileNotFoundError(
            f"Training summary not found: {summary_path}"
        )

    summary_df = pd.read_csv(summary_path)

    if summary_df.empty:
        raise ValueError("Training summary CSV is empty.")

    summary = summary_df.iloc[0].to_dict()

    # Only load checkpoints from trusted sources.
    try:
        checkpoint = torch.load(
            model_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(
            model_path,
            map_location="cpu",
        )

    encoder = checkpoint.get(
        "encoder",
        summary.get("encoder", "resnet34"),
    )

    threshold = checkpoint.get(
        "threshold",
        summary.get("threshold", DEFAULT_THRESHOLD),
    )

    encoder = str(encoder)
    threshold = float(threshold)

    if not 0.0 < threshold < 1.0:
        raise ValueError(
            f"Invalid threshold: {threshold}"
        )

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain model_state_dict."
        )

    return checkpoint, encoder, threshold, summary


# ============================================================
# Load model
# ============================================================

def load_model(checkpoint, encoder, device):
    """
    Reconstruct the U-Net and load the trained parameters.
    No pretrained weights are downloaded.
    """

    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model = model.to(device)
    model.eval()

    return model


# ============================================================
# Patch classification and Dice
# ============================================================

def calculate_patch_dice(prediction, mask):
    intersection = (prediction * mask).sum()
    denominator = prediction.sum() + mask.sum()

    if denominator.item() == 0:
        return 1.0

    return (
        2.0 * intersection / denominator
    ).item()


def classify_patch(mask):
    if mask.max().item() == 0:
        return "background"

    if mask.min().item() == 1:
        return "foreground"

    return "edge"


# ============================================================
# Evaluate overall test set
# ============================================================

def evaluate_overall(
    model,
    loader,
    criterion,
    device,
    threshold,
):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    dice_values = []
    sensitivity_values = []
    precision_values = []

    start_time = time.time()

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

            logits = model(images)
            loss = criterion(logits, masks)

            batch_size = images.size(0)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            metrics = calculate_metrics(
                logits,
                masks,
                threshold=threshold,
            )

            dice_values.append(metrics["dice"])
            sensitivity_values.append(
                metrics["sensitivity"]
            )
            precision_values.append(
                metrics["precision"]
            )

    if total_samples == 0:
        raise RuntimeError("No test patches were evaluated.")

    return {
        "test_size": total_samples,
        "test_loss": total_loss / total_samples,
        "test_dice": float(np.mean(dice_values)),
        "test_sensitivity": float(
            np.mean(sensitivity_values)
        ),
        "test_precision": float(
            np.mean(precision_values)
        ),
        "test_runtime_seconds": time.time() - start_time,
    }


# ============================================================
# Evaluate metrics per slide
# ============================================================

def evaluate_by_slide(
    model,
    image_paths,
    mask_paths,
    criterion,
    device,
    threshold,
):
    slides = {}

    for image_path, mask_path in zip(
        image_paths, mask_paths
    ):
        slide_id = extract_slide_id(image_path)

        slides.setdefault(
            slide_id, []
        ).append((image_path, mask_path))

    results = []

    model.eval()

    with torch.no_grad():
        for slide_id in sorted(slides):
            pairs = slides[slide_id]

            dataset = TestDataset(
                [pair[0] for pair in pairs],
                [pair[1] for pair in pairs],
            )

            loader = DataLoader(
                dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=NUM_WORKERS,
                pin_memory=(
                    torch.cuda.is_available()
                    and PIN_MEMORY
                ),
            )

            losses = []
            dices = []
            sensitivities = []
            precisions = []

            background_dice = []
            edge_dice = []
            foreground_dice = []

            for images, masks in loader:
                images = images.to(
                    device,
                    non_blocking=True,
                )

                masks = masks.to(
                    device,
                    non_blocking=True,
                )

                logits = model(images)
                loss = criterion(logits, masks)

                metrics = calculate_metrics(
                    logits,
                    masks,
                    threshold=threshold,
                )

                losses.append(loss.item())
                dices.append(metrics["dice"])
                sensitivities.append(
                    metrics["sensitivity"]
                )
                precisions.append(
                    metrics["precision"]
                )

                predictions = (
                    torch.sigmoid(logits) >= threshold
                ).float()

                for prediction, mask in zip(
                    predictions, masks
                ):
                    patch_type = classify_patch(mask)

                    patch_dice = calculate_patch_dice(
                        prediction, mask
                    )

                    if patch_type == "background":
                        background_dice.append(patch_dice)
                    elif patch_type == "foreground":
                        foreground_dice.append(patch_dice)
                    else:
                        edge_dice.append(patch_dice)

            results.append({
                "slide_id": slide_id,
                "number_of_patches": len(dataset),
                "loss": float(np.mean(losses)),
                "dice": float(np.mean(dices)),
                "sensitivity": float(np.mean(sensitivities)),
                "precision": float(np.mean(precisions)),
                "dice_background": (
                    float(np.mean(background_dice))
                    if background_dice else np.nan
                ),
                "dice_edge": (
                    float(np.mean(edge_dice))
                    if edge_dice else np.nan
                ),
                "dice_foreground": (
                    float(np.mean(foreground_dice))
                    if foreground_dice else np.nan
                ),
            })

            print(
                f"Slide {slide_id}: "
                f"Dice = {results[-1]['dice']:.4f}"
            )

    return pd.DataFrame(results)


# ============================================================
# Main
# ============================================================

def main():
    SUMMARY_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    SLIDE_SUMMARY_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    checkpoint, encoder, threshold, training_summary = (
        load_configuration(
            MODEL_PATH,
            TRAINING_SUMMARY_PATH,
        )
    )

    print(f"Encoder: {encoder}")
    print(f"Threshold: {threshold}")
    print(f"Checkpoint: {MODEL_PATH}")

    image_paths, mask_paths = find_test_pairs(TEST_DIR)

    dataset = TestDataset(image_paths, mask_paths)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(
            torch.cuda.is_available() and PIN_MEMORY
        ),
    )

    model = load_model(
        checkpoint,
        encoder,
        device,
    )

    criterion = BCEDiceLoss(
        bce_weight=0.5,
        dice_weight=0.5,
    )

    print("\nEvaluating overall test set...")

    overall = evaluate_overall(
        model,
        loader,
        criterion,
        device,
        threshold,
    )

    print("\nEvaluating individual slides...")

    slide_df = evaluate_by_slide(
        model,
        image_paths,
        mask_paths,
        criterion,
        device,
        threshold,
    )

    slide_df.to_csv(
        SLIDE_SUMMARY_PATH,
        index=False,
    )

    summary = {
        "encoder": encoder,
        "encoder_weights": training_summary.get(
            "encoder_weights", np.nan
        ),
        "test_size": overall["test_size"],
        "number_of_slides": len(slide_df),
        "threshold": threshold,
        "checkpoint": str(MODEL_PATH),
        "test_loss": overall["test_loss"],
        "test_dice": overall["test_dice"],
        "test_sensitivity": overall["test_sensitivity"],
        "test_precision": overall["test_precision"],
        "test_dice_background": float(
            slide_df["dice_background"].mean()
        ),
        "test_dice_edge": float(
            slide_df["dice_edge"].mean()
        ),
        "test_dice_foreground": float(
            slide_df["dice_foreground"].mean()
        ),
        "test_runtime_seconds": overall[
            "test_runtime_seconds"
        ],
    }

    pd.DataFrame([summary]).to_csv(
        SUMMARY_PATH,
        index=False,
    )

    print("\n" + "=" * 60)
    print("TEST EVALUATION COMPLETE")
    print("=" * 60)

    for key in [
        "test_size",
        "test_loss",
        "test_dice",
        "test_sensitivity",
        "test_precision",
        "test_dice_background",
        "test_dice_edge",
        "test_dice_foreground",
    ]:
        print(f"{key}: {summary[key]}")

    print(f"\nSummary CSV: {SUMMARY_PATH}")
    print(f"Slide CSV:   {SLIDE_SUMMARY_PATH}")


if __name__ == "__main__":
    main()