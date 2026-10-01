### Model DPT ###


# ============================================================
# Model_DPT.py - Custom ViT-DPT Implementation (Option 1: 512x512)
# ============================================================

from pathlib import Path
import random
import time
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import timm

from Losses_metrics import (
    BCEDiceLoss,
    calculate_metrics,
)


# ============================================================
# Configuration
# ============================================================

TRAIN_DIR = Path(r"/home/20234257/BEP_Project/External_data/training_34_33_33") # CHANGE #
VALIDATION_DIR = Path(r"/home/20234257/BEP_Project/External_data/validation")
RESULTS_DIR = Path(r"/home/20234257/BEP_Project/BEP_blood_vessel_graft_segmentation-main/results_full")

MODEL_DIR = RESULTS_DIR / "models"
METRICS_DIR = RESULTS_DIR / "metrics"
VISUALIZATION_DIR = RESULTS_DIR / "visualizations"

# Custom ViT-DPT Configuration
ENCODER = "vit_small_patch14_dinov2.lvd142m"
ENCODER_WEIGHTS = True
CLASSES = 1
IMG_SIZE = 512  # Native 512x512 input size (automatically padded to 532x532 internally)

BATCH_SIZE = 8
EPOCHS = 50
SAMPLE_SIZE = 10000
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
THRESHOLD = 0.5

NUM_WORKERS = 4
PIN_MEMORY = True
RANDOM_SEED = 42


# ============================================================
# Model Architecture (Custom ViT-DPT)
# ============================================================

class CustomViTDPT(nn.Module):
    """
    DPT architecture with dynamic reflection padding for ViT backbones (DINOv2).
    Pads 512x512 inputs to 532x532 internally for 14px patch division, then
    crops the output mask back to exact 512x512 dimensions.
    """
    def __init__(self, encoder_name=ENCODER, out_channels=CLASSES, pretrained=ENCODER_WEIGHTS):
        super().__init__()
        
        # Load timm ViT backbone with dynamic image size support
        model_key = encoder_name.replace("tu-", "")
        self.encoder = timm.create_model(
            model_key,
            pretrained=pretrained,
            dynamic_img_size=True,
            features_only=False
        )
        
        embed_dim = self.encoder.embed_dim  # 384 for ViT-Small
        
        # 1x1 Convolutions to project token channels from embed_dim -> 256
        self.proj1 = nn.Sequential(nn.Conv2d(embed_dim, 256, kernel_size=1), nn.BatchNorm2d(256), nn.GELU())
        self.proj2 = nn.Sequential(nn.Conv2d(embed_dim, 256, kernel_size=1), nn.BatchNorm2d(256), nn.GELU())
        self.proj3 = nn.Sequential(nn.Conv2d(embed_dim, 256, kernel_size=1), nn.BatchNorm2d(256), nn.GELU())
        self.proj4 = nn.Sequential(nn.Conv2d(embed_dim, 256, kernel_size=1), nn.BatchNorm2d(256), nn.GELU())
        
        # Decoder Refinement Block
        self.decoder = nn.Sequential(
            nn.Conv2d(256 * 4, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, out_channels, kernel_size=1)
        )

    def forward(self, x):
        orig_H, orig_W = x.shape[-2:]  # (512, 512)
        
        # Reflection padding for 14px patch division (512 -> 518)
        pad_h = (14 - (orig_H % 14)) % 14
        pad_w = (14 - (orig_W % 14)) % 14
        
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
            
        padded_H, padded_W = x.shape[-2:]
        
        # Extract intermediate transformer feature layers [3, 5, 7, 11]
        intermediates = self.encoder.get_intermediate_layers(x, n=[3, 5, 7, 11], reshape=True)
        f1, f2, f3, f4 = intermediates
        
        # Channel projections
        p1 = self.proj1(f1)
        p2 = self.proj2(f2)
        p3 = self.proj3(f3)
        p4 = self.proj4(f4)
        
        # Target feature map size (H/4, W/4)
        target_stage_size = (padded_H // 4, padded_W // 4)
        
        p1 = F.interpolate(p1, size=target_stage_size, mode="bilinear", align_corners=False)
        p2 = F.interpolate(p2, size=target_stage_size, mode="bilinear", align_corners=False)
        p3 = F.interpolate(p3, size=target_stage_size, mode="bilinear", align_corners=False)
        p4 = F.interpolate(p4, size=target_stage_size, mode="bilinear", align_corners=False)
        
        # Concatenation and decoding
        fused = torch.cat([p1, p2, p3, p4], dim=1)
        decoded = self.decoder(fused)
        
        # Upsample back to padded input size
        out = F.interpolate(decoded, size=(padded_H, padded_W), mode="bilinear", align_corners=False)
        
        # Crop back to exact original input resolution (512, 512)
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :orig_H, :orig_W]
            
        return out


# ============================================================
# Reproducibility & Device Setup
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("\nGPU detected:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("\nNo CUDA GPU detected. Using CPU.")
    return device


# ============================================================
# Dataset
# ============================================================

class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, img_size=IMG_SIZE):
        if len(image_paths) != len(mask_paths):
            raise ValueError("Number of images and masks do not match.")
        self.image_paths = list(image_paths)
        self.mask_paths = list(mask_paths)
        self.img_size = img_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert("RGB")
        mask = Image.open(self.mask_paths[index]).convert("L")

        # Resize to native 512x512 resolution
        image = image.resize((self.img_size, self.img_size), resample=Image.Resampling.BILINEAR)
        mask = mask.resize((self.img_size, self.img_size), resample=Image.Resampling.NEAREST)

        image = np.asarray(image, dtype=np.float32)
        mask = np.asarray(mask, dtype=np.uint8)
        mask = (mask > 0).astype(np.float32)

        image = image.transpose(2, 0, 1) / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
        image = (image - mean) / std

        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).float().unsqueeze(0)

        return image, mask


# ============================================================
# Helper Functions
# ============================================================

def find_pairs(directory, sample_size=None, seed=42):
    directory = Path(directory)
    tissue_paths = sorted(directory.glob("*_tissue.png"))

    pairs = []
    for tissue_path in tissue_paths:
        mask_path = Path(str(tissue_path).replace("_tissue.png", "_mask.png"))
        if mask_path.exists():
            pairs.append((tissue_path, mask_path))

    if sample_size and sample_size < len(pairs):
        rng = random.Random(seed)
        pairs = rng.sample(pairs, sample_size)

    image_paths = [p[0] for p in pairs]
    mask_paths = [p[1] for p in pairs]
    return image_paths, mask_paths


def extract_slide_id(image_path):
    filename = Path(image_path).name

    pattern = r"^image_(.+?)_patch_\d+_tissue\.png$"

    match = re.match(pattern, filename)

    if match is None:
        raise ValueError(
            "Could not extract slide ID from filename:\n"
            f"{filename}\n\n"
            "Expected format:\n"
            "image_{slide_id}_patch_{patch_nr}_tissue.png"
        )

    return match.group(1)


def create_model(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS):
    model = CustomViTDPT(
        encoder_name=encoder_name,
        out_channels=CLASSES,
        pretrained=encoder_weights
    )
    return model


def create_dataloaders(train_dataset, validation_dataset, batch_size, num_workers=NUM_WORKERS):
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(torch.cuda.is_available() and PIN_MEMORY),
        persistent_workers=(num_workers > 0),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(torch.cuda.is_available() and PIN_MEMORY),
        persistent_workers=(num_workers > 0),
    )
    return train_loader, validation_loader


def evaluate_training_dice(model, loader, device, threshold=0.5):
    model.eval()
    dice_values = []
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            logits = model(images)
            metrics = calculate_metrics(logits, masks, threshold=threshold)
            dice_values.append(metrics["dice"])
    return float(np.mean(dice_values))


def classify_mask_batch(masks):
    classifications = []
    for mask in masks:
        mask_min = mask.min().item()
        mask_max = mask.max().item()
        if mask_max == 0:
            patch_type = "background"
        elif mask_min == 1 and mask_max == 1:
            patch_type = "foreground"
        else:
            patch_type = "edge"
        classifications.append(patch_type)
    return classifications


def evaluate_validation_by_slide(
    model,
    image_paths,
    mask_paths,
    criterion,
    device,
    batch_size,
    threshold=0.5,
):
    # Check image/mask pairing
    if len(image_paths) != len(mask_paths):
        raise ValueError("Number of images and masks do not match.")

    # Group validation patches by slide
    slide_data = {}
    for image_path, mask_path in zip(image_paths, mask_paths):
        slide_id = extract_slide_id(image_path)

        if slide_id not in slide_data:
            slide_data[slide_id] = {
                "image_paths": [],
                "mask_paths": [],
            }

        slide_data[slide_id]["image_paths"].append(image_path)
        slide_data[slide_id]["mask_paths"].append(mask_path)

    if len(slide_data) == 0:
        raise RuntimeError("No validation slides found.")

    results = []
    model.eval()

    with torch.no_grad():
        for slide_id in sorted(slide_data.keys()):
            slide_image_paths = slide_data[slide_id]["image_paths"]
            slide_mask_paths = slide_data[slide_id]["mask_paths"]

            slide_dataset = SegmentationDataset(
                slide_image_paths,
                slide_mask_paths,
            )

            slide_loader = DataLoader(
                slide_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=NUM_WORKERS,
                pin_memory=(
                    torch.cuda.is_available()
                    and PIN_MEMORY
                ),
            )

            slide_loss_values = []
            slide_dice_values = []
            slide_sensitivity_values = []
            slide_precision_values = []

            background_dice = []
            edge_dice = []
            foreground_dice = []

            for images, masks in slide_loader:
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                # Model prediction
                logits = model(images)
                loss = criterion(logits, masks)

                metrics = calculate_metrics(
                    logits,
                    masks,
                    threshold=threshold,
                )

                slide_loss_values.append(loss.item())
                slide_dice_values.append(metrics["dice"])
                slide_sensitivity_values.append(metrics["sensitivity"])
                slide_precision_values.append(metrics["precision"])

                # Calculate patch-level Dice
                predictions = (
                    torch.sigmoid(logits) >= threshold
                ).float()

                patch_types = classify_mask_batch(masks)

                for i, patch_type in enumerate(patch_types):
                    mask = masks[i]
                    prediction = predictions[i]

                    intersection = (
                        prediction * mask
                    ).sum()

                    dice_denominator = (
                        prediction.sum()
                        + mask.sum()
                    )

                    if dice_denominator == 0:
                        patch_dice = 1.0
                    else:
                        patch_dice = (
                            2.0
                            * intersection
                            / dice_denominator
                        ).item()

                    # Store Dice according to patch class
                    if patch_type == "background":
                        background_dice.append(patch_dice)
                    elif patch_type == "edge":
                        edge_dice.append(patch_dice)
                    else:
                        foreground_dice.append(patch_dice)

            # Store slide-level results
            results.append({
                "slide_id": slide_id,
                "number_of_patches": len(slide_dataset),
                "loss": float(np.mean(slide_loss_values)),
                "dice": float(np.mean(slide_dice_values)),
                "sensitivity": float(np.mean(slide_sensitivity_values)),
                "precision": float(np.mean(slide_precision_values)),
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

    return pd.DataFrame(results)

# ============================================================
# Training Logic
# ============================================================

def train_one_epoch(model, loader, optimizer, criterion, device, threshold=0.5):
    model.train()
    running_loss, total_samples = 0.0, 0
    dice_vals = []

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        current_batch_size = images.size(0)
        running_loss += loss.item() * current_batch_size
        total_samples += current_batch_size

        with torch.no_grad():
            metrics = calculate_metrics(logits, masks, threshold=threshold)
            dice_vals.append(metrics["dice"])

    return running_loss / total_samples, float(np.mean(dice_vals))


def validate_one_epoch(model, loader, criterion, device, threshold=0.5):
    model.eval()
    running_loss, total_samples = 0.0, 0
    dice_vals, sens_vals, prec_vals = [], [], []

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, masks)

            current_batch_size = images.size(0)
            running_loss += loss.item() * current_batch_size
            total_samples += current_batch_size

            metrics = calculate_metrics(logits, masks, threshold=threshold)
            dice_vals.append(metrics["dice"])
            sens_vals.append(metrics["sensitivity"])
            prec_vals.append(metrics["precision"])

    return {
        "loss": running_loss / total_samples,
        "dice": float(np.mean(dice_vals)),
        "sensitivity": float(np.mean(sens_vals)),
        "precision": float(np.mean(prec_vals)),
    }


def plot_training_history(history, output_path):
    epochs = history["epoch"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, history["train_loss"], marker="o", label="Training loss")
    axes[0].plot(epochs, history["val_loss"], marker="o", label="Validation loss")
    axes[0].set_title("Training and Validation Loss (Custom ViT-DPT)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_dice"], marker="o", label="Training Dice")
    axes[1].plot(epochs, history["val_dice"], marker="o", label="Validation Dice")
    axes[1].set_title("Training and Validation Dice (Custom ViT-DPT)")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


# ============================================================
# Main Experiment Execution
# ============================================================

def run_initial_experiment(
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    sample_size=SAMPLE_SIZE,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    threshold=THRESHOLD,
):
    experiment_name = f"dpt_n{sample_size}_bs{batch_size}_ep{epochs}"

    # Create output directories
    for dir_path in [
        RESULTS_DIR,
        MODEL_DIR,
        METRICS_DIR,
        VISUALIZATION_DIR,
    ]:
        dir_path.mkdir(parents=True, exist_ok=True)

    set_seed(RANDOM_SEED)
    device = get_device()

    # Load training and validation data
    train_image_paths, train_mask_paths = find_pairs(
        TRAIN_DIR,
        sample_size=sample_size,
        seed=RANDOM_SEED,
    )

    val_image_paths, val_mask_paths = find_pairs(
        VALIDATION_DIR
    )

    train_dataset = SegmentationDataset(
        train_image_paths,
        train_mask_paths,
    )

    validation_dataset = SegmentationDataset(
        val_image_paths,
        val_mask_paths,
    )

    train_loader, validation_loader = create_dataloaders(
        train_dataset,
        validation_dataset,
        batch_size=batch_size,
        num_workers=NUM_WORKERS,
    )

    # Create model, loss and optimizer
    model = create_model().to(device)

    criterion = BCEDiceLoss(
        bce_weight=0.5,
        dice_weight=0.5,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_dice": [],
        "val_dice": [],
        "val_sensitivity": [],
        "val_precision": [],
    }

    best_validation_dice = -1.0
    best_epoch = None

    best_model_path = (
        MODEL_DIR
        / f"{experiment_name}_34_33_33_DPT_best.pth"
    )

    experiment_start = time.time()

    print("=" * 70)
    print("STARTING DPT EXPERIMENT (CUSTOM ViT-DPT DINOv2 @ 512x512)")
    print("=" * 70)

    # Training loop
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        train_loss, _ = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            threshold=threshold,
        )

        # Calculate training Dice after the epoch
        train_dice = evaluate_training_dice(
            model,
            train_loader,
            device,
            threshold=threshold,
        )

        # Validate model
        val_results = validate_one_epoch(
            model,
            validation_loader,
            criterion,
            device,
            threshold=threshold,
        )

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_results["loss"])
        history["train_dice"].append(train_dice)
        history["val_dice"].append(val_results["dice"])
        history["val_sensitivity"].append(
            val_results["sensitivity"]
        )
        history["val_precision"].append(
            val_results["precision"]
        )

        epoch_time = time.time() - epoch_start

        print(
            f"\nEpoch {epoch:03d}/{epochs:03d} "
            f"[{epoch_time:.2f}s]"
        )
        print(
            f"  Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_results['loss']:.4f}"
        )
        print(
            f"  Train Dice: {train_dice:.4f} | "
            f"Val Dice: {val_results['dice']:.4f}"
        )

        # Save best validation model
        if val_results["dice"] > best_validation_dice:
            best_validation_dice = val_results["dice"]
            best_epoch = epoch

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_dice": val_results["dice"],
                    "encoder": ENCODER,
                    "encoder_weights": ENCODER_WEIGHTS,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "threshold": threshold,
                },
                best_model_path,
            )

            print("  -> New best model saved.")

    total_runtime = time.time() - experiment_start

    # Save training history and curves
    history_df = pd.DataFrame(history)

    history_path = (
        METRICS_DIR
        / f"{experiment_name}_34_33_33_DPT_training_history.csv"
    )

    history_df.to_csv(
        history_path,
        index=False,
    )

    plot_path = (
        VISUALIZATION_DIR
        / f"{experiment_name}_34_33_33_DPT_training_curves.png"
    )

    plot_training_history(
        history,
        plot_path,
    )

    best_row = history_df.loc[
        history_df["val_dice"].idxmax()
    ]

    # Load best model for slide evaluation
    best_checkpoint = torch.load(
        best_model_path,
        map_location=device,
    )

    model.load_state_dict(
        best_checkpoint["model_state_dict"]
    )

    # Calculate final slide-level metrics
    slide_summary_df = evaluate_validation_by_slide(
        model=model,
        image_paths=val_image_paths,
        mask_paths=val_mask_paths,
        criterion=criterion,
        device=device,
        batch_size=batch_size,
        threshold=threshold,
    )

    slide_summary_path = (
        METRICS_DIR
        / f"{experiment_name}_34_33_33_DPT_training_summary_slides.csv"
    )

    slide_summary_df.to_csv(
        slide_summary_path,
        index=False,
    )

    # Create overall experiment summary
    summary = {
        "encoder": ENCODER,
        "encoder_weights": ENCODER_WEIGHTS,
        "sample_size": len(train_dataset),
        "validation_size": len(validation_dataset),
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "threshold": threshold,
        "best_epoch": int(best_row["epoch"]),
        "best_val_dice": float(best_row["val_dice"]),
        "best_val_loss": float(best_row["val_loss"]),
        "best_val_sensitivity": float(
            best_row["val_sensitivity"]
        ),
        "best_val_precision": float(
            best_row["val_precision"]
        ),
        "best_val_dice_background": float(
            slide_summary_df["dice_background"].mean()
        ),
        "best_val_dice_edge": float(
            slide_summary_df["dice_edge"].mean()
        ),
        "best_val_dice_foreground": float(
            slide_summary_df["dice_foreground"].mean()
        ),
        "total_runtime_seconds": total_runtime,
    }

    summary_df = pd.DataFrame([summary])

    summary_path = (
        METRICS_DIR
        / f"{experiment_name}_34_33_33_DPT_training_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    print("\n" + "=" * 70)
    print("EXPERIMENT FINISHED")
    print(
        f"Best Epoch: {best_epoch} | "
        f"Best Val Dice: {best_validation_dice:.4f}"
    )
    print(
        f"Total Runtime: "
        f"{total_runtime / 60:.2f} minutes"
    )
    print("=" * 70)

    return {
        "model": model,
        "history": history,
        "summary": summary,
        "best_model_path": best_model_path,
    }


if __name__ == "__main__":
    run_initial_experiment()
