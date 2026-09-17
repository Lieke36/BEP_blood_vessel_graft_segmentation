### Model DPT ###


# ============================================================
# Model_DPT.py - DINOv2 Patch-14 Implementation
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

TRAIN_DIR = Path(r"/home/20234257/BEP_Project/External_data/training")
VALIDATION_DIR = Path(r"/home/20234257/BEP_Project/External_data/validation")
RESULTS_DIR = Path(r"/home/20234257/BEP_Project/BEP_blood_vessel_graft_segmentation-main/results")

MODEL_DIR = RESULTS_DIR / "models"
METRICS_DIR = RESULTS_DIR / "metrics"
VISUALIZATION_DIR = RESULTS_DIR / "visualizations"

# DINOv2 Patch-14 Configuration
ENCODER = "tu-vit_small_patch14_dinov2.lvd142m"
ENCODER_WEIGHTS = True
CLASSES = 1
IMG_SIZE = 504  # 504 / 14 = 36 patches (even spatial grid)

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

        # Resize to 504x504 for DINOv2 Patch-14 alignment
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


def create_model(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS):
    # 1. Instantiate smp.DPT normally
    model = smp.DPT(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        classes=CLASSES,
        activation=None,
    )
    
    # 2. Set dynamic image size directly on the underlying timm ViT backbone
    # This prevents position embedding size mismatches during the forward pass at 504x504
    if hasattr(model.encoder.model, "dynamic_img_size"):
        model.encoder.model.dynamic_img_size = True
        
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
    axes[0].set_title("Training and Validation Loss (DPT DINOv2)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_dice"], marker="o", label="Training Dice")
    axes[1].plot(epochs, history["val_dice"], marker="o", label="Validation Dice")
    axes[1].set_title("Training and Validation Dice (DPT DINOv2)")
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

    for dir_path in [RESULTS_DIR, MODEL_DIR, METRICS_DIR, VISUALIZATION_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)

    set_seed(RANDOM_SEED)
    device = get_device()

    train_image_paths, train_mask_paths = find_pairs(
        TRAIN_DIR, sample_size=sample_size, seed=RANDOM_SEED
    )
    val_image_paths, val_mask_paths = find_pairs(VALIDATION_DIR)

    train_dataset = SegmentationDataset(train_image_paths, train_mask_paths)
    validation_dataset = SegmentationDataset(val_image_paths, val_mask_paths)

    train_loader, validation_loader = create_dataloaders(
        train_dataset, validation_dataset, batch_size=batch_size, num_workers=NUM_WORKERS
    )

    model = create_model().to(device)

    criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
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
    best_model_path = MODEL_DIR / f"{experiment_name}_DPT_best.pth"
    experiment_start = time.time()

    print("=" * 70)
    print("STARTING DPT EXPERIMENT (DINOv2 PATCH-14 @ 504x504)")
    print("=" * 70)

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        train_loss, train_dice = train_one_epoch(
            model, train_loader, optimizer, criterion, device, threshold=threshold
        )

        val_results = validate_one_epoch(
            model, validation_loader, criterion, device, threshold=threshold
        )

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_results["loss"])
        history["train_dice"].append(train_dice)
        history["val_dice"].append(val_results["dice"])
        history["val_sensitivity"].append(val_results["sensitivity"])
        history["val_precision"].append(val_results["precision"])

        epoch_time = time.time() - epoch_start

        print(f"\nEpoch {epoch:03d}/{epochs:03d} [{epoch_time:.2f}s]")
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_results['loss']:.4f}")
        print(f"  Train Dice: {train_dice:.4f} | Val Dice: {val_results['dice']:.4f}")

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

    history_df = pd.DataFrame(history)
    history_path = METRICS_DIR / f"{experiment_name}_DPT_training_history.csv"
    history_df.to_csv(history_path, index=False)

    plot_path = VISUALIZATION_DIR / f"{experiment_name}_DPT_training_curves.png"
    plot_training_history(history, plot_path)

    best_row = history_df.loc[history_df["val_dice"].idxmax()]
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
        "best_val_sensitivity": float(best_row["val_sensitivity"]),
        "best_val_precision": float(best_row["val_precision"]),
        "total_runtime_seconds": total_runtime,
    }

    summary_df = pd.DataFrame([summary])
    summary_path = METRICS_DIR / f"{experiment_name}_DPT_training_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 70)
    print("EXPERIMENT FINISHED")
    print(f"Best Epoch: {best_epoch} | Best Val Dice: {best_validation_dice:.4f}")
    print(f"Total Runtime: {total_runtime / 60:.2f} minutes")
    print("=" * 70)

    return {
        "model": model,
        "history": history,
        "summary": summary,
        "best_model_path": best_model_path,
    }


if __name__ == "__main__":
    run_initial_experiment()
