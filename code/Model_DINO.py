### Model DINO ###

from pathlib import Path
import random
import time
import re

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

# Configuration
TRAIN_DIR = Path(r"/home/20234257/BEP_Project/External_data/training_34_33_33") # CHANGE #
VALIDATION_DIR = Path(r"/home/20234257/BEP_Project/External_data/validation")
RESULTS_DIR = Path(r"/home/20234257/BEP_Project/BEP_blood_vessel_graft_segmentation-main/results_full")

MODEL_DIR = RESULTS_DIR / "models"
METRICS_DIR = RESULTS_DIR / "metrics"
VISUALIZATION_DIR = RESULTS_DIR / "visualizations"

ENCODER = "resnet50"
IN_CHANNELS = 3
CLASSES = 1

BATCH_SIZE = 8
EPOCHS = 50
SAMPLE_SIZE = 10000
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
THRESHOLD = 0.5

NUM_WORKERS = 4
PIN_MEMORY = True
RANDOM_SEED = 42

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
        print(f"\nGPU detected: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        device = torch.device("cpu")
        print("\nNo CUDA GPU detected. Using CPU.")
    return device

class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths):
        if len(image_paths) != len(mask_paths):
            raise ValueError("Number of images and masks do not match.")
        self.image_paths = list(image_paths)
        self.mask_paths = list(mask_paths)
        if len(self.image_paths) == 0:
            raise ValueError("Dataset contains no images.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert("RGB")
        mask = Image.open(self.mask_paths[index]).convert("L")

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

def find_pairs(directory, sample_size=None, seed=42):
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory}")

    tissue_paths = sorted(directory.glob("*_tissue.png"))
    if len(tissue_paths) == 0:
        raise FileNotFoundError(f"No *_tissue.png files found in: {directory}")

    pairs = []
    for tissue_path in tissue_paths:
        mask_path = Path(str(tissue_path).replace("_tissue.png", "_mask.png"))
        if not mask_path.exists():
            print(f"WARNING: mask missing for {tissue_path}")
            continue
        pairs.append((tissue_path, mask_path))

    if len(pairs) == 0:
        raise RuntimeError("No valid tissue/mask pairs found.")

    if sample_size is not None and sample_size < len(pairs):
        rng = random.Random(seed)
        pairs = rng.sample(pairs, sample_size)

    image_paths = [p[0] for p in pairs]
    mask_paths = [p[1] for p in pairs]
    return image_paths, mask_paths

def extract_slide_id(image_path):
    filename = Path(image_path).name
    match = re.match(
        r"^image_(.+?)_patch_\d+_tissue\.png$",
        filename,
    )

    if match is None:
        raise ValueError(
            f"Could not extract slide ID from filename: {filename}"
        )

    return match.group(1)

def evaluate_validation_by_slide(
    model,
    image_paths,
    mask_paths,
    criterion,
    device,
    batch_size,
    threshold=0.5,
):
    slide_data = {}

    for image_path, mask_path in zip(image_paths, mask_paths):
        slide_id = extract_slide_id(image_path)
        slide_data.setdefault(slide_id, []).append(
            (image_path, mask_path)
        )

    results = []
    model.eval()

    with torch.no_grad():
        for slide_id in sorted(slide_data):
            pairs = slide_data[slide_id]

            images = [p[0] for p in pairs]
            masks = [p[1] for p in pairs]

            dataset = SegmentationDataset(images, masks)

            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=NUM_WORKERS,
                pin_memory=(
                    torch.cuda.is_available()
                    and PIN_MEMORY
                ),
            )

            losses, dices = [], []
            sensitivities, precisions = [], []

            for batch_images, batch_masks in loader:
                batch_images = batch_images.to(
                    device,
                    non_blocking=True,
                )
                batch_masks = batch_masks.to(
                    device,
                    non_blocking=True,
                )

                logits = model(batch_images)
                loss = criterion(logits, batch_masks)

                metrics = calculate_metrics(
                    logits,
                    batch_masks,
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

            results.append({
                "slide_id": slide_id,
                "number_of_patches": len(dataset),
                "loss": float(np.mean(losses)),
                "dice": float(np.mean(dices)),
                "sensitivity": float(np.mean(sensitivities)),
                "precision": float(np.mean(precisions)),
            })

    return pd.DataFrame(results)

def create_model():
    """
    Creates a U-Net model with a ResNet-50 backbone initialized 
    using Meta's self-supervised DINO v1 weights via Torch Hub.
    """
    print("Loading pretrained DINO v1 ResNet-50 weights from Torch Hub...")
    dino_backbone = torch.hub.load("facebookresearch/dino:main", "dino_resnet50")

    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=None,  # Do not load ImageNet weights; we inject DINO v1 weights
        in_channels=IN_CHANNELS,
        classes=CLASSES,
    )

    # Load DINO v1 state dictionary into the U-Net encoder
    model.encoder.load_state_dict(dino_backbone.state_dict(), strict=False)
    print("Successfully initialized U-Net encoder with DINO v1 weights.")

    return model

def create_dataloaders(train_dataset, validation_dataset, batch_size, num_workers=0):
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(torch.cuda.is_available() and PIN_MEMORY),
    )
    val_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(torch.cuda.is_available() and PIN_MEMORY),
    )
    return train_loader, val_loader

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss, total_samples = 0.0, 0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

    return running_loss / total_samples

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

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

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

def evaluate_training_dice(model, loader, device, threshold=0.5):
    model.eval()
    dice_vals = []
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            logits = model(images)
            metrics = calculate_metrics(logits, masks, threshold=threshold)
            dice_vals.append(metrics["dice"])
    return float(np.mean(dice_vals))

def plot_training_history(history, output_path):
    epochs = history["epoch"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, history["train_loss"], marker="o", label="Training loss")
    axes[0].plot(epochs, history["val_loss"], marker="o", label="Validation loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss (U-Net DINO v1)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_dice"], marker="o", label="Training Dice")
    axes[1].plot(epochs, history["val_dice"], marker="o", label="Validation Dice")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice")
    axes[1].set_title("Training and Validation Dice (U-Net DINO v1)")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

def run_initial_experiment(
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    sample_size=SAMPLE_SIZE,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    threshold=THRESHOLD,
):
    experiment_name = f"dino_resnet50_n{sample_size}_bs{batch_size}_ep{epochs}"

    for path in [RESULTS_DIR, MODEL_DIR, METRICS_DIR, VISUALIZATION_DIR]:
        path.mkdir(parents=True, exist_ok=True)

    set_seed(RANDOM_SEED)
    device = get_device()

    train_img, train_mask = find_pairs(TRAIN_DIR, sample_size=sample_size, seed=RANDOM_SEED)
    val_img, val_mask = find_pairs(VALIDATION_DIR)

    train_dataset = SegmentationDataset(train_img, train_mask)
    val_dataset = SegmentationDataset(val_img, val_mask)

    train_loader, val_loader = create_dataloaders(
        train_dataset, val_dataset, batch_size=batch_size, num_workers=NUM_WORKERS
    )

    model = create_model().to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_dice": [],
        "val_dice": [],
        "val_sensitivity": [],
        "val_precision": [],
    }

    best_val_dice = -1.0
    best_epoch = None
    best_model_path = MODEL_DIR / f"{experiment_name}_34_33_33_DINO_best.pth" # CHANGE #

    start_time = time.time()
    print("\n" + "=" * 70)
    print("STARTING U-NET DINO v1 EXPERIMENT")
    print("=" * 70)

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        train_dice = evaluate_training_dice(model, train_loader, device, threshold=threshold)

        val_res = validate_one_epoch(model, val_loader, criterion, device, threshold=threshold)
        val_loss, val_dice = val_res["loss"], val_res["dice"]
        val_sens, val_prec = val_res["sensitivity"], val_res["precision"]

        epoch_time = time.time() - epoch_start

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_dice"].append(train_dice)
        history["val_dice"].append(val_dice)
        history["val_sensitivity"].append(val_sens)
        history["val_precision"].append(val_prec)

        print(f"\nEpoch {epoch:03d}/{epochs:03d}")
        print(f"  Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")
        print(f"  Train Dice: {train_dice:.4f} | Val Dice: {val_dice:.4f}")
        print(f"  Val sens.:  {val_sens:.4f} | Val prec.: {val_prec:.4f}")
        print(f"  Runtime:    {epoch_time:.2f} s")

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_dice": val_dice,
                    "encoder": ENCODER,
                },
                best_model_path,
            )
            print("  -> New best model saved.")

    total_runtime = time.time() - start_time

    history_df = pd.DataFrame(history)
    history_path = METRICS_DIR / f"{experiment_name}_34_33_33_DINO_training_history.csv" # CHANGE #
    history_df.to_csv(history_path, index=False)

    plot_path = VISUALIZATION_DIR / f"{experiment_name}_34_33_33_DINO_training_curves.png" # CHANGE #
    plot_training_history(history, plot_path)

    best_row = history_df.loc[history_df["val_dice"].idxmax()]

    best_checkpoint = torch.load(
        best_model_path,
        map_location=device,
    )

    model.load_state_dict(
        best_checkpoint["model_state_dict"]
    )

    slide_summary_df = evaluate_validation_by_slide(
        model=model,
        image_paths=val_img,
        mask_paths=val_mask,
        criterion=criterion,
        device=device,
        batch_size=batch_size,
        threshold=threshold,
    )

    slide_summary_path = (
        METRICS_DIR
        / f"{experiment_name}_34_33_33_DINO_training_summary_slides.csv" # CHANGE #
    )

    slide_summary_df.to_csv(
        slide_summary_path,
        index=False,
    )

    summary = {
        "encoder": ENCODER,
        "sample_size": len(train_dataset),
        "validation_size": len(val_dataset),
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "best_epoch": int(best_row["epoch"]),
        "best_val_dice": float(best_row["val_dice"]),
        "best_val_loss": float(best_row["val_loss"]),
        "total_runtime_seconds": total_runtime,
    }

    summary_df = pd.DataFrame([summary])
    summary_path = METRICS_DIR / f"{experiment_name}_34_33_33_DINO_training_summary.csv" # CHANGE #
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 70)
    print("EXPERIMENT FINISHED")
    print(f"Best epoch: {best_epoch} | Best val Dice: {best_val_dice:.4f}")
    print(f"Total runtime: {total_runtime / 60:.2f} minutes")
    print("=" * 70)

    return {"model": model, "history": history, "summary": summary, "best_model_path": best_model_path}

if __name__ == "__main__":
    run_initial_experiment()
