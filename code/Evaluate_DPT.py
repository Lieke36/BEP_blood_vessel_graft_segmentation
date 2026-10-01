### Evaluate_DPT ###

from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from Losses_metrics import calculate_metrics


# ============================================================
# Configuration
# ============================================================

ENCODER = "vit_small_patch14_dinov2.lvd142m"
CLASSES = 1
IMG_SIZE = 512
THRESHOLD = 0.5

IMAGE_MEAN = np.array(
    [0.485, 0.456, 0.406], dtype=np.float32
).reshape(3, 1, 1)

IMAGE_STD = np.array(
    [0.229, 0.224, 0.225], dtype=np.float32
).reshape(3, 1, 1)


# ============================================================
# Device
# ============================================================

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("CUDA not available. Using CPU.")
    return device


# ============================================================
# Model
# ============================================================

class CustomViTDPT(nn.Module):
    def __init__(self, encoder_name=ENCODER, out_channels=CLASSES, pretrained=False):
        super().__init__()

        model_key = encoder_name.replace("tu-", "")
        self.encoder = timm.create_model(
            model_key,
            pretrained=pretrained,
            dynamic_img_size=True,
            features_only=False,
        )

        embed_dim = self.encoder.embed_dim

        self.proj1 = nn.Sequential(
            nn.Conv2d(embed_dim, 256, 1),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )
        self.proj2 = nn.Sequential(
            nn.Conv2d(embed_dim, 256, 1),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )
        self.proj3 = nn.Sequential(
            nn.Conv2d(embed_dim, 256, 1),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )
        self.proj4 = nn.Sequential(
            nn.Conv2d(embed_dim, 256, 1),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(1024, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, out_channels, 1),
        )

    def forward(self, x):
        orig_H, orig_W = x.shape[-2:]

        pad_h = (14 - orig_H % 14) % 14
        pad_w = (14 - orig_W % 14) % 14

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        padded_H, padded_W = x.shape[-2:]

        f1, f2, f3, f4 = self.encoder.get_intermediate_layers(
            x, n=[3, 5, 7, 11], reshape=True
        )

        p1 = self.proj1(f1)
        p2 = self.proj2(f2)
        p3 = self.proj3(f3)
        p4 = self.proj4(f4)

        target_size = (padded_H // 4, padded_W // 4)

        p1 = F.interpolate(p1, size=target_size, mode="bilinear", align_corners=False)
        p2 = F.interpolate(p2, size=target_size, mode="bilinear", align_corners=False)
        p3 = F.interpolate(p3, size=target_size, mode="bilinear", align_corners=False)
        p4 = F.interpolate(p4, size=target_size, mode="bilinear", align_corners=False)

        fused = torch.cat([p1, p2, p3, p4], dim=1)
        decoded = self.decoder(fused)

        out = F.interpolate(
            decoded,
            size=(padded_H, padded_W),
            mode="bilinear",
            align_corners=False,
        )

        return out[:, :, :orig_H, :orig_W]


def create_model():
    return CustomViTDPT(
        encoder_name=ENCODER,
        out_channels=CLASSES,
        pretrained=False,
    )


def load_model(model_path, device):
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found:\n{model_path}")

    print(f"\nLoading DPT model:\n{model_path}")

    model = create_model()
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])

        if "epoch" in checkpoint:
            print(f"Checkpoint epoch: {checkpoint['epoch']}")

        if "validation_dice" in checkpoint:
            print(
                f"Checkpoint validation Dice: "
                f"{checkpoint['validation_dice']:.4f}"
            )
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    return model


# ============================================================
# Preprocessing
# ============================================================

def preprocess_image(image_path):
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Tissue image not found:\n{image_path}")

    image = Image.open(image_path).convert("RGB")
    image = image.resize(
        (IMG_SIZE, IMG_SIZE),
        resample=Image.Resampling.BILINEAR,
    )

    image = np.asarray(image, dtype=np.float32)
    image = image.transpose(2, 0, 1) / 255.0
    image = (image - IMAGE_MEAN) / IMAGE_STD

    return torch.from_numpy(image).float().unsqueeze(0)


def load_ground_truth(mask_path):
    mask_path = Path(mask_path)

    if not mask_path.exists():
        raise FileNotFoundError(f"Ground-truth mask not found:\n{mask_path}")

    mask = Image.open(mask_path).convert("L")
    mask = mask.resize(
        (IMG_SIZE, IMG_SIZE),
        resample=Image.Resampling.NEAREST,
    )

    mask = np.asarray(mask, dtype=np.uint8)
    mask = (mask > 0).astype(np.float32)

    return torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)


# ============================================================
# Prediction
# ============================================================

def download_predicted_mask(
    slide_id,
    patch_nr,
    input_path,
    output_path,
    model,
    device,
    threshold=THRESHOLD,
):
    input_path = Path(input_path)
    output_path = Path(output_path)

    tissue_path = input_path / (
        f"image_{slide_id}_patch_{patch_nr}_tissue.png"
    )
    mask_path = input_path / (
        f"image_{slide_id}_patch_{patch_nr}_mask.png"
    )
    predicted_path = output_path / (
        f"image_{slide_id}_patch_{patch_nr}_predicted.png"
    )
    sigmoid_path = output_path / (
        f"image_{slide_id}_patch_{patch_nr}_sigmoid.png"
    )

    if not tissue_path.exists():
        raise FileNotFoundError(f"Tissue patch not found:\n{tissue_path}")

    if not mask_path.exists():
        raise FileNotFoundError(f"Ground-truth mask not found:\n{mask_path}")

    output_path.mkdir(parents=True, exist_ok=True)

    image = preprocess_image(tissue_path).to(device)
    ground_truth = load_ground_truth(mask_path).to(device)

    with torch.no_grad():
        logits = model(image)
        probabilities = torch.sigmoid(logits)

        sigmoid_map = (
            probabilities[0, 0].cpu().numpy() * 255
        ).clip(0, 255).astype(np.uint8)

        Image.fromarray(
            sigmoid_map,
            mode="L",
        ).save(sigmoid_path)

        predictions = (probabilities >= threshold).float()

    metrics = calculate_metrics(
        logits,
        ground_truth,
        threshold=threshold,
    )

    predicted_mask = (
        predictions[0, 0].cpu().numpy() * 255
    ).astype(np.uint8)

    Image.fromarray(
        predicted_mask,
        mode="L",
    ).save(predicted_path)

    print("\n" + "=" * 60)
    print("DPT PATCH EVALUATION")
    print("=" * 60)
    print(f"Slide ID:       {slide_id}")
    print(f"Patch number:   {patch_nr}")
    print(f"Input size:     {IMG_SIZE} x {IMG_SIZE}")
    print(f"Threshold:      {threshold}")
    print(f"Dice:           {metrics['dice']:.4f}")
    print(f"Sensitivity:    {metrics['sensitivity']:.4f}")
    print(f"Precision:      {metrics['precision']:.4f}")
    print(f"\nSaved prediction:\n{predicted_path}")
    print(f"\nSaved sigmoid prediction:\n{sigmoid_path}")
    print("=" * 60)

    return {
        "slide_id": slide_id,
        "patch_nr": patch_nr,
        "dice": metrics["dice"],
        "sensitivity": metrics["sensitivity"],
        "precision": metrics["precision"],
        "predicted_mask_path": str(predicted_path),
        "sigmoid_prediction_path": str(sigmoid_path),
    }


# ============================================================
# Main
# ============================================================

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

if __name__ == "__main__":

    INPUT_PATH = Path(
        r"/home/20234257/BEP_Project/External_data/validation"
    )

    OUTPUT_PATH = Path(
        r"/home/20234257/BEP_Project/BEP_blood_vessel_graft_segmentation-main/results_full/evaluation/dpt_n10000_bs8_ep50"
    )

    MODEL_PATH = Path(
        r"/home/20234257/BEP_Project/BEP_blood_vessel_graft_segmentation-main/results_full/models/dpt_n10000_bs8_ep50_10_80_10_DPT_best.pth"
    )

    device = get_device()
    model = load_model(MODEL_PATH, device)

    for slide in slide_id_list:
        for patch in patch_nr_list:
            download_predicted_mask(
                slide_id=slide,
                patch_nr=patch,
                input_path=INPUT_PATH,
                output_path=OUTPUT_PATH,
                model=model,
                device=device,
                threshold=THRESHOLD,
            )
