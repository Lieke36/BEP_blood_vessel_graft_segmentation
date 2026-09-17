### Data distribution ###

from pathlib import Path
from PIL import Image
import numpy as np


# ============================================================
# Configuration
# ============================================================

MASK_PATH = Path(
    r"/home/20234257/BEP_Project/External_data/training_34_33_33"
)


# ============================================================
# Count mask types
# ============================================================

fully_background = 0
fully_foreground = 0
mixed = 0

total_masks = 0


# ============================================================
# Loop through masks
# ============================================================

for mask_path in MASK_PATH.glob("*_mask.png"):

    mask = np.asarray(
        Image.open(mask_path).convert("L")
    )

    total_masks += 1

    # --------------------------------------------------------
    # Fully background
    # --------------------------------------------------------

    if np.all(mask == 0):

        fully_background += 1

    # --------------------------------------------------------
    # Fully foreground
    # --------------------------------------------------------

    elif np.all(mask == 255):

        fully_foreground += 1

    # --------------------------------------------------------
    # Contains both background and foreground
    # --------------------------------------------------------

    else:

        mixed += 1


# ============================================================
# Calculate percentages
# ============================================================

background_percentage = (
    fully_background / total_masks * 100
)

foreground_percentage = (
    fully_foreground / total_masks * 100
)

mixed_percentage = (
    mixed / total_masks * 100
)


# ============================================================
# Print results
# ============================================================

print("=" * 60)
print("MASK DISTRIBUTION")
print("=" * 60)

print(f"Total masks:        {total_masks}")

print(
    f"Fully background:   "
    f"{fully_background:5d} "
    f"({background_percentage:.2f}%)"
)

print(
    f"Fully foreground:   "
    f"{fully_foreground:5d} "
    f"({foreground_percentage:.2f}%)"
)

print(
    f"Foreground + "
    f"background:         "
    f"{mixed:5d} "
    f"({mixed_percentage:.2f}%)"
)

print("=" * 60)

print(
    f"Percentage total:   "
    f"{background_percentage + foreground_percentage + mixed_percentage:.2f}%"
)
