### Data_processing ###
# make sure the code is applicable to HPC GPU/CPU
# Asume image of full tissue to be segmented is available


## training patches sampling + masks ##
# use downloaded tissue samples and masks (from data -> slides)
# use random augmented sampling (randow orientation, scaling, stain?)
# download and divide sampled patches and masks in (data -> images -> train) and (data -> masks -> train)
# put location, orientation, size, stain? in train.csv (with slide id and patch id)


## validation / testing patches sampling ##
# use full pathological images and masks
# put grid with no augmentation
# download all sampled all patches + masks in (data-> images -> val,test) and (data-> masks -> val,test)
# put location of patches in val.csv and test.csv (with slide id and patch id)


## turn saved images into pytorch datasets? ##


## libaries ##
from pathlib import Path
import csv
import math
import random

from PIL import Image



## functions ##

def random_patch_train(
    slides_dir,
    train_output_dir,
    slide_id,
    n=1,
    magnification_change=0.1,
):
    """
    Samples n random 512x512 tissue and mask patches from one slide.

    For every patch:
    - random location on the slide
    - random magnification between
      1 - magnification_change and
      1 + magnification_change
    - random rotation between 0 and 360 degrees
    - final patch size is always 512x512 px
    - tissue and mask receive exactly the same transformations

    Patch information is stored in:
        data/training/train_tissue_patches.csv
        data/training/train_mask_patches.csv

    Patch images are stored in:
        train_output_dir
    """

    patch_size = 512

    slides_dir = Path(slides_dir)
    train_output_dir = Path(train_output_dir)

    train_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_dir = Path("data/training")
    csv_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    tissue_csv = csv_dir / "train_tissue_patches.csv"
    mask_csv = csv_dir / "train_mask_patches.csv"

    tissue_path = (
        slides_dir / f"image_{slide_id}_tissue.png"
    )

    mask_path = (
        slides_dir / f"image_{slide_id}_mask.png"
    )

    if not tissue_path.exists():
        raise FileNotFoundError(
            f"Tissue slide not found: {tissue_path}"
        )

    if not mask_path.exists():
        raise FileNotFoundError(
            f"Mask slide not found: {mask_path}"
        )

    # ---------------------------------------------------------
    # Open the images without converting the entire slide.
    # ---------------------------------------------------------

    tissue_image = Image.open(tissue_path)
    mask_image = Image.open(mask_path)

    try:

        if tissue_image.size != mask_image.size:
            raise ValueError(
                f"Tissue and mask sizes do not match for "
                f"slide {slide_id}: "
                f"{tissue_image.size} vs {mask_image.size}"
            )

        width, height = tissue_image.size

        # -----------------------------------------------------
        # Random patches
        # -----------------------------------------------------

        for patch_number in range(n):

            # -------------------------------------------------
            # Random magnification.
            # -------------------------------------------------

            magnification = random.uniform(
                1 - magnification_change,
                1 + magnification_change,
            )

            source_size = round(
                patch_size * magnification
            )

            # Make sure the source crop fits on the slide.
            source_size = min(
                source_size,
                width,
                height,
            )

            # -------------------------------------------------
            # Random top-left coordinate.
            # -------------------------------------------------

            max_x = width - source_size
            max_y = height - source_size

            x = random.randint(
                0,
                max_x,
            )

            y = random.randint(
                0,
                max_y,
            )

            # -------------------------------------------------
            # Random rotation.
            # -------------------------------------------------

            degrees = random.uniform(
                0,
                360,
            )

            # -------------------------------------------------
            # Crop ONLY the required region.
            #
            # We do not convert the entire slide.
            # -------------------------------------------------

            tissue_patch = tissue_image.crop(
                (
                    x,
                    y,
                    x + source_size,
                    y + source_size,
                )
            )

            mask_patch = mask_image.crop(
                (
                    x,
                    y,
                    x + source_size,
                    y + source_size,
                )
            )

            # -------------------------------------------------
            # Convert ONLY the small patches.
            # -------------------------------------------------

            tissue_patch = tissue_patch.convert("RGB")
            mask_patch = mask_patch.convert("L")

            # -------------------------------------------------
            # Resize both to 512x512.
            #
            # Tissue: LANCZOS
            # Mask: NEAREST
            # -------------------------------------------------

            tissue_patch = tissue_patch.resize(
                (patch_size, patch_size),
                Image.Resampling.LANCZOS,
            )

            mask_patch = mask_patch.resize(
                (patch_size, patch_size),
                Image.Resampling.NEAREST,
            )

            # -------------------------------------------------
            # Rotate both by exactly the same amount.
            # -------------------------------------------------

            tissue_patch = tissue_patch.rotate(
                degrees,
                resample=Image.Resampling.BICUBIC,
                expand=False,
            )

            mask_patch = mask_patch.rotate(
                degrees,
                resample=Image.Resampling.NEAREST,
                expand=False,
            )

            # -------------------------------------------------
            # Future stain augmentation
            #
            # 0 = no stain augmentation
            # 1 = weak
            # 2 = medium
            # 3 = strong
            # -------------------------------------------------

            stain_intensity = 0

            # -------------------------------------------------
            # Patch ID
            # -------------------------------------------------

            patch_id = (
                f"{slide_id}_{patch_number}"
            )

            tissue_destination = (
                train_output_dir
                / f"image_{slide_id}_patch_{patch_number}_tissue.png"
            )

            mask_destination = (
                train_output_dir
                / f"image_{slide_id}_patch_{patch_number}_mask.png"
            )

            # -------------------------------------------------
            # Save patches.
            # -------------------------------------------------

            tissue_patch.save(
                tissue_destination
            )

            mask_patch.save(
                mask_destination
            )

            # -------------------------------------------------
            # Close the small patch images immediately.
            # -------------------------------------------------

            tissue_patch.close()
            mask_patch.close()

            # -------------------------------------------------
            # Store CSV information.
            # -------------------------------------------------

            append_csv_row(
                tissue_csv,
                {
                    "slide_id": slide_id,
                    "patch_id": patch_id,
                    "x": x,
                    "y": y,
                    "magnification": magnification,
                    "degrees": degrees,
                    "stain_intensity": stain_intensity,
                    "tissue_path": tissue_destination.name,
                },
            )

            append_csv_row(
                mask_csv,
                {
                    "slide_id": slide_id,
                    "patch_id": patch_id,
                    "x": x,
                    "y": y,
                    "magnification": magnification,
                    "degrees": degrees,
                    "stain_intensity": stain_intensity,
                    "mask_path": mask_destination.name,
                },
            )

            print(
                f"Created training patch {patch_id}: "
                f"x={x}, y={y}, "
                f"magnification={magnification:.3f}, "
                f"degrees={degrees:.1f}"
            )

    finally:

        # -----------------------------------------------------
        # Close the original slide files.
        # -----------------------------------------------------

        tissue_image.close()
        mask_image.close()

    print(
        f"Created {n} random training patch(es) "
        f"for slide {slide_id}."
    )


def full_patch_extraction(
    slides_dir,
    output_dir,
    slide_id,
    group,
):
    """
    Extracts all complete 512x512 patches from one slide.

    The largest possible grid of complete patches is used.

    The grid is centered in the slide.

    For example:
        slide = 20x20
        patch = 6x6

        floor(20/6) = 3

        grid = 3x3 = 9 patches

        grid size = 18x18

        remaining = 2 pixels

        therefore:
            1 pixel border on each side

    group must be:
        'training'
        'test'
        'validation'

    Patch information is stored in:
        data/[group]/

    The function returns the total number of patches.
    """

    patch_size = 512

    slides_dir = Path(slides_dir)
    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if group not in (
        "training",
        "test",
        "validation",
    ):
        raise ValueError(
            "group must be 'training', 'test' "
            "or 'validation'."
        )

    csv_dir = Path("data") / group

    csv_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    tissue_csv = (
        csv_dir / f"{group}_tissue_patches.csv"
    )

    mask_csv = (
        csv_dir / f"{group}_mask_patches.csv"
    )

    tissue_path = (
        slides_dir / f"image_{slide_id}_tissue.png"
    )

    mask_path = (
        slides_dir / f"image_{slide_id}_mask.png"
    )

    if not tissue_path.exists():
        raise FileNotFoundError(
            f"Tissue slide not found: {tissue_path}"
        )

    if not mask_path.exists():
        raise FileNotFoundError(
            f"Mask slide not found: {mask_path}"
        )

    with Image.open(tissue_path) as tissue_image, \
         Image.open(mask_path) as mask_image:

        tissue_image = tissue_image.convert("RGB")
        mask_image = mask_image.convert("L")

        if tissue_image.size != mask_image.size:
            raise ValueError(
                f"Tissue and mask sizes do not match "
                f"for slide {slide_id}: "
                f"{tissue_image.size} vs "
                f"{mask_image.size}"
            )

        width, height = tissue_image.size

        # -----------------------------------------------------
        # Determine number of complete patches.
        #
        # floor is intentional:
        # incomplete patches at the edges are not used.
        # -----------------------------------------------------

        patches_x = math.floor(
            width / patch_size
        )

        patches_y = math.floor(
            height / patch_size
        )

        total_patches = (
            patches_x * patches_y
        )

        print(
            f"Slide {slide_id}: "
            f"{width} x {height} px"
        )

        print(
            f"Patch grid: "
            f"{patches_x} x {patches_y}"
        )

        print(
            f"Total patches: {total_patches}"
        )

        # -----------------------------------------------------
        # Determine size of centered grid.
        # -----------------------------------------------------

        grid_width = (
            patches_x * patch_size
        )

        grid_height = (
            patches_y * patch_size
        )

        # Remaining pixels are split between both sides.
        start_x = (
            width - grid_width
        ) // 2

        start_y = (
            height - grid_height
        ) // 2

        print(
            f"Grid starts at: "
            f"x={start_x}, y={start_y}"
        )

        patch_number = 0

        for row in range(patches_y):

            for column in range(patches_x):

                x = (
                    start_x
                    + column * patch_size
                )

                y = (
                    start_y
                    + row * patch_size
                )

                # -------------------------------------------------
                # Crop tissue and corresponding mask.
                # -------------------------------------------------

                tissue_patch = tissue_image.crop(
                    (
                        x,
                        y,
                        x + patch_size,
                        y + patch_size,
                    )
                )

                mask_patch = mask_image.crop(
                    (
                        x,
                        y,
                        x + patch_size,
                        y + patch_size,
                    )
                )

                patch_id = (
                    f"{slide_id}_{patch_number}"
                )

                tissue_destination = (
                    output_dir
                    / f"image_{slide_id}_patch_{patch_number}_tissue.png"
                )

                mask_destination = (
                    output_dir
                    / f"image_{slide_id}_patch_{patch_number}_mask.png"
                )

                tissue_patch.save(
                    tissue_destination
                )

                mask_patch.save(
                    mask_destination
                )

                # No augmentation for full-slide patches.
                magnification = 0
                degrees = 0

                append_csv_row(
                    tissue_csv,
                    {
                        "slide_id": slide_id,
                        "patch_id": patch_id,
                        "x": x,
                        "y": y,
                        "magnification": magnification,
                        "degrees": degrees,
                        "tissue_path": tissue_destination.name,
                    },
                )

                append_csv_row(
                    mask_csv,
                    {
                        "slide_id": slide_id,
                        "patch_id": patch_id,
                        "x": x,
                        "y": y,
                        "magnification": magnification,
                        "degrees": degrees,
                        "mask_path": mask_destination.name,
                    },
                )

                patch_number += 1

    print(
        f"Extracted {total_patches} patches "
        f"from slide {slide_id}."
    )

    return total_patches


def append_csv_row(path, row):
    """
    Adds one row to a CSV file.

    Creates the CSV and header if it does not exist.
    """

    path = Path(path)

    file_exists = (
        path.exists()
        and path.stat().st_size > 0
    )

    with open(
        path,
        "a",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=row.keys(),
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def get_slide_ids(csv_path):
    """
    Reads slide IDs from tissue_slides.csv.

    Returns them as a list of integers.
    """

    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Could not find: {csv_path}"
        )

    slide_ids = []

    with open(
        csv_path,
        "r",
        newline="",
        encoding="utf-8",
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:
            slide_ids.append(
                int(row["slide_id"])
            )

    return slide_ids


## set paths of the slides and the output paths ##
# Note to self: for HPC, use the correct (different) paths!
slides_dir = r"C:\Users\20234257\OneDrive - TU Eindhoven\BEP Project\external data\slides"
train_output_dir = r"C:\Users\20234257\OneDrive - TU Eindhoven\BEP Project\external data\training"
test_output_dir = r"C:\Users\20234257\OneDrive - TU Eindhoven\BEP Project\external data\testing"
val_output_dir = r"C:\Users\20234257\OneDrive - TU Eindhoven\BEP Project\external data\validation"


# code to get the slide id's from tissue_slides.csv

slides_csv = Path("data/slides/tissue_slides.csv")

image_id_list = get_slide_ids(slides_csv)


print(f"Found {len(image_id_list)} slide IDs:")
print(image_id_list)


## train-test-val split ##
# Note to self: maybe do random test/train/val split?
train_slide_id_list = [2152, 7522, 10010, 10925, 11712]
test_slide_id_list = [10955, 11525]
val_slide_id_list = [9943, 10535]


## test functionality ##
random_patch_train(
        slides_dir=slides_dir,
        train_output_dir=train_output_dir,
        slide_id=2152,
        n=1,
        magnification_change=0.1,
    )



## download random training patches ##
#for slide_id in train_slide_id_list:
#
#    random_patch_train(
#        slides_dir=slides_dir,
#        train_output_dir=train_output_dir,
#        slide_id=slide_id,
#        n=1,
#        magnification_change=0.1,
#    )


## download training/testing patches ##
# for slide_id in test_slide_id_list:
#
#     full_patch_extraction(
#         slides_dir=slides_dir,
#         output_dir=test_output_dir,
#         slide_id=slide_id,
#         group="test",
#     )


# for slide_id in val_slide_id_list:
#
#     full_patch_extraction(
#         slides_dir=slides_dir,
#         output_dir=val_output_dir,
#         slide_id=slide_id,
#         group="validation",
#     )