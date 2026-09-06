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
import time # to see runtime of data set

from PIL import Image

Image.MAX_IMAGE_PIXELS = None # avoid error due to large image (slide)

start_time = time.time()

## functions ##

def get_required_source_size(
    patch_size,
    degrees,
    magnification,
    extra_margin=4,
):
    """
    Calculates how large the source crop must be so that a
    rotated patch can completely fill the final output grid.

    For a square patch, the required bounding-box size is:

        patch_size * (abs(cos(theta)) + abs(sin(theta)))

    The magnification factor is then applied to this source
    size.

    A small extra margin is added to avoid interpolation
    touching the source boundary.
    """

    theta = math.radians(degrees)

    rotation_factor = (
        abs(math.cos(theta))
        + abs(math.sin(theta))
    )

    required_size = (
        patch_size
        * rotation_factor
        * magnification
    )

    required_size = math.ceil(required_size)

    # Small safety margin for interpolation.
    required_size += extra_margin

    return required_size


def rotate_backward(
    image,
    degrees,
    output_size=512,
    resample=Image.Resampling.BICUBIC,
):
    """
    Rotates an image using backward/inverse mapping.

    The output grid is created first.

    For every output pixel, the corresponding source coordinate
    is calculated and the source intensity is interpolated there.

    This is different from rotating a 512x512 canvas and then
    dealing with the empty corners.

    Parameters
    ----------
    image : PIL.Image
        Source image. It must be large enough to contain the
        complete rotated output region.

    degrees : float
        Rotation angle in degrees.

    output_size : int
        Final output width and height.

    resample : PIL.Image.Resampling
        Interpolation method.

    Returns
    -------
    PIL.Image
        Rotated image of exactly output_size x output_size.
    """

    source_width, source_height = image.size

    output_width = output_size
    output_height = output_size

    # ---------------------------------------------------------
    # Centre of source image.
    # ---------------------------------------------------------

    source_cx = (
        source_width - 1
    ) / 2.0

    source_cy = (
        source_height - 1
    ) / 2.0

    # ---------------------------------------------------------
    # Centre of output image.
    # ---------------------------------------------------------

    output_cx = (
        output_width - 1
    ) / 2.0

    output_cy = (
        output_height - 1
    ) / 2.0

    # ---------------------------------------------------------
    # Rotation angle.
    # ---------------------------------------------------------

    theta = math.radians(degrees)

    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)

    # ---------------------------------------------------------
    # Backward transformation.
    #
    # For every output pixel (x, y), calculate the corresponding
    # source coordinate.
    #
    # The inverse mapping is:
    #
    # source_x =
    #     cos(theta) * output_x
    #     + sin(theta) * output_y
    #
    # source_y =
    #     -sin(theta) * output_x
    #     + cos(theta) * output_y
    #
    # ---------------------------------------------------------

    a = cos_theta
    b = sin_theta

    c = (
        source_cx
        - cos_theta * output_cx
        - sin_theta * output_cy
    )

    d = -sin_theta
    e = cos_theta

    f = (
        source_cy
        + sin_theta * output_cx
        - cos_theta * output_cy
    )

    # ---------------------------------------------------------
    # Perform the inverse transformation.
    # ---------------------------------------------------------

    transformed = image.transform(
        (
            output_width,
            output_height,
        ),
        Image.Transform.AFFINE,
        (
            a,
            b,
            c,
            d,
            e,
            f,
        ),
        resample=resample,
    )

    return transformed


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
    - random magnification
    - random rotation
    - final output is always 512x512
    - tissue and mask receive exactly the same geometric
      transformation

    Rotation is performed using backward/inverse mapping.

    The important difference from the previous implementation is
    that the 512x512 output grid is determined first. Each output
    pixel is then mapped back to the source image.

    This prevents black corners caused by rotating a 512x512
    image canvas.

    The CSV x/y coordinates refer to the top-left corner of the
    larger source crop used for the transformation.
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

    tissue_csv = (
        csv_dir / "train_tissue_patches.csv"
    )

    mask_csv = (
        csv_dir / "train_mask_patches.csv"
    )

    tissue_path = (
        slides_dir
        / f"image_{slide_id}_tissue.png"
    )

    mask_path = (
        slides_dir
        / f"image_{slide_id}_mask.png"
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
    # Open slide images.
    # ---------------------------------------------------------

    tissue_image = Image.open(
        tissue_path
    )

    mask_image = Image.open(
        mask_path
    )

    try:

        # -----------------------------------------------------
        # Check that tissue and mask have identical dimensions.
        # -----------------------------------------------------

        if tissue_image.size != mask_image.size:

            raise ValueError(
                f"Tissue and mask sizes do not match "
                f"for slide {slide_id}: "
                f"{tissue_image.size} vs "
                f"{mask_image.size}"
            )

        width, height = tissue_image.size

        # -----------------------------------------------------
        # Generate random patches.
        # -----------------------------------------------------

        for patch_number in range(n):

            # -------------------------------------------------
            # Random magnification.
            #
            # This preserves the meaning of your original code:
            #
            # magnification > 1
            #     -> larger source region
            #
            # magnification < 1
            #     -> smaller source region
            # -------------------------------------------------

            magnification = random.uniform(
                1 - magnification_change,
                1 + magnification_change,
            )

            # -------------------------------------------------
            # Random rotation.
            # -------------------------------------------------

            degrees = random.uniform(
                0,
                360,
            )

            # -------------------------------------------------
            # Determine how large the source region needs to be.
            #
            # At 0 degrees:
            #
            #     source ~= 512 * magnification
            #
            # At 45 degrees:
            #
            #     source ~= 512 * sqrt(2) * magnification
            #
            # This gives enough source pixels to completely fill
            # the 512x512 output after rotation.
            # -------------------------------------------------

            source_size = get_required_source_size(
                patch_size=patch_size,
                degrees=degrees,
                magnification=magnification,
            )

            # -------------------------------------------------
            # Make sure the source crop can actually fit inside
            # the slide.
            #
            # We do NOT pad with black.
            #
            # If the slide is too small, there is no mathematically
            # valid way to create the requested fully populated
            # patch without going outside the slide.
            # -------------------------------------------------

            if source_size > width:

                raise ValueError(
                    f"Slide {slide_id} is too narrow for "
                    f"the requested transformation. "
                    f"Required source width: {source_size}, "
                    f"slide width: {width}."
                )

            if source_size > height:

                raise ValueError(
                    f"Slide {slide_id} is too short for "
                    f"the requested transformation. "
                    f"Required source height: {source_size}, "
                    f"slide height: {height}."
                )

            # -------------------------------------------------
            # Random top-left coordinate.
            #
            # Because we crop the larger source region, x/y now
            # describe that larger source region.
            # -------------------------------------------------

            max_x = (
                width - source_size
            )

            max_y = (
                height - source_size
            )

            x = random.randint(
                0,
                max_x,
            )

            y = random.randint(
                0,
                max_y,
            )

            # -------------------------------------------------
            # Crop the larger source region.
            #
            # This is the key to avoiding black corners.
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
            # Convert only the small crops.
            # -------------------------------------------------

            tissue_patch = tissue_patch.convert(
                "RGB"
            )

            mask_patch = mask_patch.convert(
                "L"
            )

            # -------------------------------------------------
            # Calculate the size of the source image after
            # magnification has been applied.
            #
            # We want the geometric relationship to be:
            #
            # large source crop
            #       ↓
            # magnification
            #       ↓
            # sufficiently large rotated source
            #       ↓
            # 512x512 output
            # -------------------------------------------------

            rotation_factor = (
                abs(
                    math.cos(
                        math.radians(degrees)
                    )
                )
                +
                abs(
                    math.sin(
                        math.radians(degrees)
                    )
                )
            )

            transformed_source_size = math.ceil(
                patch_size
                * rotation_factor
            )

            # Safety margin.
            transformed_source_size += 4

            # -------------------------------------------------
            # Resize the source crop.
            #
            # Tissue:
            #     LANCZOS
            #
            # Mask:
            #     NEAREST
            # -------------------------------------------------

            tissue_patch = tissue_patch.resize(
                (
                    transformed_source_size,
                    transformed_source_size,
                ),
                Image.Resampling.LANCZOS,
            )

            mask_patch = mask_patch.resize(
                (
                    transformed_source_size,
                    transformed_source_size,
                ),
                Image.Resampling.NEAREST,
            )

            # -------------------------------------------------
            # BACKWARD TRANSFORMATION
            #
            # The final 512x512 grid is created first.
            #
            # For every output pixel, rotate backwards to find
            # the corresponding source pixel.
            #
            # No black canvas is created.
            # -------------------------------------------------

            tissue_patch = rotate_backward(
                image=tissue_patch,
                degrees=degrees,
                output_size=patch_size,
                resample=Image.Resampling.BICUBIC,
            )

            mask_patch = rotate_backward(
                image=mask_patch,
                degrees=degrees,
                output_size=patch_size,
                resample=Image.Resampling.NEAREST,
            )

            # -------------------------------------------------
            # Safety check.
            # -------------------------------------------------

            if tissue_patch.size != (
                patch_size,
                patch_size,
            ):

                raise RuntimeError(
                    "Tissue patch has incorrect size: "
                    f"{tissue_patch.size}"
                )

            if mask_patch.size != (
                patch_size,
                patch_size,
            ):

                raise RuntimeError(
                    "Mask patch has incorrect size: "
                    f"{mask_patch.size}"
                )

            # -------------------------------------------------
            # Future stain augmentation.
            #
            # 0 = no stain augmentation
            # 1 = weak
            # 2 = medium
            # 3 = strong
            # -------------------------------------------------

            stain_intensity = 0

            # -------------------------------------------------
            # Patch ID.
            # -------------------------------------------------

            patch_id = (
                f"{slide_id}_{patch_number}"
            )

            # -------------------------------------------------
            # Output paths.
            # -------------------------------------------------

            tissue_destination = (
                train_output_dir
                / (
                    f"image_{slide_id}"
                    f"_patch_{patch_number}"
                    f"_tissue.png"
                )
            )

            mask_destination = (
                train_output_dir
                / (
                    f"image_{slide_id}"
                    f"_patch_{patch_number}"
                    f"_mask.png"
                )
            )

            # -------------------------------------------------
            # Save.
            # -------------------------------------------------

            tissue_patch.save(
                tissue_destination
            )

            mask_patch.save(
                mask_destination
            )

            # -------------------------------------------------
            # Close temporary images.
            # -------------------------------------------------

            tissue_patch.close()
            mask_patch.close()

            # -------------------------------------------------
            # Store tissue information.
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
                    "tissue_path": (
                        tissue_destination.name
                    ),
                },
            )

            # -------------------------------------------------
            # Store mask information.
            # -------------------------------------------------

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
                    "mask_path": (
                        mask_destination.name
                    ),
                },
            )

            print(
                f"Created training patch "
                f"{patch_id}: "
                f"x={x}, y={y}, "
                f"source_size={source_size}, "
                f"magnification="
                f"{magnification:.3f}, "
                f"degrees={degrees:.1f}"
            )

    finally:

        # -----------------------------------------------------
        # Close original slide files.
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
slides_dir = r"C:\Users\20234257\OneDrive - TU Eindhoven\BEP Project\external data\slides" # !!! Change for HPC (copy slides pngs to HPC data location) !!!
train_output_dir = r"C:\Users\20234257\OneDrive - TU Eindhoven\BEP Project\external data\training" # !!! Change for HPC !!!
test_output_dir = r"C:\Users\20234257\OneDrive - TU Eindhoven\BEP Project\external data\testing" # !!! Change for HPC !!!
val_output_dir = r"C:\Users\20234257\OneDrive - TU Eindhoven\BEP Project\external data\validation" # !!! Change for HPC !!!


# code to get the slide id's from tissue_slides.csv

slides_csv = Path("data/slides/tissue_slides.csv") # No change needed for HPC

image_id_list = get_slide_ids(slides_csv)


print(f"Found {len(image_id_list)} slide IDs:")
print(image_id_list)


## train-test-val split ##
# Note to self: maybe do random test/train/val split?
train_slide_id_list = [2152, 7522, 10010, 10925, 11712]
test_slide_id_list = [10955, 11525]
val_slide_id_list = [9943, 10535]



## download random training patches ##

for slide_id in train_slide_id_list:
    random_patch_train(
        slides_dir=slides_dir,
        train_output_dir=train_output_dir,
        slide_id=slide_id,
        n=1,
        magnification_change=0.1,
    )


## download training/testing patches ##

for slide_id in test_slide_id_list:
    full_patch_extraction(
        slides_dir=slides_dir,
        output_dir=test_output_dir,
        slide_id=slide_id,
        group="test",
    )


for slide_id in val_slide_id_list:
    full_patch_extraction(
        slides_dir=slides_dir,
        output_dir=val_output_dir,
        slide_id=slide_id,
        group="validation",
    )


end_time = time.time()
print(f"Runtime: {end_time - start_time:.2f} seconds")