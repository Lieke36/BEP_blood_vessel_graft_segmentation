### Data processing v2 ###

# add set background - edges - foreground ratio

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
    background_ratio=None,
    edge_ratio=None,
    foreground_ratio=None,
):
    """
    Samples n random 512x512 tissue and mask patches from one slide.

    If background_ratio, edge_ratio and foreground_ratio are all None:
        patches are sampled completely randomly, as before.

    If ratios are provided:
        - background = final mask is completely black
        - foreground = final mask is completely white
        - edge = anything else

        The requested ratios determine the exact target number of
        patches in each category.

        Example:
            n=2000
            background_ratio=0.10
            edge_ratio=0.80
            foreground_ratio=0.10

        results in:
            200 background patches
            1600 edge patches
            200 foreground patches

    The mask is evaluated AFTER all geometric transformations
    (resize and rotation) have been applied.

    The tissue is only extracted and processed after the mask
    has been accepted.

    Rotation is performed using backward/inverse mapping.

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

    if (
        background_ratio is None
        and edge_ratio is None
        and foreground_ratio is None
    ):
        ratio_name = "random"

    else:
        ratio_name = (
            f"{int(background_ratio * 100)}_"
            f"{int(edge_ratio * 100)}_"
            f"{int(foreground_ratio * 100)}"
        )

    tissue_csv = (
        csv_dir
        / f"train_{ratio_name}_tissue_patches.csv"
    )

    mask_csv = (
        csv_dir
        / f"train_{ratio_name}_mask_patches.csv"
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
    # Check ratio settings.
    # ---------------------------------------------------------

    ratios = (
        background_ratio,
        edge_ratio,
        foreground_ratio,
    )

    if any(
        ratio is not None
        for ratio in ratios
    ):

        if not all(
            ratio is not None
            for ratio in ratios
        ):
            raise ValueError(
                "background_ratio, edge_ratio and "
                "foreground_ratio must either all be None "
                "or all be specified."
            )

        if any(
            ratio < 0 or ratio > 1
            for ratio in ratios
        ):
            raise ValueError(
                "Ratios must be between 0 and 1."
            )

        if not math.isclose(
            sum(ratios),
            1.0,
        ):
            raise ValueError(
                "background_ratio + edge_ratio + "
                "foreground_ratio must equal 1."
            )

        background_target = round(
            n * background_ratio
        )

        edge_target = round(
            n * edge_ratio
        )

        # Calculate foreground as the remainder so that
        # the three target numbers always add up exactly to n.
        foreground_target = (
            n
            - background_target
            - edge_target
        )

        background_count = 0
        edge_count = 0
        foreground_count = 0

        # Prevent an infinite loop if the slide does not
        # contain enough patches of a required category.
        max_attempts = max(
            n * 100,
            10000,
        )

        attempts = 0

    else:

        # No ratio restriction.
        background_target = None
        edge_target = None
        foreground_target = None

        background_count = None
        edge_count = None
        foreground_count = None

        max_attempts = None
        attempts = None

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

        patch_number = 0

        while patch_number < n:

            # -------------------------------------------------
            # Count attempts.
            # -------------------------------------------------

            if max_attempts is not None:

                attempts += 1

                if attempts > max_attempts:

                    raise RuntimeError(
                        f"Could not generate the requested "
                        f"training set for slide {slide_id} "
                        f"after {max_attempts} random attempts. "
                        f"Generated: "
                        f"{background_count} background, "
                        f"{edge_count} edge, "
                        f"{foreground_count} foreground. "
                        f"Target: "
                        f"{background_target} background, "
                        f"{edge_target} edge, "
                        f"{foreground_target} foreground."
                    )

            # -------------------------------------------------
            # Random magnification.
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
            # Determine source size.
            # -------------------------------------------------

            source_size = get_required_source_size(
                patch_size=patch_size,
                degrees=degrees,
                magnification=magnification,
            )

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
            # Crop ONLY the mask first.
            #
            # The tissue is deliberately not loaded/cropped yet.
            # -------------------------------------------------

            mask_patch = mask_image.crop(
                (
                    x,
                    y,
                    x + source_size,
                    y + source_size,
                )
            )

            mask_patch = mask_patch.convert(
                "L"
            )

            # -------------------------------------------------
            # Calculate transformed source size.
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
            # Resize MASK.
            #
            # NEAREST is important because this is a mask.
            # -------------------------------------------------

            mask_patch = mask_patch.resize(
                (
                    transformed_source_size,
                    transformed_source_size,
                ),
                Image.Resampling.NEAREST,
            )

            # -------------------------------------------------
            # Rotate MASK.
            #
            # This produces the actual 512x512 mask that will
            # be used for training.
            # -------------------------------------------------

            mask_patch = rotate_backward(
                image=mask_patch,
                degrees=degrees,
                output_size=patch_size,
                resample=Image.Resampling.NEAREST,
            )

            # -------------------------------------------------
            # Evaluate the FINAL transformed mask.
            #
            # Completely black = background
            # Completely white = foreground
            # Anything else = edge
            # -------------------------------------------------

            if (
                background_target is not None
            ):

                mask_min, mask_max = (
                    mask_patch.getextrema()
                )

                if mask_max == 0:

                    patch_category = "background"

                elif (
                    mask_min == 255
                    and mask_max == 255
                ):

                    patch_category = "foreground"

                else:

                    patch_category = "edge"

                # -------------------------------------------------
                # If this category already has enough samples,
                # discard this patch and try another random one.
                # -------------------------------------------------

                if (
                    patch_category == "background"
                    and background_count >= background_target
                ):

                    mask_patch.close()
                    continue

                if (
                    patch_category == "edge"
                    and edge_count >= edge_target
                ):

                    mask_patch.close()
                    continue

                if (
                    patch_category == "foreground"
                    and foreground_count >= foreground_target
                ):

                    mask_patch.close()
                    continue

            # -------------------------------------------------
            # The mask is accepted.
            #
            # NOW extract the tissue.
            # -------------------------------------------------

            tissue_patch = tissue_image.crop(
                (
                    x,
                    y,
                    x + source_size,
                    y + source_size,
                )
            )

            tissue_patch = tissue_patch.convert(
                "RGB"
            )

            # -------------------------------------------------
            # Resize TISSUE.
            # -------------------------------------------------

            tissue_patch = tissue_patch.resize(
                (
                    transformed_source_size,
                    transformed_source_size,
                ),
                Image.Resampling.LANCZOS,
            )

            # -------------------------------------------------
            # Rotate TISSUE.
            # -------------------------------------------------

            tissue_patch = rotate_backward(
                image=tissue_patch,
                degrees=degrees,
                output_size=patch_size,
                resample=Image.Resampling.BICUBIC,
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
            # Update category counter.
            # -------------------------------------------------

            if (
                background_target is not None
            ):

                if patch_category == "background":
                    background_count += 1

                elif patch_category == "edge":
                    edge_count += 1

                elif patch_category == "foreground":
                    foreground_count += 1

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

            patch_number += 1

    finally:

        # -----------------------------------------------------
        # Close original slide files.
        # -----------------------------------------------------

        tissue_image.close()
        mask_image.close()

    if (
        background_target is not None
    ):

        print(
            f"Created {n} training patches "
            f"for slide {slide_id}: "
            f"{background_count} background, "
            f"{edge_count} edge, "
            f"{foreground_count} foreground."
        )

    else:

        print(
            f"Created {n} random training patch(es) "
            f"for slide {slide_id}."
        )


def full_patch_extraction(
    slides_dir,
    output_dir,
    slide_id,
    group,
    overlap=0,
):
    """
    Extracts all complete 512x512 patches from one slide.

    The largest possible grid of complete patches is used.

    The grid is centered in the slide.

    overlap determines the overlap between neighboring patches:
        0.0 = no overlap, stride = 512 pixels
        0.5 = 50% overlap, stride = 256 pixels

    overlap must be between 0 and 0.5.

    Patch types are determined from the ground-truth mask:
        background = completely black mask
        foreground = completely white mask
        edge = anything else

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

    if not 0 <= overlap <= 0.5:
        raise ValueError(
            "overlap must be between 0 and 0.5."
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
        # Determine stride from overlap.
        # -----------------------------------------------------

        stride = int(
            patch_size * (1 - overlap)
        )

        # -----------------------------------------------------
        # Determine number of complete patches.
        # -----------------------------------------------------

        patches_x = math.floor(
            (width - patch_size) / stride
        ) + 1

        patches_y = math.floor(
            (height - patch_size) / stride
        ) + 1

        patches_x = max(patches_x, 0)
        patches_y = max(patches_y, 0)

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
            f"Overlap: {overlap}"
        )

        print(
            f"Stride: {stride} px"
        )

        print(
            f"Total patches: {total_patches}"
        )

        # -----------------------------------------------------
        # Determine size of centered grid.
        # -----------------------------------------------------

        if total_patches > 0:

            grid_width = (
                patch_size
                + (patches_x - 1) * stride
            )

            grid_height = (
                patch_size
                + (patches_y - 1) * stride
            )

        else:
            grid_width = 0
            grid_height = 0

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
                    + column * stride
                )

                y = (
                    start_y
                    + row * stride
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

                # -------------------------------------------------
                # Classify patch using ground-truth mask.
                #
                # Completely black  -> background
                # Completely white  -> foreground
                # Anything else     -> edge
                # -------------------------------------------------

                mask_min, mask_max = (
                    mask_patch.getextrema()
                )

                if mask_max == 0:

                    patch_type = "background"

                elif (
                    mask_min == 255
                    and mask_max == 255
                ):

                    patch_type = "foreground"

                else:

                    patch_type = "edge"

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
                        "patch_type": patch_type,
                        "tissue_path": tissue_destination.name,
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
                        "patch_type": patch_type,
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


# set ratios
# ratios: (fill in None for all for no set ratio)
background_ratio = None
edge_ratio = None
foreground_ratio = None

#ratio_name = (
#    f"{int(background_ratio * 100)}_"
#    f"{int(edge_ratio * 100)}_"
#    f"{int(foreground_ratio * 100)}"
#)


## set paths of the slides and the output paths ##
# Note to self: for HPC, use the correct (different) paths!
slides_dir = r"/home/20234257/BEP_Project/External_data/slides"
train_output_dir = r"/home/20234257/BEP_Project/External_data/training_random" # -> for data processing without set ratio
#train_output_dir = (
#    f"/home/20234257/BEP_Project/External_data/"
#    f"training_{ratio_name}"
#)
test_output_dir = r"/home/20234257/BEP_Project/External_data/testing"
val_output_dir = r"/home/20234257/BEP_Project/External_data/validation"


# code to get the slide id's from tissue_slides.csv

slides_csv = Path("data/slides/tissue_slides.csv") # No change needed for HPC

image_id_list = get_slide_ids(slides_csv)


print(f"Found {len(image_id_list)} slide IDs:")
print(image_id_list)


## train-test-val split ##
# Note to self: maybe do random test/train/val split?
train_slide_id_list = [
    9943, 10010, 10210, 10143, 10315, 9972, 10174,
    10955, 10925, 11014, 10858, 11057, 11133, 10765, 
    10782, 10535, 10361, 10404, 10449, 10284, 10503, 
    11525, 11418, 11629, 7522, 80490, 80501, 80589
]
test_slide_id_list = [2152, 11193, 10648, 11743, 80975, 80600]
val_slide_id_list = [10251, 11084, 10886, 11712, 11273, 80655]



## download random training patches ##

for slide_id in train_slide_id_list:
    random_patch_train(
        slides_dir=slides_dir,
        train_output_dir=train_output_dir,
        slide_id=slide_id,
        n=1000,
        magnification_change=0.1,
        background_ratio=background_ratio,
        edge_ratio=edge_ratio,
        foreground_ratio=foreground_ratio,
    )


## download validation/testing patches ##

#for slide_id in test_slide_id_list:
#    full_patch_extraction(
#        slides_dir=slides_dir,
#        output_dir=test_output_dir,
#        slide_id=slide_id,
#        group="test",
#        overlap=0.125,
#    )


#for slide_id in val_slide_id_list:
#    full_patch_extraction(
#        slides_dir=slides_dir,
#        output_dir=val_output_dir,
#        slide_id=slide_id,
#        group="validation",
#        overlap=0.125
#    )


end_time = time.time()
print(f"Runtime: {end_time - start_time:.2f} seconds")
