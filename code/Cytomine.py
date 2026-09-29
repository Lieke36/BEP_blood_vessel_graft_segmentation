### Cytomine data extraction ###
# make sure the code is applicable to HPC GPU/CPU -> currently doesn't work

## libaries ##
# IMPORTANT: run [pip install -r requirements.txt] in terminal using "python -m pip install -r requirements.txt"

import os
from pathlib import Path
from getpass import getpass
from math import ceil

import pandas as pd
from PIL import Image, ImageDraw
from shapely import wkt
from shapely.geometry import Polygon, MultiPolygon

from cytomine import Cytomine
from cytomine.models import (
    ImageGroupCollection,
    ImageInstanceCollection,
    AnnotationCollection,
    TermCollection,
)

## Functions ##

def get_image_group_images(project_id, group_name="Annotated"):
    """Return Cytomine image objects from a named image group."""

    groups = ImageGroupCollection().fetch_with_filter("project", project_id)
    group = next((item for item in groups if item.name == group_name), None)

    if group is None:
        available = [item.name for item in groups]
        raise ValueError(
            f"No image group named {group_name!r} in project {project_id}. "
            f"Available groups: {available}"
        )

    return list(
        ImageInstanceCollection().fetch_with_filter("imagegroup", group.id)
    )



def _draw_geometry_on_mask(
    draw,
    geometry,
    min_x,
    max_y,
    scale_factor,
    fill,
):
    """
    Draw a Shapely geometry onto a PIL mask.

    Cytomine annotation coordinates use a bottom-left origin.
    PIL images use a top-left origin.

    Therefore the y-coordinate is flipped.

    Parameters
    ----------
    draw : PIL.ImageDraw.Draw
        Drawing object for the mask.

    geometry : shapely geometry
        Polygon or MultiPolygon.

    min_x : float
        Left boundary of the downloaded source region.

    max_y : float
        Top boundary of the downloaded source region in
        Cytomine coordinates.

    scale_factor : float
        Source-to-output scale factor.

    fill : int
        Pixel value to draw.
        255 = white / foreground
        0 = black / background
    """

    def transform_point(point):
        source_x, source_y = point

        output_x = (
            source_x - min_x
        ) / scale_factor

        output_y = (
            max_y - source_y
        ) / scale_factor

        return (
            int(round(output_x)),
            int(round(output_y)),
        )

    if isinstance(geometry, Polygon):

        # Draw polygon exterior.
        exterior = [
            transform_point(point)
            for point in geometry.exterior.coords
        ]

        draw.polygon(
            exterior,
            fill=fill,
        )

        # Draw polygon holes using the opposite colour.
        hole_fill = 0 if fill == 255 else 255

        for interior in geometry.interiors:

            hole = [
                transform_point(point)
                for point in interior.coords
            ]

            draw.polygon(
                hole,
                fill=hole_fill,
            )

    elif isinstance(geometry, MultiPolygon):

        for polygon in geometry.geoms:

            _draw_geometry_on_mask(
                draw=draw,
                geometry=polygon,
                min_x=min_x,
                max_y=max_y,
                scale_factor=scale_factor,
                fill=fill,
            )

    else:
        print(
            f"Warning: unsupported geometry type: "
            f"{geometry.geom_type}"
        )



def download_slides(
    project_id,
    margin=0.1,
    directory_slides="",
    target="Tissue area",
    group_name="Annotated",
    override=False,
    max_single_patch_size=9000,
    slide_id=None,
):
    """
    Download one tissue image and its binary mask for every image in an
    annotated Cytomine group.

    Bounding box:
        ALL annotations on the slide are used to determine the bounding box.

    Margin:
        The bounding box gets a margin of `margin` times its width and height
        on each side. For example, margin=0.1 adds 10% on every side.

    Tissue:
        The selected region is downloaded from Cytomine at approximately 10x.

    Mask:
        White  = target ("Tissue area")
        Black  = "Lumen" and everything else

        Lumen is drawn after Tissue area, so Lumen has priority if the
        annotations overlap.

    Large regions:
        Cytomine has problems with requests larger than approximately
        10000 x 10000 px. Large regions are therefore downloaded as a grid
        and stitched back together.

    CSV:
        Metadata is stored in:
            data/slides/tissue_slides.csv
            data/slides/masks_slides.csv

    slide_id:
        If supplied, only that slide is processed.
        If None, all slides in the Annotated group are processed.
    """

    from tempfile import TemporaryDirectory

    try:
        from PIL import Image, ImageDraw
    except ImportError as error:
        raise ImportError(
            "Install Pillow first with `pip install pillow`."
        ) from error

    import csv

    output_dir = Path(directory_slides)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_dir = Path("data/slides")
    csv_dir.mkdir(parents=True, exist_ok=True)

    tissue_csv = csv_dir / "tissue_slides.csv"
    mask_csv = csv_dir / "masks_slides.csv"

    # ---------------------------------------------------------
    # Find image group
    # ---------------------------------------------------------

    groups = ImageGroupCollection().fetch_with_filter(
        "project", project_id
    )

    group = next(
        (item for item in groups if item.name == group_name),
        None
    )

    if group is None:
        available = [item.name for item in groups]
        raise ValueError(
            f"No image group named {group_name!r} in project {project_id}. "
            f"Available groups: {available}"
        )

    images = ImageInstanceCollection().fetch_with_filter(
        "imagegroup", group.id
    )

    # ---------------------------------------------------------
    # Find target and Lumen terms
    # ---------------------------------------------------------

    from cytomine.models import TermCollection

    terms = TermCollection().fetch_with_filter(
        "project", project_id
    )

    target_term = next(
        (term for term in terms if term.name == target),
        None
    )

    lumen_term = next(
        (term for term in terms if term.name == "Lumen"),
        None
    )

    if target_term is None:
        available = [term.name for term in terms]
        raise ValueError(
            f"No term named {target!r} in project {project_id}. "
            f"Available terms: {available}"
        )

    if lumen_term is None:
        available = [term.name for term in terms]
        raise ValueError(
            f"No term named 'Lumen' in project {project_id}. "
            f"Available terms: {available}"
        )

    tissue_rows = []
    mask_rows = []

    # ---------------------------------------------------------
    # Process images
    # ---------------------------------------------------------

    for image in images:

        # If slide_id was supplied, skip all other slides.
        if slide_id is not None and image.id != slide_id:
            continue

        # -----------------------------------------------------
        # Get source magnification
        # -----------------------------------------------------

        source_magnification = getattr(
            image,
            "magnification",
            None
        )

        if source_magnification is None:
            raise ValueError(
                f"Image {image.id} does not have a source magnification. "
                f"Cannot reliably convert it to 10x."
            )

        try:
            source_magnification = float(source_magnification)
        except (TypeError, ValueError):
            raise ValueError(
                f"Invalid magnification for image {image.id}: "
                f"{source_magnification!r}"
            )

        if source_magnification <= 0:
            raise ValueError(
                f"Invalid magnification for image {image.id}: "
                f"{source_magnification}"
            )

        target_magnification = 10.0

        # How much should the image dimensions be reduced?
        #
        # 40x -> 10x = divide dimensions by 4
        # 20x -> 10x = divide dimensions by 2
        # 10x -> 10x = divide dimensions by 1
        scale_reduction = (
            source_magnification / target_magnification
        )

        print(
            f"Image {image.id}: "
            f"{source_magnification:g}x -> {target_magnification:g}x "
            f"(reduction factor {scale_reduction:g})"
        )

        # -----------------------------------------------------
        # Get ALL annotations.
        #
        # These annotations are ONLY used to determine the
        # bounding box.
        # -----------------------------------------------------

        all_annotations = AnnotationCollection(
            image=image.id,
            showWKT=True
        ).fetch()

        if not all_annotations:
            print(
                f"Skipping image {image.id}: "
                f"it has no annotations."
            )
            continue

        all_bounds = [
            wkt.loads(annotation.location).bounds
            for annotation in all_annotations
            if annotation.location
        ]

        if not all_bounds:
            print(
                f"Skipping image {image.id}: "
                f"no annotations have valid WKT."
            )
            continue

        # -----------------------------------------------------
        # Bounding box of ALL annotations
        # -----------------------------------------------------

        min_x = min(bound[0] for bound in all_bounds)
        min_y = min(bound[1] for bound in all_bounds)
        max_x = max(bound[2] for bound in all_bounds)
        max_y = max(bound[3] for bound in all_bounds)

        annotation_width = max_x - min_x
        annotation_height = max_y - min_y

        # 10% of annotation width/height on EACH side.
        x_margin = annotation_width * margin
        y_margin = annotation_height * margin

        min_x = max(0, min_x - x_margin)
        max_x = min(image.width, max_x + x_margin)

        min_y = max(0, min_y - y_margin)
        max_y = min(image.height, max_y + y_margin)

        box_width = ceil(max_x - min_x)
        box_height = ceil(max_y - min_y)

        if box_width <= 0 or box_height <= 0:
            print(
                f"Skipping image {image.id}: "
                f"invalid bounding box."
            )
            continue

        if box_width > image.width or box_height > image.height:
            raise ValueError(
                f"Bounding box for image {image.id} is larger "
                f"than the image itself."
            )

        # -----------------------------------------------------
        # Get Tissue area annotations for the mask.
        #
        # IMPORTANT:
        # We do NOT use these to determine the bounding box.
        # -----------------------------------------------------

        tissue_annotations = AnnotationCollection(
            image=image.id,
            term=target_term.id,
            showWKT=True
        ).fetch()

        lumen_annotations = AnnotationCollection(
            image=image.id,
            term=lumen_term.id,
            showWKT=True
        ).fetch()

        if not tissue_annotations:
            print(
                f"Skipping image {image.id}: "
                f"no '{target}' annotations."
            )
            continue

        tissue_geometries = [
            wkt.loads(annotation.location)
            for annotation in tissue_annotations
            if annotation.location
        ]

        lumen_geometries = [
            wkt.loads(annotation.location)
            for annotation in lumen_annotations
            if annotation.location
        ]

        # -----------------------------------------------------
        # Convert Cytomine bottom-left coordinates to the
        # top-left coordinates required by image.window().
        # -----------------------------------------------------

        x = round(min_x)

        y = round(image.height - max_y)

        # Make sure the requested region stays inside the image.
        x = min(
            max(0, x),
            image.width - box_width
        )

        y = min(
            max(0, y),
            image.height - box_height
        )

        # -----------------------------------------------------
        # Determine grid size.
        #
        # The original working logic is retained here.
        # -----------------------------------------------------

        if box_width <= max_single_patch_size and box_height <= max_single_patch_size:
            grid_x = 1
            grid_y = 1

        else:
            grid_x = ceil(
                box_width / max_single_patch_size
            )

            grid_y = ceil(
                box_height / max_single_patch_size
            )

        tile_width = ceil(box_width / grid_x)
        tile_height = ceil(box_height / grid_y)

        # -----------------------------------------------------
        # Download tissue
        # -----------------------------------------------------

        destination = (
            output_dir
            / f"image_{image.id}_tissue.png"
        )

        tile_images = []

        with TemporaryDirectory(
            prefix=f"cytomine_image_{image.id}_"
        ) as temporary_dir:

            temporary_dir = Path(temporary_dir)

            for row in range(grid_y):

                for column in range(grid_x):

                    offset_x = column * tile_width
                    offset_y = row * tile_height

                    current_width = min(
                        tile_width,
                        box_width - offset_x
                    )

                    current_height = min(
                        tile_height,
                        box_height - offset_y
                    )

                    temporary_tile = (
                        temporary_dir
                        / f"tile_{row}_{column}.png"
                    )

                    # Request the tile at the resolution needed
                    # to obtain 10x.
                    #
                    # 40x -> 10x = divide by 4
                    # 20x -> 10x = divide by 2
                    # 10x -> 10x = divide by 1
                    #
                    # max_size is applied to the larger dimension.
                    requested_max_size = ceil(
                        max(current_width, current_height)
                        / scale_reduction
                    )

                    downloaded = image.window(
                        x + offset_x,
                        y + offset_y,
                        current_width,
                        current_height,
                        str(temporary_tile),
                        override=True,
                        max_size=requested_max_size
                    )

                    if not downloaded:
                        print(
                            f"Could not retrieve tile "
                            f"{row}, {column} for image "
                            f"{image.id}."
                        )

                        tile_images = []
                        break

                    with Image.open(temporary_tile) as tile:
                        tile_images.append(
                            (
                                offset_x,
                                offset_y,
                                tile.convert("RGB")
                            )
                        )

                if not tile_images:
                    break

            expected_tiles = grid_x * grid_y

            if len(tile_images) != expected_tiles:
                print(
                    f"Skipping image {image.id}: "
                    f"not all tiles were retrieved."
                )
                continue

            # -------------------------------------------------
            # Determine actual Cytomine scale from first tile.
            #
            # This avoids assuming that Cytomine returns
            # exactly the requested number of pixels.
            # -------------------------------------------------

            scale_x = (
                tile_images[0][2].width
                / tile_width
            )

            scale_y = (
                tile_images[0][2].height
                / tile_height
            )

            # Use the same scale for the complete stitched image.
            scale = (scale_x + scale_y) / 2

            stitched_width = round(
                box_width * scale
            )

            stitched_height = round(
                box_height * scale
            )

            stitched = Image.new(
                "RGB",
                (stitched_width, stitched_height)
            )

            for offset_x, offset_y, tile in tile_images:

                paste_x = round(
                    offset_x * scale
                )

                paste_y = round(
                    offset_y * scale
                )

                stitched.paste(
                    tile,
                    (paste_x, paste_y)
                )

            stitched.save(destination)
            stitched.close()

        # -----------------------------------------------------
        # Create mask
        # -----------------------------------------------------

        mask = Image.new(
            "L",
            (stitched_width, stitched_height),
            0
        )

        draw = ImageDraw.Draw(mask)

        # -----------------------------------------------------
        # Convert a Shapely geometry from Cytomine coordinates
        # to the coordinates of the downloaded 10x image.
        # -----------------------------------------------------

        def transform_point(point_x, point_y):
            local_x = (
                (point_x - x)
                * scale
            )

            # Cytomine WKT has bottom-left origin.
            # Image/mask has top-left origin.
            local_y = (
                (max_y - point_y)
                * scale
            )

            return (
                round(local_x),
                round(local_y)
            )

        def draw_geometry(geometry, value):
            if geometry.geom_type == "Polygon":

                exterior = [
                    transform_point(px, py)
                    for px, py in geometry.exterior.coords
                ]

                draw.polygon(
                    exterior,
                    fill=value
                )

                # Polygon holes remain black.
                for interior in geometry.interiors:

                    hole = [
                        transform_point(px, py)
                        for px, py in interior.coords
                    ]

                    draw.polygon(
                        hole,
                        fill=0
                    )

            elif geometry.geom_type == "MultiPolygon":

                for polygon in geometry.geoms:
                    draw_geometry(
                        polygon,
                        value
                    )

            elif geometry.geom_type == "GeometryCollection":

                for geometry_part in geometry.geoms:
                    draw_geometry(
                        geometry_part,
                        value
                    )

        # -----------------------------------------------------
        # Tissue area = WHITE
        # -----------------------------------------------------

        for geometry in tissue_geometries:
            draw_geometry(
                geometry,
                255
            )

        # -----------------------------------------------------
        # Lumen = BLACK
        #
        # Drawn AFTER Tissue area, so Lumen has priority.
        # -----------------------------------------------------

        for geometry in lumen_geometries:
            draw_geometry(
                geometry,
                0
            )

        # -----------------------------------------------------
        # Save mask
        # -----------------------------------------------------

        mask_destination = (
            output_dir
            / f"image_{image.id}_mask.png"
        )

        mask.save(mask_destination)
        mask.close()

        # -----------------------------------------------------
        # CSV information
        # -----------------------------------------------------

        tissue_rows.append({
            "slide_id": image.id,
            "filename": image.instanceFilename,
            "tissue_path": str(destination),
            "width": stitched_width,
            "height": stitched_height,
            "magnification": target_magnification,
            "source_magnification": source_magnification,
            "min_x": min_x,
            "min_y": min_y,
            "max_x": max_x,
            "max_y": max_y,
            "margin": margin,
        })

        mask_rows.append({
            "slide_id": image.id,
            "filename": image.instanceFilename,
            "mask_path": str(mask_destination),
            "width": stitched_width,
            "height": stitched_height,
            "magnification": target_magnification,
            "source_magnification": source_magnification,
            "min_x": min_x,
            "min_y": min_y,
            "max_x": max_x,
            "max_y": max_y,
            "margin": margin,
        })

        print(
            f"Downloaded image {image.id}: "
            f"{stitched_width} x {stitched_height} px "
            f"({source_magnification:g}x -> {target_magnification:g}x)."
        )

    # ---------------------------------------------------------
    # Write CSV files
    # ---------------------------------------------------------

    def append_csv(path, rows):

        if not rows:
            return

        file_exists = (
            path.exists()
            and path.stat().st_size > 0
        )

        with open(
            path,
            "a",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=rows[0].keys()
            )

            if not file_exists:
                writer.writeheader()

            writer.writerows(rows)

    append_csv(
        tissue_csv,
        tissue_rows
    )

    append_csv(
        mask_csv,
        mask_rows
    )

    print(
        f"Downloaded {len(tissue_rows)} slide(s)."
    )

    return tissue_rows, mask_rows




## connect to cytomine ##

HOST = "http://131.155.65.6:8080/"
PROJECT_ID = 1903  # Replace with the Cytomine project ID that contains 'Annotated'.

# for HPC use: /home/20234257/BEP_Project/External_data/slides
# for laptop storage use: C:\Users\20234257\OneDrive - TU Eindhoven\BEP Project\external data\slides
OUTPUT_DIR = Path(r"C:\Users\20234257\OneDrive - TU Eindhoven\BEP Project\external data\slides") # place to store data



public_key = getpass("Cytomine public key: ")
private_key = getpass("Cytomine private key: ")

cytomine = Cytomine.connect(HOST, public_key, private_key)
print(f"Connected as {cytomine.current_user}")

## get annotated image group ##
image_id_list = []

annotated_images = get_image_group_images(PROJECT_ID)
for image in annotated_images:
    image_id_list.append(image.id)
    print(image.id, image.instanceFilename, image.width, image.height)

## download full annotated images and masks ##

image_id_list_downloaded = [10955, 2152, 10535, 10925, 11712, 9943, 10010, 11525, 10064, 11014, 10210, 10886, 11057, 11084, 11273, 10251, 10858] #18/19

# image 2143 wont work because there is no "Tissue area" annotations

new_image_ids = [7522]

for slide in image_id_list:
    if slide not in image_id_list_downloaded:

        check = input(f"Proceed with slide {slide}? Enter 'yes': ")
        if check.lower() != "yes":
            break

        download_slides(
            project_id=PROJECT_ID,
            margin=0.1,
            directory_slides=OUTPUT_DIR,
            target="Tissue area",
            slide_id=slide,
        )

        new_image_ids.append(slide)

print("\nNew image IDs downloaded:")
for image_id in new_image_ids:
    print(image_id)





## visualize annotated mask over tissue to show its aligned ##
