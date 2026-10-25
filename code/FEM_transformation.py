### FEM Transformation ###

# What does FEM_transformation.py do?
# FEM mapping of a vessel cross-section to an ideal annulus.
#
# Each angular sector consists of:
#
#   1. A 4-node quadrilateral for the tissue
#   2. A 3-node triangle for the lumen
#
# IMPORTANT:
# The lumen center C is ONE SINGLE FEM NODE shared by
# every triangular lumen element.
#
#
# Original geometry:
#
#   inner nodes = first mask pixels encountered along each ray
#   outer nodes = last mask pixels encountered along each ray
#   lumen center = centroid/mean of all inner boundary nodes
#
# Target geometry:
#
#   inner nodes -> circle with average inner radius
#   outer nodes -> circle with average outer radius
#   lumen center -> center of ideal annulus
#
# The image is transformed using an inverse FEM mapping:
#
#   target point
#       |
#       v
#   find FEM element
#       |
#       v
#   calculate (xi, eta)
#       |
#       v
#   evaluate same (xi, eta) in original element
#       |
#       v
#   interpolate original image
#
# Assumption:
#   mask > 0 = tissue
#   mask = 0 inside lumen and outside vessel
#
# The lumen itself is therefore transformed together with the tissue.
#
# Issue: Very deformed bloodvessels are unable to be analysed (see slide 10010 for example)

# How to use FEM_transformation.py?
# 1. Set SLIDE_ID to the slide id of the slide you want to analysis using FEM transformation.
# 2. Set paths: TISSUE_PATH should go to the correct image tissue slide, MASK_PATH should go to the correct image mask slide, OUTPUT_DIR should go to the folder where the maps and graphs should be stored.


## libaries ##

from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage
import matplotlib.pyplot as plt
import pandas as pd
import time


from PIL import Image

Image.MAX_IMAGE_PIXELS = None


## Settings ##
SLIDE_ID = "11743"

TISSUE_PATH = Path(
    rf"C:\Users\20234257\OneDrive - TU Eindhoven\BEP Project\external data\slides\image_{SLIDE_ID}_tissue.png"
)
MASK_PATH = Path(
    rf"C:\Users\20234257\OneDrive - TU Eindhoven\BEP Project\external data\slides\image_{SLIDE_ID}_mask.png"
)
OUTPUT_DIR = Path(
    r"C:\Users\20234257\OneDrive - TU Eindhoven\BEP Project\external data\slides\FEM_transformation"
)


## IMAGE LOADING ##

def load_and_preprocess_images(
    tissue_path: Path,
    mask_path: Path,
):
    """
    Load tissue and binary mask.

    The tissue can be grayscale or RGB.
    The mask is converted to a binary 2D array.
    """

    tissue = np.asarray(
        Image.open(tissue_path)
    )

    mask = np.asarray(
        Image.open(mask_path)
    )

    # Convert mask to grayscale
    if mask.ndim == 3:
        mask_gray = mask[..., 0]
    else:
        mask_gray = mask

    # Binary mask:
    # True  = tissue
    # False = lumen/background
    mask_binary = mask_gray > 0

    if tissue.shape[:2] != mask_binary.shape:
        raise ValueError(
            "Tissue and mask images must have the same "
            "X/Y dimensions."
        )

    tissue_float = tissue.astype(
        np.float32
    )

    return tissue_float, mask_binary


## FIND INITIAL MASK CENTER ##

def find_mask_center(
    mask_binary: np.ndarray,
) -> tuple[float, float]:
    """
    Find the centroid of the entire tissue mask.

    This is used as the initial ray origin.

    The actual FEM lumen node is calculated separately
    from the inner boundary nodes.
    """

    y_mask, x_mask = np.nonzero(
        mask_binary
    )

    if len(x_mask) == 0:
        raise ValueError(
            "Mask contains no foreground pixels."
        )

    center_x = float(
        x_mask.mean()
    )

    center_y = float(
        y_mask.mean()
    )

    return center_x, center_y


## FIND INNER AND OUTER BOUNDARY NODES ##

def find_boundary_nodes(
    mask_binary: np.ndarray,
    ray_center_x: float,
    ray_center_y: float,
    n_angles: int = 720,
    max_radius: float = 7000.0,
):
    """
    Find inner and outer mask boundary nodes.

    For every angular direction:

        inner node = first tissue pixel
        outer node = last tissue pixel

    The ray center is only used to find the boundaries.

    The FEM lumen center is calculated later from the
    resulting inner nodes.

    Returns
    -------
    inner_nodes : ndarray
        Shape (n_angles, 2), containing x,y coordinates.

    outer_nodes : ndarray
        Shape (n_angles, 2), containing x,y coordinates.

    valid_rays : ndarray
        Boolean array indicating successful rays.
    """

    angles = np.linspace(
        0,
        2 * np.pi,
        n_angles,
        endpoint=False,
    )

    search_radii = np.arange(
        0,
        max_radius,
        dtype=np.float32,
    )

    inner_nodes = np.full(
        (n_angles, 2),
        np.nan,
        dtype=np.float32,
    )

    outer_nodes = np.full(
        (n_angles, 2),
        np.nan,
        dtype=np.float32,
    )

    valid_rays = np.zeros(
        n_angles,
        dtype=bool,
    )

    height, width = mask_binary.shape

    for i, theta in enumerate(angles):

        # ----------------------------------------------------
        # Cast ray
        # ----------------------------------------------------

        x = (
            ray_center_x
            + search_radii * np.cos(theta)
        )

        y = (
            ray_center_y
            + search_radii * np.sin(theta)
        )

        x_int = np.round(x).astype(int)
        y_int = np.round(y).astype(int)

        valid = (
            (x_int >= 0)
            & (x_int < width)
            & (y_int >= 0)
            & (y_int < height)
        )

        radial_mask = np.zeros(
            search_radii.shape,
            dtype=bool,
        )

        radial_mask[valid] = mask_binary[
            y_int[valid],
            x_int[valid],
        ]

        foreground_indices = np.flatnonzero(
            radial_mask
        )

        if len(foreground_indices) == 0:
            continue

        first_idx = foreground_indices[0]
        last_idx = foreground_indices[-1]

        r_inner = search_radii[first_idx]
        r_outer = search_radii[last_idx]

        if r_outer <= r_inner:
            continue

        # ----------------------------------------------------
        # Store Cartesian coordinates
        # ----------------------------------------------------

        inner_nodes[i] = [
            ray_center_x
            + r_inner * np.cos(theta),

            ray_center_y
            + r_inner * np.sin(theta),
        ]

        outer_nodes[i] = [
            ray_center_x
            + r_outer * np.cos(theta),

            ray_center_y
            + r_outer * np.sin(theta),
        ]

        valid_rays[i] = True

    # --------------------------------------------------------
    # Fill missing rays using nearest angular neighbor
    # --------------------------------------------------------

    if not np.all(valid_rays):

        if not np.any(valid_rays):
            raise ValueError(
                "No valid vessel boundaries were found."
            )

        valid_indices = np.flatnonzero(
            valid_rays
        )

        for idx in np.flatnonzero(
            ~valid_rays
        ):

            circular_distance = np.abs(
                (
                    valid_indices
                    - idx
                    + n_angles // 2
                )
                % n_angles
                - n_angles // 2
            )

            nearest = valid_indices[
                np.argmin(
                    circular_distance
                )
            ]

            inner_nodes[idx] = (
                inner_nodes[nearest]
            )

            outer_nodes[idx] = (
                outer_nodes[nearest]
            )

    return (
        inner_nodes,
        outer_nodes,
        valid_rays,
    )


## CALCULATE LUMEN CENTER ##

def calculate_lumen_center(
    inner_nodes: np.ndarray,
):
    """
    Calculate ONE shared FEM node for the lumen.

    The center is calculated as the mean of all inner
    boundary nodes.

    This point is shared by EVERY triangular lumen element.
    """

    lumen_center_x = float(
        np.mean(inner_nodes[:, 0])
    )

    lumen_center_y = float(
        np.mean(inner_nodes[:, 1])
    )

    return (
        lumen_center_x,
        lumen_center_y,
    )


## CALCULATE AVERAGE RADII ##

def calculate_average_radii(
    inner_nodes: np.ndarray,
    outer_nodes: np.ndarray,
    center_x: float,
    center_y: float,
):
    """
    Calculate average inner and outer radius.

    The radii are measured relative to the lumen center.

    This means the ideal annulus is centered on the FEM
    lumen node.
    """

    inner_dx = (
        inner_nodes[:, 0]
        - center_x
    )

    inner_dy = (
        inner_nodes[:, 1]
        - center_y
    )

    outer_dx = (
        outer_nodes[:, 0]
        - center_x
    )

    outer_dy = (
        outer_nodes[:, 1]
        - center_y
    )

    inner_radii = np.sqrt(
        inner_dx**2
        + inner_dy**2
    )

    outer_radii = np.sqrt(
        outer_dx**2
        + outer_dy**2
    )

    r_inner = float(
        np.mean(inner_radii)
    )

    r_outer = float(
        np.mean(outer_radii)
    )

    if r_outer <= r_inner:
        raise ValueError(
            "Average outer radius must be larger "
            "than average inner radius."
        )

    return (
        r_inner,
        r_outer,
    )


def calculate_target_canvas(
    r_outer: float,
    margin: float = 10.0,
):
    """
    Calculate a square output canvas large enough to contain
    the complete ideal annulus.

    Parameters
    ----------
    r_outer : float
        Outer radius of the ideal annulus.

    margin : float
        Extra pixels around the outer boundary.

    Returns
    -------
    canvas_size : tuple[int, int]
        (height, width)

    target_center : tuple[float, float]
        Center of the ideal annulus in the output image.
    """

    if r_outer <= 0:
        raise ValueError(
            "Outer radius must be positive."
        )

    # Diameter + margin on both sides
    dimension = int(
        np.ceil(
            2 * r_outer
            + 2 * margin
        )
    )

    canvas_size = (
        dimension,
        dimension,
    )

    # Put the ideal annulus exactly in the
    # center of the output image.
    target_center = (
        (dimension - 1) / 2.0,
        (dimension - 1) / 2.0,
    )

    return (
        canvas_size,
        target_center,
    )


## BILINEAR SHAPE FUNCTIONS ##

def bilinear_shape_functions(
    xi,
    eta,
):
    """
    Four-node quadrilateral shape functions.

    Node ordering:

        4 -------- 3
        |          |
        |          |
        1 -------- 2
    """

    N1 = 0.25 * (
        1 - xi
    ) * (
        1 - eta
    )

    N2 = 0.25 * (
        1 + xi
    ) * (
        1 - eta
    )

    N3 = 0.25 * (
        1 + xi
    ) * (
        1 + eta
    )

    N4 = 0.25 * (
        1 - xi
    ) * (
        1 + eta
    )

    return (
        N1,
        N2,
        N3,
        N4,
    )


## BILINEAR QUAD MAPPING ##

def bilinear_map(
    nodes: np.ndarray,
    xi,
    eta,
):
    """
    Evaluate a 4-node bilinear FEM element.

    nodes shape:
        (4, 2)

    Returns:
        (..., 2)
    """

    N1, N2, N3, N4 = (
        bilinear_shape_functions(
            xi,
            eta,
        )
    )

    x = (
        N1 * nodes[0, 0]
        + N2 * nodes[1, 0]
        + N3 * nodes[2, 0]
        + N4 * nodes[3, 0]
    )

    y = (
        N1 * nodes[0, 1]
        + N2 * nodes[1, 1]
        + N3 * nodes[2, 1]
        + N4 * nodes[3, 1]
    )

    return np.stack(
        [x, y],
        axis=-1,
    )


## INVERSE BILINEAR MAPPING ##

def inverse_bilinear_map(
    nodes: np.ndarray,
    target_x,
    target_y,
    iterations: int = 8,
):
    """
    Find natural coordinates (xi, eta) for points inside
    a four-node quadrilateral.

    Newton-Raphson is used to solve:

        target = FEM(xi, eta)

    Returns
    -------
    xi
    eta
    """

    # --------------------------------------------------------
    # Initial guess
    # --------------------------------------------------------

    xi = np.zeros_like(
        target_x,
        dtype=np.float32,
    )

    eta = np.zeros_like(
        target_y,
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # Newton iterations
    # --------------------------------------------------------

    for _ in range(iterations):

        N1, N2, N3, N4 = (
            bilinear_shape_functions(
                xi,
                eta,
            )
        )

        # Current mapped point
        x = (
            N1 * nodes[0, 0]
            + N2 * nodes[1, 0]
            + N3 * nodes[2, 0]
            + N4 * nodes[3, 0]
        )

        y = (
            N1 * nodes[0, 1]
            + N2 * nodes[1, 1]
            + N3 * nodes[2, 1]
            + N4 * nodes[3, 1]
        )

        # ----------------------------------------------------
        # Derivatives of shape functions
        # ----------------------------------------------------

        dN1_dxi = (
            -0.25 * (1 - eta)
        )

        dN2_dxi = (
            0.25 * (1 - eta)
        )

        dN3_dxi = (
            0.25 * (1 + eta)
        )

        dN4_dxi = (
            -0.25 * (1 + eta)
        )

        dN1_deta = (
            -0.25 * (1 - xi)
        )

        dN2_deta = (
            -0.25 * (1 + xi)
        )

        dN3_deta = (
            0.25 * (1 + xi)
        )

        dN4_deta = (
            0.25 * (1 - xi)
        )

        # ----------------------------------------------------
        # Jacobian
        # ----------------------------------------------------

        dx_dxi = (
            dN1_dxi * nodes[0, 0]
            + dN2_dxi * nodes[1, 0]
            + dN3_dxi * nodes[2, 0]
            + dN4_dxi * nodes[3, 0]
        )

        dx_deta = (
            dN1_deta * nodes[0, 0]
            + dN2_deta * nodes[1, 0]
            + dN3_deta * nodes[2, 0]
            + dN4_deta * nodes[3, 0]
        )

        dy_dxi = (
            dN1_dxi * nodes[0, 1]
            + dN2_dxi * nodes[1, 1]
            + dN3_dxi * nodes[2, 1]
            + dN4_dxi * nodes[3, 1]
        )

        dy_deta = (
            dN1_deta * nodes[0, 1]
            + dN2_deta * nodes[1, 1]
            + dN3_deta * nodes[2, 1]
            + dN4_deta * nodes[3, 1]
        )

        # ----------------------------------------------------
        # Residual
        # ----------------------------------------------------

        residual_x = (
            target_x - x
        )

        residual_y = (
            target_y - y
        )

        determinant = (
            dx_dxi * dy_deta
            - dx_deta * dy_dxi
        )

        # Avoid division by zero
        determinant = np.where(
            np.abs(determinant) < 1e-8,
            np.sign(determinant) * 1e-8
            + (determinant == 0) * 1e-8,
            determinant,
        )

        # ----------------------------------------------------
        # Newton correction
        # ----------------------------------------------------

        delta_xi = (
            residual_x * dy_deta
            - residual_y * dx_deta
        ) / determinant

        delta_eta = (
            dx_dxi * residual_y
            - dy_dxi * residual_x
        ) / determinant

        xi += delta_xi
        eta += delta_eta

    return (
        xi,
        eta,
    )


## TRIANGLE BARYCENTRIC COORDINATES ##

def triangle_barycentric(
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    x,
    y,
):
    """
    Calculate barycentric coordinates of points inside a
    triangle.

    Returns:
        lambda1, lambda2, lambda3
    """

    denominator = (
        (p2[1] - p3[1])
        * (p1[0] - p3[0])
        + (p3[0] - p2[0])
        * (p1[1] - p3[1])
    )

    if abs(denominator) < 1e-10:
        raise ValueError(
            "Degenerate triangle encountered."
        )

    lambda1 = (
        (
            (p2[1] - p3[1])
            * (x - p3[0])
            + (p3[0] - p2[0])
            * (y - p3[1])
        )
        / denominator
    )

    lambda2 = (
        (
            (p3[1] - p1[1])
            * (x - p3[0])
            + (p1[0] - p3[0])
            * (y - p3[1])
        )
        / denominator
    )

    lambda3 = (
        1
        - lambda1
        - lambda2
    )

    return (
        lambda1,
        lambda2,
        lambda3,
    )


## TRIANGLE MAPPING ##

def triangle_map(
    source_nodes: np.ndarray,
    target_nodes: np.ndarray,
    x,
    y,
):
    """
    Map target triangle coordinates back to the source
    triangle using barycentric coordinates.

    source_nodes:
        (3, 2)

    target_nodes:
        (3, 2)
    """

    l1, l2, l3 = (
        triangle_barycentric(
            target_nodes[0],
            target_nodes[1],
            target_nodes[2],
            x,
            y,
        )
    )

    source_x = (
        l1 * source_nodes[0, 0]
        + l2 * source_nodes[1, 0]
        + l3 * source_nodes[2, 0]
    )

    source_y = (
        l1 * source_nodes[0, 1]
        + l2 * source_nodes[1, 1]
        + l3 * source_nodes[2, 1]
    )

    return (
        source_x,
        source_y,
    )


## INTERPOLATE IMAGE ##

def interpolate_image(
    tissue_padded: np.ndarray,
    sample_x,
    sample_y,
    interpolation_order: int = 1,
):
    """
    Interpolate an RGB or grayscale image.
    """

    sample_x = (
        sample_x + 1.0
    )

    sample_y = (
        sample_y + 1.0
    )

    if tissue_padded.ndim == 3:

        result = np.empty(
            (
                sample_x.size,
                tissue_padded.shape[2],
            ),
            dtype=np.float32,
        )

        for ch in range(
            tissue_padded.shape[2]
        ):

            result[:, ch] = (
                ndimage.map_coordinates(
                    tissue_padded[..., ch],
                    [
                        sample_y.ravel(),
                        sample_x.ravel(),
                    ],
                    order=interpolation_order,
                    mode="nearest",
                )
            )

        return result

    return ndimage.map_coordinates(
        tissue_padded,
        [
            sample_y.ravel(),
            sample_x.ravel(),
        ],
        order=interpolation_order,
        mode="nearest",
    )


## FEM RECONSTRUCTION ##

def fem_reconstruct_annulus(
    tissue: np.ndarray,
    inner_nodes: np.ndarray,
    outer_nodes: np.ndarray,
    lumen_center: tuple[float, float],
    target_center: tuple[float, float],
    r_inner: float,
    r_outer: float,
    canvas_size: tuple[int, int],
    interpolation_order: int = 1,
):
    """
    Reconstruct the tissue + lumen in an ideal annulus.

    Every angular section contains:

        Triangle:
            inner[i]
            inner[i+1]
            lumen_center

        Quadrilateral:
            inner[i]
            inner[i+1]
            outer[i+1]
            outer[i]

    The lumen-center node is shared by all triangles.
    """

    height, width = canvas_size

    # --------------------------------------------------------
    # Prepare output
    # --------------------------------------------------------

    if tissue.ndim == 3:

        channels = tissue.shape[2]

        output = np.zeros(
            (
                height,
                width,
                channels,
            ),
            dtype=np.float32,
        )

    else:

        output = np.zeros(
            (
                height,
                width,
            ),
            dtype=np.float32,
        )

    # --------------------------------------------------------
    # Pad original tissue image
    # --------------------------------------------------------

    if tissue.ndim == 3:

        tissue_padded = np.pad(
            tissue,
            (
                (1, 1),
                (1, 1),
                (0, 0),
            ),
            mode="edge",
        )

    else:

        tissue_padded = np.pad(
            tissue,
            (
                (1, 1),
                (1, 1),
            ),
            mode="edge",
        )

    # --------------------------------------------------------
    # Target center
    # --------------------------------------------------------

    target_cx, target_cy = (
        target_center
    )

    source_cx, source_cy = (
        lumen_center
    )

    n_elements = len(
        inner_nodes
    )

    # ========================================================
    # LOOP THROUGH ANGULAR ELEMENTS
    # ========================================================

    for i in range(n_elements):

        j = (
            i + 1
        ) % n_elements

        # ----------------------------------------------------
        # SOURCE NODES
        # ----------------------------------------------------

        # Single shared lumen center
        source_lumen_node = np.array(
            [
                source_cx,
                source_cy,
            ],
            dtype=np.float32,
        )

        # Tissue quadrilateral
        source_quad = np.array(
            [
                inner_nodes[i],
                inner_nodes[j],
                outer_nodes[j],
                outer_nodes[i],
            ],
            dtype=np.float32,
        )

        # Lumen triangle
        source_triangle = np.array(
            [
                inner_nodes[i],
                inner_nodes[j],
                source_lumen_node,
            ],
            dtype=np.float32,
        )

        # ----------------------------------------------------
        # TARGET ANGLES
        # ----------------------------------------------------

        theta_i = (
            2
            * np.pi
            * i
            / n_elements
        )

        theta_j = (
            2
            * np.pi
            * (i + 1)
            / n_elements
        )

        # ----------------------------------------------------
        # TARGET INNER NODES
        # ----------------------------------------------------

        target_inner_i = np.array(
            [
                target_cx
                + r_inner
                * np.cos(theta_i),

                target_cy
                + r_inner
                * np.sin(theta_i),
            ],
            dtype=np.float32,
        )

        target_inner_j = np.array(
            [
                target_cx
                + r_inner
                * np.cos(theta_j),

                target_cy
                + r_inner
                * np.sin(theta_j),
            ],
            dtype=np.float32,
        )

        # ----------------------------------------------------
        # TARGET OUTER NODES
        # ----------------------------------------------------

        target_outer_i = np.array(
            [
                target_cx
                + r_outer
                * np.cos(theta_i),

                target_cy
                + r_outer
                * np.sin(theta_i),
            ],
            dtype=np.float32,
        )

        target_outer_j = np.array(
            [
                target_cx
                + r_outer
                * np.cos(theta_j),

                target_cy
                + r_outer
                * np.sin(theta_j),
            ],
            dtype=np.float32,
        )

        # ====================================================
        # TISSUE QUADRILATERAL
        # ====================================================

        target_quad = np.array(
            [
                target_inner_i,
                target_inner_j,
                target_outer_j,
                target_outer_i,
            ],
            dtype=np.float32,
        )

        # ----------------------------------------------------
        # Bounding box of target quad
        # ----------------------------------------------------

        min_x = max(
            0,
            int(
                np.floor(
                    np.min(
                        target_quad[:, 0]
                    )
                )
            ) - 1,
        )

        max_x = min(
            width - 1,
            int(
                np.ceil(
                    np.max(
                        target_quad[:, 0]
                    )
                )
            ) + 1,
        )

        min_y = max(
            0,
            int(
                np.floor(
                    np.min(
                        target_quad[:, 1]
                    )
                )
            ) - 1,
        )

        max_y = min(
            height - 1,
            int(
                np.ceil(
                    np.max(
                        target_quad[:, 1]
                    )
                )
            ) + 1,
        )

        if (
            max_x < min_x
            or max_y < min_y
        ):
            continue

        grid_y, grid_x = np.mgrid[
            min_y:max_y + 1,
            min_x:max_x + 1,
        ]

        target_x = grid_x.astype(
            np.float32
        )

        target_y = grid_y.astype(
            np.float32
        )

        # ----------------------------------------------------
        # Determine approximate annulus coordinates
        # ----------------------------------------------------

        local_dx = (
            target_x - target_cx
        )

        local_dy = (
            target_y - target_cy
        )

        local_radius = np.hypot(
            local_dx,
            local_dy,
        )

        local_theta = (
            np.arctan2(
                local_dy,
                local_dx,
            )
            % (2 * np.pi)
        )

        # Angular interval
        theta_start = theta_i
        theta_end = theta_j

        # Handle final wrap-around
        if i == n_elements - 1:

            angular_fraction = (
                (
                    local_theta
                    - theta_start
                )
                % (2 * np.pi)
            ) / (
                2
                * np.pi
                / n_elements
            )

        else:

            angular_fraction = (
                local_theta
                - theta_start
            ) / (
                theta_end
                - theta_start
            )

        # ----------------------------------------------------
        # Element mask
        # ----------------------------------------------------

        element_mask = (
            (local_radius >= r_inner)
            & (local_radius <= r_outer)
            & (angular_fraction >= 0)
            & (angular_fraction <= 1)
        )

        if not np.any(
            element_mask
        ):
            continue

        # ----------------------------------------------------
        # Inverse FEM:
        #
        # target quadrilateral
        #          ↓
        #       xi, eta
        #          ↓
        # source quadrilateral
        # ----------------------------------------------------

        target_points_x = (
            target_x[element_mask]
        )

        target_points_y = (
            target_y[element_mask]
        )

        xi, eta = (
            inverse_bilinear_map(
                target_quad,
                target_points_x,
                target_points_y,
            )
        )

        source_points = (
            bilinear_map(
                source_quad,
                xi,
                eta,
            )
        )

        source_x = (
            source_points[:, 0]
        )

        source_y = (
            source_points[:, 1]
        )

        values = interpolate_image(
            tissue_padded,
            source_x,
            source_y,
            interpolation_order,
        )

        output_y = (
            grid_y[element_mask]
            .astype(int)
        )

        output_x = (
            grid_x[element_mask]
            .astype(int)
        )

        if tissue.ndim == 3:

            for ch in range(
                tissue.shape[2]
            ):

                output[
                    output_y,
                    output_x,
                    ch,
                ] = values[:, ch]

        else:

            output[
                output_y,
                output_x,
            ] = values

        # ====================================================
        # LUMEN TRIANGLE
        # ====================================================

        target_lumen_node = np.array(
            [
                target_cx,
                target_cy,
            ],
            dtype=np.float32,
        )

        target_triangle = np.array(
            [
                target_inner_i,
                target_inner_j,
                target_lumen_node,
            ],
            dtype=np.float32,
        )

        # ----------------------------------------------------
        # Triangle bounding box
        # ----------------------------------------------------

        tri_min_x = max(
            0,
            int(
                np.floor(
                    np.min(
                        target_triangle[:, 0]
                    )
                )
            ) - 1,
        )

        tri_max_x = min(
            width - 1,
            int(
                np.ceil(
                    np.max(
                        target_triangle[:, 0]
                    )
                )
            ) + 1,
        )

        tri_min_y = max(
            0,
            int(
                np.floor(
                    np.min(
                        target_triangle[:, 1]
                    )
                )
            ) - 1,
        )

        tri_max_y = min(
            height - 1,
            int(
                np.ceil(
                    np.max(
                        target_triangle[:, 1]
                    )
                )
            ) + 1,
        )

        if (
            tri_max_x < tri_min_x
            or tri_max_y < tri_min_y
        ):
            continue

        tri_y, tri_x = np.mgrid[
            tri_min_y:tri_max_y + 1,
            tri_min_x:tri_max_x + 1,
        ]

        tri_x = tri_x.astype(
            np.float32
        )

        tri_y = tri_y.astype(
            np.float32
        )

        # ----------------------------------------------------
        # Barycentric coordinates
        # ----------------------------------------------------

        l1, l2, l3 = (
            triangle_barycentric(
                target_triangle[0],
                target_triangle[1],
                target_triangle[2],
                tri_x,
                tri_y,
            )
        )

        triangle_mask = (
            (l1 >= -1e-5)
            & (l2 >= -1e-5)
            & (l3 >= -1e-5)
        )

        if not np.any(
            triangle_mask
        ):
            continue

        # ----------------------------------------------------
        # Map target triangle -> source triangle
        #
        # IMPORTANT:
        #
        # source_lumen_node is the SAME point for every
        # angular element.
        #
        # This is what makes the lumen-center a shared
        # FEM node.
        # ----------------------------------------------------

        target_l1 = l1[
            triangle_mask
        ]

        target_l2 = l2[
            triangle_mask
        ]

        target_l3 = l3[
            triangle_mask
        ]

        source_x = (
            target_l1
            * source_triangle[0, 0]
            + target_l2
            * source_triangle[1, 0]
            + target_l3
            * source_triangle[2, 0]
        )

        source_y = (
            target_l1
            * source_triangle[0, 1]
            + target_l2
            * source_triangle[1, 1]
            + target_l3
            * source_triangle[2, 1]
        )

        values = interpolate_image(
            tissue_padded,
            source_x,
            source_y,
            interpolation_order,
        )

        output_y = (
            tri_y[triangle_mask]
            .astype(int)
        )

        output_x = (
            tri_x[triangle_mask]
            .astype(int)
        )

        if tissue.ndim == 3:

            for ch in range(
                tissue.shape[2]
            ):

                output[
                    output_y,
                    output_x,
                    ch,
                ] = values[:, ch]

        else:

            output[
                output_y,
                output_x,
            ] = values

    return np.clip(
        output,
        0,
        255,
    ).astype(np.uint8)


## SAVE IMAGE ##

def save_image(
    array: np.ndarray,
    output_path: Path,
):
    """
    Save NumPy image array as PNG.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_uint8 = np.clip(
        array,
        0,
        255,
    ).astype(np.uint8)

    Image.fromarray(
        output_uint8
    ).save(output_path)

    return output_path


## SAVE FEM NODES ##

def save_fem_nodes(
    output_path: Path,
    inner_nodes: np.ndarray,
    outer_nodes: np.ndarray,
    lumen_center: tuple[float, float],
    target_center: tuple[float, float],
    r_inner: float,
    r_outer: float,
):
    """
    Save FEM mesh information for later inspection.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez(
        output_path,
        inner_nodes=inner_nodes,
        outer_nodes=outer_nodes,
        lumen_center=np.asarray(
            lumen_center
        ),
        target_center=np.asarray(
            target_center
        ),
        r_inner=r_inner,
        r_outer=r_outer,
    )


## GRAPHS ##

def calculate_and_save_intensity_graphs(
    tissue: np.ndarray,
    target_center: tuple[float, float],
    r_inner: float,
    r_outer: float,
    output_dir: Path,
    slide_id: str = "?",
    n_angle_bins: int = 360,
    radial_bin_size: float = 10.0,
    create_graphs: bool = True,
):
    """Create grayscale intensity-vs-radius and intensity-vs-angle graphs."""

    if not create_graphs:
        return

    csv_path = Path(__file__).parent.parent / "data" / "slides" / "tissue_slides.csv"
    slide_data = pd.read_csv(csv_path)
    slide_row = slide_data[slide_data["slide_id"] == int(slide_id)]

    if slide_row.empty:
        raise ValueError(f"Slide {slide_id} was not found in {csv_path}.")

    pixel_size_um = slide_row.iloc[0]["pixel_size_um"]

    if pd.isna(pixel_size_um):
        raise ValueError(f"No pixel size available for slide {slide_id}.")

    h, w = tissue.shape[:2]
    cy, cx = target_center[1], target_center[0]

    y, x = np.indices((h, w))
    radius = np.hypot(x - cx, y - cy)
    angle = np.degrees(
        np.arctan2(y - cy, x - cx)
    ) % 360

    annulus = (
        (radius >= r_inner)
        & (radius <= r_outer)
    )

    # Convert RGB tissue to grayscale
    if tissue.ndim == 3:
        tissue_gray = np.mean(tissue, axis=2)
    else:
        tissue_gray = tissue

    # --------------------------------------------------------
    # 1. INTENSITY VS RADIUS
    # --------------------------------------------------------

    if radial_bin_size <= 0:
        raise ValueError(
            "radial_bin_size must be greater than 0."
        )

    radii = np.arange(
        r_inner,
        r_outer,
        radial_bin_size,
    )

    radius_intensity = np.full(
        len(radii),
        np.nan,
        dtype=np.float32,
    )

    for i, r in enumerate(radii):

        pixels = (
            annulus
            & (radius >= r)
            & (radius < r + radial_bin_size)
        )

        if np.any(pixels):
            radius_intensity[i] = np.mean(
                tissue_gray[pixels]
            )

    # Plot at the center of each radial bin
    radius_plot = (
        radii
        + radial_bin_size / 2
    )

    # Convert pixel distance to physical distance
    radius_plot_um = (
        radius_plot * pixel_size_um
    )

    plt.figure(figsize=(8, 5))

    plt.plot(
        radius_plot_um,
        radius_intensity,
        color="black",
    )

    plt.xlabel("Radius / distance (µm)")
    plt.ylabel("Average intensity")
    plt.title(
        f"Average grayscale intensity vs radius "
        f"(bin size = {radial_bin_size:g} px)"
    )
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        output_dir
        / f"{slide_id}_intensity_vs_radius.png",
        dpi=300,
    )
    plt.close()

    # --------------------------------------------------------
    # 2. INTENSITY VS ANGLE
    # --------------------------------------------------------

    angle_edges = np.linspace(
        0,
        360,
        n_angle_bins + 1,
    )

    angle_centers = (
        angle_edges[:-1]
        + angle_edges[1:]
    ) / 2

    angle_intensity = np.full(
        n_angle_bins,
        np.nan,
        dtype=np.float32,
    )

    for i in range(n_angle_bins):

        pixels = (
            annulus
            & (angle >= angle_edges[i])
            & (angle < angle_edges[i + 1])
        )

        if np.any(pixels):
            angle_intensity[i] = np.mean(
                tissue_gray[pixels]
            )

    plt.figure(figsize=(8, 5))

    plt.plot(
        angle_centers,
        angle_intensity,
        color="black",
    )

    plt.xlabel("Angle (degrees)")
    plt.ylabel("Average intensity")
    plt.title("Average grayscale intensity vs angle")
    plt.xlim(0, 360)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        output_dir
        / f"{slide_id}_intensity_vs_angle.png",
        dpi=300,
    )
    plt.close()

## MAIN PIPELINE ##

def process_fem_pipeline(
    tissue_path: Path,
    mask_path: Path,
    output_dir: Path,
    slide_id: str = "?",
    n_angles: int = 720,
    max_radius: float = 7000.0,
    interpolation_order: int = 1,
    create_graphs: bool = False,
    radial_bin_size: float = 10.0,
):
    """Complete FEM pipeline: map tissue + lumen to an ideal annulus."""

    # 1. Load images
    print("1. Loading tissue and mask...")
    tissue, mask_binary = load_and_preprocess_images(
        tissue_path,
        mask_path,
    )
    print(f"   Tissue shape: {tissue.shape}")

    # 2. Find initial center for boundary rays
    print("2. Finding mask center...")
    ray_center_x, ray_center_y = find_mask_center(mask_binary)
    print(f"   Ray center: x={ray_center_x:.2f}, y={ray_center_y:.2f}")

    # 3. Find inner and outer boundary nodes
    print("3. Finding boundary nodes...")
    inner_nodes, outer_nodes, valid_rays = find_boundary_nodes(
        mask_binary=mask_binary,
        ray_center_x=ray_center_x,
        ray_center_y=ray_center_y,
        n_angles=n_angles,
        max_radius=max_radius,
    )
    print(f"   Valid rays: {np.sum(valid_rays)} / {len(valid_rays)}")

    # 4. Find the single shared lumen-center node
    print("4. Calculating lumen center...")
    source_lumen_center = calculate_lumen_center(inner_nodes)
    source_cx, source_cy = source_lumen_center
    print(f"   Source center: x={source_cx:.2f}, y={source_cy:.2f}")

    # 5. Calculate average inner and outer radii
    print("5. Calculating average radii...")
    r_inner, r_outer = calculate_average_radii(
        inner_nodes=inner_nodes,
        outer_nodes=outer_nodes,
        center_x=source_cx,
        center_y=source_cy,
    )
    print(f"   Inner radius: {r_inner:.2f} px")
    print(f"   Outer radius: {r_outer:.2f} px")

    # 6. Calculate target canvas and center
    print("6. Calculating target canvas...")
    canvas_size, target_center = calculate_target_canvas(
        r_outer=r_outer,
        margin=10.0,
    )
    print(f"   Canvas: {canvas_size[1]} x {canvas_size[0]}")
    print(
        f"   Target center: "
        f"x={target_center[0]:.2f}, "
        f"y={target_center[1]:.2f}"
    )

    # 7. FEM transformation
    print("7. Performing FEM transformation...")
    print(f"   Tissue elements: {n_angles} quadrilaterals")
    print(f"   Lumen elements: {n_angles} triangles")
    print("   Shared lumen-center nodes: 1")

    reconstructed = fem_reconstruct_annulus(
        tissue=tissue,
        inner_nodes=inner_nodes,
        outer_nodes=outer_nodes,
        lumen_center=source_lumen_center,
        target_center=target_center,
        r_inner=r_inner,
        r_outer=r_outer,
        canvas_size=canvas_size,
        interpolation_order=interpolation_order,
    )

    # 8. Save reconstructed image
    reconstructed_path = (
        output_dir / f"{slide_id}_fem_tissue_and_lumen_annulus.png"
    )
    save_image(
        reconstructed,
        reconstructed_path,
    )
    print(f"   Saved: {reconstructed_path}")

    calculate_and_save_intensity_graphs(
        tissue=reconstructed,
        target_center=target_center,
        r_inner=r_inner,
        r_outer=r_outer,
        output_dir=output_dir,
        slide_id=slide_id,
        n_angle_bins=360,
        radial_bin_size=radial_bin_size,
        create_graphs=create_graphs,
    )

    # 9. Save FEM node information
    nodes_path = output_dir / f"{slide_id}_fem_nodes.npz"
    save_fem_nodes(
        output_path=nodes_path,
        inner_nodes=inner_nodes,
        outer_nodes=outer_nodes,
        lumen_center=source_lumen_center,
        target_center=target_center,
        r_inner=r_inner,
        r_outer=r_outer,
    )
    print(f"   Saved FEM nodes: {nodes_path}")

    print("8. FEM pipeline complete.")


## EXECUTION ##

start_time = time.perf_counter()

if __name__ == "__main__":
    process_fem_pipeline(
        tissue_path=TISSUE_PATH,
        mask_path=MASK_PATH,
        output_dir=OUTPUT_DIR,
        slide_id=SLIDE_ID,
        n_angles=720,
        max_radius=7000.0,
        interpolation_order=1,
        create_graphs=True,
        radial_bin_size=10.0,
    )


total_time = time.perf_counter() - start_time
print(f"TOTAL RUNTIME: {total_time:.2f} seconds")