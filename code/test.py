### test ###

from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage


from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage


def load_and_preprocess_images(tissue_path: Path, mask_path: Path):
    """Loads tissue (RGB or Grayscale) and mask images, keeping color channels intact."""
    tissue = np.asarray(Image.open(tissue_path))
    mask = np.asarray(Image.open(mask_path))

    # Convert mask to binary boolean array
    mask_gray = mask[..., 0] if mask.ndim == 3 else mask
    mask_binary = mask_gray > 0

    if tissue.shape[:2] != mask_binary.shape:
        raise ValueError("Tissue and mask images must have the same X/Y dimensions.")

    # Convert to float32 for interpolation
    tissue_float = tissue.astype(np.float32)

    return tissue_float, mask_binary


def find_mask_center(mask_binary: np.ndarray) -> tuple[float, float]:
    """Calculates the centroid (center of mass) of the binary mask."""
    y_mask, x_mask = np.nonzero(mask_binary)

    if len(x_mask) == 0:
        raise ValueError("Mask contains no foreground pixels.")

    center_x = float(x_mask.mean())
    center_y = float(y_mask.mean())

    return center_x, center_y


def unroll_annulus_rgb(
    tissue: np.ndarray,
    mask_binary: np.ndarray,
    center_x: float,
    center_y: float,
    n_angles: int = 360,
    n_radial: int = 2048,
    max_radius: float = 6000.0,
    interpolation_order: int = 1,
) -> np.ndarray:
    """Unrolls a circular tissue/vessel annulus into a rectangular RGB image.

    Output layout:
    - Axis 0 (Rows / Height): Angle theta = 0 to 2*pi
    - Axis 1 (Cols / Width) : Radial distance d = 0 (lumen/inner) to d = 1 (outer)
    """
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    search_radii = np.arange(0, max_radius, dtype=np.float32)
    normalized_radius = np.linspace(0, 1, n_radial)

    is_rgb = tissue.ndim == 3
    num_channels = tissue.shape[2] if is_rgb else 1

    # Output shape: (N_ANGLES, N_RADIAL, CHANNELS) or (N_ANGLES, N_RADIAL)
    if is_rgb:
        output = np.zeros((n_angles, n_radial, num_channels), dtype=np.float32)
    else:
        output = np.zeros((n_angles, n_radial), dtype=np.float32)

    valid_rays = np.zeros(n_angles, dtype=bool)

    # Pad tissue by 1 pixel on borders to prevent out-of-bounds boundary artifacts
    if is_rgb:
        tissue_padded = np.pad(tissue, ((1, 1), (1, 1), (0, 0)), mode="edge")
    else:
        tissue_padded = np.pad(tissue, ((1, 1), (1, 1)), mode="edge")

    for angle_index, theta in enumerate(angles):
        # 1. Cast search ray outward from center
        x = center_x + search_radii * np.cos(theta)
        y = center_y + search_radii * np.sin(theta)

        x_int = np.round(x).astype(int)
        y_int = np.round(y).astype(int)

        # 2. Keep coordinates strictly inside image bounds
        valid = (
            (x_int >= 0) & 
            (x_int < mask_binary.shape[1]) & 
            (y_int >= 0) & 
            (y_int < mask_binary.shape[0])
        )

        radial_mask = np.zeros_like(search_radii, dtype=bool)
        radial_mask[valid] = mask_binary[y_int[valid], x_int[valid]]

        # 3. Intersect with mask to find inner (d=0) and outer (d=1) boundaries
        foreground_indices = np.flatnonzero(radial_mask)
        if len(foreground_indices) == 0:
            continue

        r_inner = search_radii[foreground_indices[0]]   # d = 0 (lumen)
        r_outer = search_radii[foreground_indices[-1]]  # d = 1 (outer edge)

        if r_outer <= r_inner:
            continue

        valid_rays[angle_index] = True

        # 4. Map d=0..1 normalized depth back to real image coordinates (+1 shift for padding)
        sample_radii = r_inner + normalized_radius * (r_outer - r_inner)
        sample_x = center_x + sample_radii * np.cos(theta) + 1.0
        sample_y = center_y + sample_radii * np.sin(theta) + 1.0

        # 5. Interpolate RGB channels along Axis 1 (columns) for this angle row
        if is_rgb:
            for ch in range(num_channels):
                output[angle_index, :, ch] = ndimage.map_coordinates(
                    tissue_padded[..., ch],
                    [sample_y, sample_x],
                    order=interpolation_order,
                    mode="nearest",
                )
        else:
            output[angle_index, :] = ndimage.map_coordinates(
                tissue_padded,
                [sample_y, sample_x],
                order=interpolation_order,
                mode="nearest",
            )

    # Fill any empty rows (angles where mask wasn't found) with nearest valid ray
    if not np.all(valid_rays):
        if not np.any(valid_rays):
            raise ValueError("No valid vessel boundaries were found along any angle ray.")

        valid_indices = np.flatnonzero(valid_rays)

        # Circular nearest neighbor index lookup for angular continuity
        for idx in np.flatnonzero(~valid_rays):
            nearest_valid_idx = valid_indices[
                np.argmin(np.abs((valid_indices - idx + n_angles // 2) % n_angles - n_angles // 2))
            ]
            output[idx] = output[nearest_valid_idx]

    return output


def fold_annulus_rgb(
    unrolled_image: np.ndarray,
    r_inner: float,
    r_outer: float,
    canvas_size: tuple[int, int] | None = None,
    interpolation_order: int = 1,
) -> np.ndarray:
    """Folds a rectangular unrolled image back into an annulus (ring).

    Parameters
    ----------
    unrolled_image : np.ndarray
        Array of shape (N_ANGLES, N_RADIAL, CHANNELS) or (N_ANGLES, N_RADIAL).
        - Axis 0 (Rows): Angles theta from 0 to 2*pi
        - Axis 1 (Cols): Radial distance d from 0 (lumen) to 1 (outer edge)
    r_inner : float
        Inner radius (lumen boundary) in target pixels.
    r_outer : float
        Outer radius boundary in target pixels.
    canvas_size : tuple[int, int], optional
        Output image canvas (height, width). If None, automatically derived.
    interpolation_order : int, default=1
        Bilinear (1) or Nearest-Neighbor (0) interpolation.

    Returns
    -------
    np.ndarray
        The reconstructed 2D annular tissue image with a black lumen.
    """
    n_angles, n_radial = unrolled_image.shape[:2]
    is_rgb = unrolled_image.ndim == 3
    num_channels = unrolled_image.shape[2] if is_rgb else 1

    # 1. Determine target canvas dimensions & center
    if canvas_size is None:
        dim = int(np.ceil(2 * (r_outer + 5)))
        canvas_size = (dim, dim)

    height, width = canvas_size
    center_x = (width - 1) / 2.0
    center_y = (height - 1) / 2.0

    # 2. Build target grid coordinates
    grid_y, grid_x = np.indices((height, width), dtype=np.float32)

    dx = grid_x - center_x
    dy = grid_y - center_y

    # Distance r and angle theta [0, 2*pi)
    r = np.hypot(dx, dy)
    theta = np.arctan2(dy, dx) % (2 * np.pi)

    # 3. Map theta to unrolled row index
    angle_idx = (theta / (2 * np.pi)) * n_angles

    # 4. Map radius r to normalized depth d in [0, 1]
    d = (r - r_inner) / (r_outer - r_inner)
    radial_idx = d * (n_radial - 1)

    # 5. Mask out pixels outside the vessel ring (lumen or background)
    valid_annulus = (r >= r_inner) & (r <= r_outer)

    # 6. Interpolate channels
    if is_rgb:
        reconstructed = np.zeros((height, width, num_channels), dtype=np.float32)
        for ch in range(num_channels):
            sampled_ch = ndimage.map_coordinates(
                unrolled_image[..., ch],
                [angle_idx, radial_idx],
                order=interpolation_order,
                mode="wrap",  # Wrap theta boundary around 2*pi -> 0
            )
            reconstructed[..., ch] = np.where(valid_annulus, sampled_ch, 0)
    else:
        sampled = ndimage.map_coordinates(
            unrolled_image,
            [angle_idx, radial_idx],
            order=interpolation_order,
            mode="wrap",
        )
        reconstructed = np.where(valid_annulus, sampled, 0)

    return np.clip(reconstructed, 0, 255).astype(np.uint8)


def save_image(array: np.ndarray, output_path: Path) -> Path:
    """Saves a NumPy array (RGB or Grayscale) to disk as a PNG."""
    output_uint8 = np.clip(array, 0, 255).astype(np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(output_uint8).save(output_path)
    return output_path


def process_vessel_pipeline(
    tissue_path: Path,
    mask_path: Path,
    output_dir: Path,
    n_angles: int = 360,
    n_radial: int = 2048,
    max_radius: float = 6000.0,
    interpolation_order: int = 1,
    fold_r_inner: float = 100.0,
    fold_r_outer: float = 350.0,
):
    """Main execution pipeline: Unrolls tissue and then folds it back."""
    print("1. Loading tissue and mask...")
    tissue, mask_binary = load_and_preprocess_images(tissue_path, mask_path)

    print("2. Locating mask center...")
    center_x, center_y = find_mask_center(mask_binary)
    print(f"   Center: x={center_x:.2f}, y={center_y:.2f}")

    print("3. Unrolling tissue to polar rectangle...")
    unrolled = unroll_annulus_rgb(
        tissue=tissue,
        mask_binary=mask_binary,
        center_x=center_x,
        center_y=center_y,
        n_angles=n_angles,
        n_radial=n_radial,
        max_radius=max_radius,
        interpolation_order=interpolation_order,
    )
    unrolled_path = output_dir / "unrolled_vessel.png"
    save_image(unrolled, unrolled_path)
    print(f"   Saved unrolled image: {unrolled_path} (Shape: {unrolled.shape})")

    print("4. Folding rectangle back into annulus ring...")
    folded = fold_annulus_rgb(
        unrolled_image=unrolled,
        r_inner=fold_r_inner,
        r_outer=fold_r_outer,
        canvas_size=(12000, 12000),
        interpolation_order=interpolation_order,
    )
    folded_path = output_dir / "reconstructed_annulus.png"
    save_image(folded, folded_path)
    print(f"   Saved reconstructed annulus: {folded_path} (Shape: {folded.shape})")


if __name__ == "__main__":
    TISSUE_PATH = Path(r"C:\Users\20234257\OneDrive - TU Eindhoven\BEP Project\external data\slides\image_2152_tissue.png")
    MASK_PATH = Path(r"C:\Users\20234257\OneDrive - TU Eindhoven\BEP Project\external data\slides\image_2152_mask.png")
    OUTPUT_DIR = Path(r"C:\Users\20234257\OneDrive - TU Eindhoven\BEP Project\external data\slides")

    process_vessel_pipeline(
        tissue_path=TISSUE_PATH,
        mask_path=MASK_PATH,
        output_dir=OUTPUT_DIR,
        n_angles=720,
        n_radial=2048,
        max_radius=7000,
        interpolation_order=1,
        fold_r_inner=2000.0,  # Inner lumen radius (black)
        fold_r_outer=5500.0,  # Outer wall radius
    )

