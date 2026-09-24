## Distance transformation ##

# functionality
# part 1 : Take binary image mask (png file with name: image_{slide_id}_mask) with Background (black), tissue (white), Lumen (black). 
# Assume lumen is surrounded by tissue, and tissue is surrounded by background. Use distance_transform_edt (scipy) to create distance map,
# with first: lumen = black, tissue and background = value dependent on pixel distance to lumen, 
# then make background black so that lumen = black, background = black, tissue = value based on distance to lumen 
# (use binary_fill_holes to determine what is lumen and what is background)
# part 2: Use image tissue belonging to the mask (png file with name: image_{slide_id}_tissue) to make a graph with 
# on y-as: distance to lumen and on x-as mean pixel intensity

# setup?

## libraries ##
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import binary_fill_holes, distance_transform_edt
from pathlib import Path
import time

Image.MAX_IMAGE_PIXELS = None

## Settings ##
slides_dir = Path(r"C:\Users\20234257\OneDrive - TU Eindhoven\BEP Project\external data\slides")
slide_id = 9943


## Functions ##
def load_mask(path):
    return np.array(Image.open(path).convert("L")) > 0


def identify_lumen_background(tissue):
    filled = binary_fill_holes(tissue)
    lumen = filled & ~tissue
    background = ~filled
    return tissue, lumen, background


def create_distance_map(tissue, lumen, background):
    distance = distance_transform_edt(~lumen)
    distance[~tissue | background] = 0
    return distance


def save_distance_map(distance, path):
    # Normalize distance values within the tissue
    max_distance = distance.max()

    normalized = np.zeros_like(distance, dtype=float)

    tissue_mask = distance > 0
    normalized[tissue_mask] = distance[tissue_mask] / max_distance

    # Apply rainbow/turbo colormap
    cmap = plt.get_cmap("turbo")
    colored = cmap(normalized)

    # Make everything outside the tissue black
    colored[~tissue_mask] = [0, 0, 0, 1]

    # Convert RGBA (0–1) to RGB (0–255)
    image = (colored[:, :, :3] * 255).astype(np.uint8)

    output_path = path.parent / "distance_transformation"
    output_path.mkdir(parents=True, exist_ok=True)

    Image.fromarray(image).save(output_path / path.name)


def load_tissue_image(path):
    return np.array(Image.open(path).convert("L"))


def calculate_mean_intensity_by_distance(
    distance,
    tissue_image,
    bin_size=10
):

    
    # Only use pixels inside the tissue
    mask = (distance > 0) & (tissue_image > 0)

    distances = distance[mask]
    intensities = tissue_image[mask]

    # Assign each pixel to a distance bin
    bins = np.floor(distances / bin_size).astype(int)

    # Calculate mean intensity for each bin
    sum_intensity = np.bincount(bins, weights=intensities)

    pixel_count = np.bincount(bins)

    mean_intensity = sum_intensity / pixel_count

    # Distance represented by the centre of each bin
    mean_distance = (np.arange(len(mean_intensity)) * bin_size + bin_size / 2)

    # Remove empty bins
    valid = pixel_count > 0

    return mean_distance[valid], mean_intensity[valid]



def plot_intensity_vs_distance(distances, mean_intensity, path, slide_id):
    plt.figure(figsize=(6, 5))
    plt.plot(distances, mean_intensity)
    plt.xlabel("Distance to lumen (pixels)")
    plt.ylabel("Mean pixel intensity")
    plt.tight_layout()

    output_path = path.parent / "distance_transformation"
    output_path.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path / path.name, dpi=300)
    plt.close()



def main(slide_id):
    total_start = time.perf_counter()

    # Load mask #
    start = time.perf_counter()
    mask_path = slides_dir / f"image_{slide_id}_mask.png"
    tissue_path = slides_dir / f"image_{slide_id}_tissue.png"
    tissue = load_mask(mask_path)
    print(f"Loading mask: {time.perf_counter() - start:.2f} seconds")

    # Identify lumen and background #
    start = time.perf_counter()
    tissue, lumen, background = identify_lumen_background(tissue)
    print(
            f"Identifying lumen/background: "
            f"{time.perf_counter() - start:.2f} seconds"
    )

    # Distance transformation #
    start = time.perf_counter()
    distance = create_distance_map(tissue, lumen, background)
    print(
        f"Distance transformation: "
        f"{time.perf_counter() - start:.2f} seconds"
    )

    # Save distance map #
    start = time.perf_counter()
    save_distance_map(
        distance,
        slides_dir / f"image_{slide_id}_distance_map.png"
    )
    print(
        f"Saving distance map: "
        f"{time.perf_counter() - start:.2f} seconds"
    )

    # Load tissue image #
    start = time.perf_counter()
    tissue_image = load_tissue_image(tissue_path)
    print(
        f"Loading tissue image: "
        f"{time.perf_counter() - start:.2f} seconds"
    )

    # Calculate mean intensity #
    start = time.perf_counter()
    distances, mean_intensity = calculate_mean_intensity_by_distance(
        distance,
        tissue_image,
        bin_size=20
    )
    print(
        f"Calculating mean intensity: "
        f"{time.perf_counter() - start:.2f} seconds"
    )

    # create graph #
    start = time.perf_counter()
    plot_intensity_vs_distance(
        distances,
        mean_intensity,
        slides_dir / f"image_{slide_id}_distance_intensity.png",
        slide_id=slide_id
    )
    print(
        f"Creating graph: "
        f"{time.perf_counter() - start:.2f} seconds"
    )

    # Total runtime #
    total_time = time.perf_counter() - total_start

    print("\n" + "=" * 50)
    print(f"TOTAL RUNTIME: {total_time:.2f} seconds")
    print(f"TOTAL RUNTIME: {total_time / 60:.2f} minutes")
    print("=" * 50)


if __name__ == "__main__":
    main(slide_id)