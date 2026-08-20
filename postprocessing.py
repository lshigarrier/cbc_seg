import json
import logging
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path
from rasterio.features import rasterize
from rasterio.transform import from_origin
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
from shapely.geometry import box

from utils import logging_conf, get_conf, CustomTimer


def chunk_geometry_recursive(geom, chunk_size_m, overlap_m):
    """
    Recursively splits a large geometry into smaller chunks to avoid CPU bottlenecks.
    Uses a binary split approach (KD-Tree style): it only splits the longest dimension
    to avoid creating extremely narrow geometries.

    Args:
        geom: The geometry to chunk.
        chunk_size_m (float): Maximum allowed size in meters for a chunk's dimension.
        overlap_m (float): Physical overlap in meters added to each chunk to preserve EDT context.

    Returns:
        list: A list of tuples (chunk_geometry, strict_bounding_box).
    """
    if geom.is_empty:
        return []

    minx, miny, maxx, maxy = geom.bounds
    width = maxx - minx
    height = maxy - miny

    # Base case: bounding box is smaller than or equal to the chunk size
    if width <= chunk_size_m and height <= chunk_size_m:
        strict_box = box(*geom.bounds)
        if geom.geom_type in ['Polygon', 'MultiPolygon']:
            return [(g, strict_box) for g in (geom.geoms if geom.geom_type == 'MultiPolygon' else [geom])]
        elif geom.geom_type == 'GeometryCollection':
            parts = []
            for part in geom.geoms:
                if part.geom_type in ['Polygon', 'MultiPolygon']:
                    parts.extend(
                        [(g, strict_box) for g in (part.geoms if part.geom_type == 'MultiPolygon' else [part])])
            return parts
        return []

    chunks = []

    # Decide which axis to split: always split the longest dimension
    if width >= height:
        midx = minx + width / 2.0

        strict_halves = [
            box(minx, miny, midx, maxy),
            box(midx, miny, maxx, maxy)
        ]
        extended_halves = [
            box(minx, miny, midx + overlap_m, maxy),
            box(midx - overlap_m, miny, maxx, maxy)
        ]
    else:
        midy = miny + height / 2.0

        strict_halves = [
            box(minx, miny, maxx, midy),
            box(minx, midy, maxx, maxy)
        ]
        extended_halves = [
            box(minx, miny, maxx, midy + overlap_m),
            box(minx, midy - overlap_m, maxx, maxy)
        ]

    for strict_half, ext_half in zip(strict_halves, extended_halves):
        # Check intersection with the extended bounding box to preserve context
        if geom.intersects(ext_half):
            intersection = geom.intersection(ext_half)
            if not intersection.is_empty:
                # Recursively chunk the intersection
                sub_chunks = chunk_geometry_recursive(intersection, chunk_size_m, overlap_m)

                # Propagate the strict mask rules back up the recursion tree
                for sub_geom, sub_strict in sub_chunks:
                    actual_strict = sub_strict.intersection(strict_half)
                    if not actual_strict.is_empty:
                        chunks.append((sub_geom, actual_strict))

    return chunks


def process_geometry_chunk(geom, strict_box, res_m, compute_widths, bin_width_mm, num_bins):
    """
    Rasterize a single manageable geometry chunk and extract its skeleton to compute length.
    If compute_widths is True, also computes the width at each skeleton pixel using a Euclidean Distance Transform (EDT).
    """
    minx, miny, maxx, maxy = geom.bounds

    # Add a small padding to ensure the polygon boundary is fully enclosed
    minx -= res_m * 2.0
    miny -= res_m * 2.0
    maxx += res_m * 2.0
    maxy += res_m * 2.0

    # Calculate raster dimensions based on bounding box and resolution
    width_px = int(np.ceil((maxx - minx) / res_m))
    height_px = int(np.ceil((maxy - miny) / res_m))

    # Handle edge case where geometry is smaller than a pixel
    if width_px == 0 or height_px == 0:
        return 0.0, (np.zeros(num_bins) if compute_widths else None)

    transform = from_origin(minx, maxy, res_m, res_m)

    # Rasterize the polygon geometry
    mask = rasterize(
        [(geom, 1)],
        out_shape=(height_px, width_px),
        transform=transform,
        fill=0,
        dtype=np.uint8
    )

    if not np.any(mask):
        return 0.0, (np.zeros(num_bins) if compute_widths else None)

    # Skeletonize to find the centerline of the defect
    skeleton = skeletonize(mask)

    # Create and apply the strict spatial mask
    s_minx, s_miny, s_maxx, s_maxy = strict_box.bounds
    # Convert strict box geographic coordinates to pixel indices
    col_start = max(0, int((s_minx - minx) / res_m))
    col_end = min(width_px, int(np.ceil((s_maxx - minx) / res_m)))
    row_start = max(0, int((maxy - s_maxy) / res_m))
    row_end = min(height_px, int(np.ceil((maxy - s_miny) / res_m)))

    strict_mask = np.zeros_like(skeleton, dtype=bool)
    strict_mask[row_start:row_end, col_start:col_end] = True

    # Keep only skeleton pixels that are strictly inside the exact quadrant
    skeleton = skeleton & strict_mask

    if not np.any(skeleton):
        return 0.0, (np.zeros(num_bins) if compute_widths else None)

    # Calculate length: each skeleton pixel represents approximately 'res_m' meters in length
    # (A more precise graph-based length could be used, but pixel counting is faster and standard)
    length_m = np.sum(skeleton) * res_m

    if not compute_widths:
        return length_m, None

    # Calculate distance transform (distances are in pixels)
    # The true width is 2 * distance to the closest edge
    dist_map = distance_transform_edt(mask)

    # Extract distances at skeleton pixels
    # True width (mm) = 2 * dist_px * res_m
    widths_mm = 2.0 * dist_map[skeleton] * (res_m * 1000.0)

    # Build the histogram
    bin_indices = (widths_mm // bin_width_mm).astype(int)
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)
    hist = np.zeros(num_bins)
    np.add.at(hist, bin_indices, res_m)

    return length_m, hist


def process_linear_class(geom_multipolygon, res_m, chunk_size_m, overlap_m, compute_widths, bin_width_mm=None, num_bins=None):
    """
    Break down a complex linear geometry into smaller spatial chunks and accumulate linear statistics.
    """
    total_length = 0.0
    total_hist = np.zeros(num_bins) if compute_widths else None

    chunks = chunk_geometry_recursive(geom_multipolygon, chunk_size_m=chunk_size_m, overlap_m=overlap_m)

    for chunk_geom, strict_box in chunks:
        length_m, hist = process_geometry_chunk(chunk_geom, strict_box, res_m, compute_widths, bin_width_mm, num_bins)
        total_length += length_m
        if compute_widths:
            total_hist += hist

    return total_length, total_hist


def compute_statistics(conf, out_dir, logger):
    """
    Reads merged detections, and computes statistics based on class_type definition.
    0: Area for surface defects.
    1: Length only for large linear defects (low res skeletonization).
    2: Length and width histograms for linear defects (high res skeletonization + EDT).
    """
    input_file = out_dir / "detections.geojson"
    output_file = out_dir / "statistics.json"

    logger.info(f"  Loading geometries from {input_file}")
    if not input_file.exists():
        logger.error(f"  Input file {input_file} does not exist.")
        return

    gdf = gpd.read_file(input_file)

    # Reprojecting geometries to conf.crs_projected
    gdf = gdf.to_crs(conf.crs_projected)

    res_high_m = conf.raster_res / 1000.0
    res_low_m = conf.raster_low_res / 1000.0

    results = {}

    for index, row in gdf.iterrows():
        cls = row["class"]

        if cls not in conf.class_type:
            continue

        # logger.info(f"    Processing class {cls} (Row index: {index})")

        ctype = conf.class_type[cls]
        geom = row["geometry"]

        # Initialize the data structure for a new class
        if cls not in results:
            if ctype == 0:
                results[cls] = {"area_m2": 0.0}
            elif ctype == 1:
                results[cls] = {"length_m": 0.0}
            elif ctype == 2:
                results[cls] = {
                    "length_m": 0.0,
                    "bin_width_mm": conf.bin_width,
                    "histogram": {i: 0.0 for i in range(conf.num_bins)}
                }

        # Compute statistics based on class type
        if ctype == 0:
            results[cls]["area_m2"] += geom.area

        elif ctype == 1:
            length_m, _ = process_linear_class(
                geom, res_low_m, conf.chunk_size_m, conf.overlap_m, compute_widths=False
            )
            results[cls]["length_m"] += length_m

        elif ctype == 2:
            length_m, hist = process_linear_class(
                geom, res_high_m, conf.chunk_size_m, conf.overlap_m, compute_widths=True,
                bin_width_mm=conf.bin_width, num_bins=conf.num_bins
            )
            results[cls]["length_m"] += length_m
            for i in range(conf.num_bins):
                results[cls]["histogram"][i] += float(hist[i])

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    logger.info("  Statistics computation completed successfully.")


def generate_histograms(out_dir, logger):
    """
    Reads the statistics.json file and generates a distribution plot
    for each linear defect detected.
    """
    stats_path = out_dir / "statistics.json"
    if not stats_path.exists():
        logger.error("  Cannot generate histograms: statistics.json not found.")
        return

    with open(stats_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    plots_dir = out_dir / "histograms"
    plots_dir.mkdir(exist_ok=True)

    for class_name, metrics in data.items():
        if "histogram" not in metrics:
            continue

        # JSON keys are strings, convert back to int for sorting/plotting
        hist_data = {int(k): v for k, v in metrics["histogram"].items()}
        bin_width = metrics["bin_width_mm"]
        total_length = round(metrics["length_m"])

        bins = sorted(hist_data.keys())
        x_values = [b * bin_width for b in bins]
        y_values = [hist_data[b] for b in bins]

        plt.figure(figsize=(10, 6))
        plt.bar(x_values, y_values, width=bin_width, align='edge', color='skyblue', edgecolor='navy')

        plt.title(f"Distribution des ouvertures pour la classe '{class_name}'\nLongueur totale : {total_length} m")
        plt.xlabel("Ouverture (mm)")
        plt.ylabel("Longueur cumulée (m)")
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Sanitize filename
        safe_name = class_name.replace(" ", "_").replace("/", "_")
        plt.savefig(plots_dir / f"hist_{safe_name}.png")
        plt.close()


def main():
    logging_conf()
    logger = logging.getLogger('Statistics')
    conf = get_conf(logger, verbose=False)

    out_dir = Path(conf.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    timer = CustomTimer()
    timer.start()

    compute_statistics(conf, out_dir, logger)
    generate_histograms(out_dir, logger)

    timer.stop(logger, show_time_per_image=False)


if __name__ == '__main__':
    main()
