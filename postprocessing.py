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


def chunk_geometry_recursive(geom, chunk_size_m=5.0):
    """
    Recursively split a large geometry into smaller chunks to avoid CPU bottlenecks.
    Uses a QuadTree approach based on bounding boxes.
    """
    if geom.is_empty:
        return []

    minx, miny, maxx, maxy = geom.bounds
    width = maxx - minx
    height = maxy - miny

    # Base case: bounding box is smaller than or equal to the chunk size
    if width <= chunk_size_m and height <= chunk_size_m:
        if geom.geom_type in ['Polygon', 'MultiPolygon']:
            return geom.geoms if geom.geom_type == 'MultiPolygon' else [geom]
        elif geom.geom_type == 'GeometryCollection':
            parts = []
            for part in geom.geoms:
                if part.geom_type in ['Polygon', 'MultiPolygon']:
                    parts.extend(part.geoms if part.geom_type == 'MultiPolygon' else [part])
            return parts
        return []

    chunks = []
    midx = minx + width / 2.0
    midy = miny + height / 2.0

    # Define the 4 quadrants
    quadrants = [
        box(minx, miny, midx, midy),
        box(midx, miny, maxx, midy),
        box(minx, midy, midx, maxy),
        box(midx, midy, maxx, maxy)
    ]

    for quad in quadrants:
        # Fast bounding box check before exact intersection
        if geom.intersects(quad):
            intersection = geom.intersection(quad)
            if not intersection.is_empty:
                # Recursively chunk the intersection
                chunks.extend(chunk_geometry_recursive(intersection, chunk_size_m))

    return chunks


def process_geometry_chunk(geom, res_m, bin_width_mm, num_bins):
    """
    Rasterize a single manageable geometry chunk, extract its skeleton, and compute the width
    at each skeleton pixel using a Euclidean Distance Transform (EDT).
    """
    minx, miny, maxx, maxy = geom.bounds

    # Calculate raster dimensions based on bounding box and resolution
    width_px = int(np.ceil((maxx - minx) / res_m))
    height_px = int(np.ceil((maxy - miny) / res_m))

    # Handle edge case where geometry is smaller than a pixel
    if width_px == 0 or height_px == 0:
        return 0.0, np.zeros(num_bins)

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
        return 0.0, np.zeros(num_bins)

    # Calculate distance transform (distances are in pixels)
    # The true width is 2 * distance to the closest edge
    dist_map = distance_transform_edt(mask)

    # Skeletonize to find the centerline of the defect
    skeleton = skeletonize(mask)

    if not np.any(skeleton):
        return 0.0, np.zeros(num_bins)

    # Extract distances at skeleton pixels and convert to mm
    # True width (mm) = 2 * dist_px * res_m * 1000
    width_mm = 2.0 * dist_map[skeleton] * (res_m * 1000.0)

    # Calculate length: each skeleton pixel represents approximately 'res_m' meters in length
    # (A more precise graph-based length could be used, but pixel counting is faster and standard)
    length_m = np.sum(skeleton) * res_m

    # Build the histogram
    bins = np.arange(num_bins + 1) * bin_width_mm
    hist, _ = np.histogram(width_mm, bins=bins)

    # Add values exceeding the max bin to the last bin
    hist[-1] += np.sum(width_mm >= bins[-1])

    return length_m, hist


def process_linear_class(geom_multipolygon, res_m, bin_width_mm, num_bins, chunk_size_m):
    """
    Break down a complex linear geometry into smaller spatial chunks and accumulate linear statistics.
    """
    total_length = 0.0
    total_hist = np.zeros(num_bins)

    chunks = chunk_geometry_recursive(geom_multipolygon, chunk_size_m=chunk_size_m)

    for chunk in chunks:
        length_m, hist = process_geometry_chunk(chunk, res_m, bin_width_mm, num_bins)
        total_length += length_m
        total_hist += hist

    return total_length, total_hist


def compute_statistics(conf, out_dir, logger):
    """
    Reads merged detections, computes area for surface defects,
    and skeletonizes linear defects to output a width histogram.
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

    # Resolution in meters corresponds to the bin width
    res_m = conf.raster_res / 1000.0

    results = {}

    for index, row in gdf.iterrows():
        cls = row["class"]

        if cls not in conf.class_type:
            continue

        logger.info(f"    Processing class {cls} (Row index: {index})")

        is_linear = conf.class_type[cls]
        geom = row["geometry"]

        # Initialize the data structure for a new class
        if cls not in results:
            if is_linear:
                results[cls] = {
                    "length_m": 0.0,
                    "bin_width_mm": conf.bin_width,
                    "histogram": {i: 0.0 for i in range(conf.num_bins)}
                }
            else:
                results[cls] = {"area_m2": 0.0}

        # Compute statistics based on class type
        if is_linear:
            length_m, hist = process_linear_class(geom, res_m, conf.bin_width, conf.num_bins, conf.chunk_size_m)
            results[cls]["length_m"] += length_m
            for i in range(conf.num_bins):
                results[cls]["histogram"][i] += hist[i]
        else:
            results[cls]["area_m2"] += geom.area

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
