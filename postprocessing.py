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

from utils import logging_conf, get_conf, CustomTimer


def process_linear_geometry(geom, res_m, bin_width_mm, num_bins):
    """
    Rasterize a linear geometry, extract its skeleton, and compute the width
    at each skeleton pixel using a Euclidean Distance Transform.
    """
    minx, miny, maxx, maxy = geom.bounds

    # Add a small padding to ensure the polygon boundary is fully enclosed
    minx -= res_m * 2
    miny -= res_m * 2
    maxx += res_m * 2
    maxy += res_m * 2

    width_px = int(np.ceil((maxx - minx) / res_m))
    height_px = int(np.ceil((maxy - miny) / res_m))

    if width_px <= 0 or height_px <= 0:
        return 0.0, np.zeros(num_bins)

    transform = from_origin(minx, maxy, res_m, res_m)

    mask = rasterize(
        [(geom, 1)],
        out_shape=(height_px, width_px),
        transform=transform,
        fill=0,
        dtype=np.uint8
    )

    if not np.any(mask):
        return 0.0, np.zeros(num_bins)

    # Distance transform: distance to the nearest zero (background) pixel
    edt = distance_transform_edt(mask)
    skeleton = skeletonize(mask)

    # The EDT returns distance in pixels.
    # Width (meters) = 2 * distance_px * res_m
    # Width (mm) = Width (meters) * 1000
    widths_mm = edt[skeleton] * res_m * 2.0 * 1000.0

    # Bin indices
    bin_indices = (widths_mm // bin_width_mm).astype(int)
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)

    # For simplicity and efficiency, we approximate the length contributed
    # by each skeleton pixel as the pixel resolution (res_m).
    pixel_length_m = res_m

    hist = np.zeros(num_bins)
    np.add.at(hist, bin_indices, pixel_length_m)

    total_length_m = len(widths_mm) * pixel_length_m

    return total_length_m, hist


def compute_statistics(conf, out_dir, logger):
    """
    Reads merged detections, computes area for surface defects,
    and skeletonizes linear defects to output a width histogram.
    """
    input_file = out_dir / "detections.geojson"
    output_file = out_dir / "statistics.json"

    logger.info(f"Loading geometries from {input_file}")
    if not input_file.exists():
        logger.error(f"Input file {input_file} does not exist.")
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
            length_m, hist = process_linear_geometry(geom, res_m, conf.bin_width, conf.num_bins)
            results[cls]["length_m"] += length_m
            for i in range(conf.num_bins):
                results[cls]["histogram"][i] += hist[i]
        else:
            results[cls]["area_m2"] += geom.area

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    logger.info("Statistics computation completed successfully.")


def generate_histograms(out_dir, logger):
    """
    Reads the statistics.json file and generates a distribution plot
    for each linear defect detected.
    """
    stats_path = out_dir / "statistics.json"
    if not stats_path.exists():
        logger.error("Cannot generate histograms: statistics.json not found.")
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
