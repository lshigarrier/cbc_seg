import logging
import torch
import os
import pytorch_lightning as pl
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
from pathlib import Path

from utils import get_one_conf, logging_conf, pytorch_perf, CustomTimer
from data.data import ImageDataModule
from models.models import get_model
from positioning import process_passage
from utils_positioning import process_detections, generate_global_cog
from postprocessing import compute_statistics, generate_histograms


def get_leaf_jpg_directories(current_dir: Path) -> list[Path]:
    valid_leaves = []
    for dirpath, dirnames, filenames in os.walk(current_dir):
        if not dirnames:
            # Check if at least one file has a .jpg or .JPG extension
            if any(f.lower().endswith('.jpg') for f in filenames):
                valid_leaves.append(Path(dirpath))
    return valid_leaves


def load_and_project_areas(areas_csv_path: Path, crs_gps: str, crs_projected: str) -> gpd.GeoDataFrame:
    """
    Reads areas.csv, creates Shapely polygons, and projects them to the local CRS.
    Keeps the original order to resolve overlaps (first match wins).
    """
    df_areas = pd.read_csv(areas_csv_path)

    # Group by zone name and create polygons
    polygons = []
    for zone_name, group in df_areas.groupby('nom', sort=False):
        # Ensure the vertices are in the correct order as defined in the CSV
        coords = list(zip(group['longitude'], group['latitude']))
        polygons.append({'Area': zone_name, 'geometry': Polygon(coords)})

    gdf_areas = gpd.GeoDataFrame(polygons, crs=crs_gps)

    # Project to the local metric CRS
    gdf_areas = gdf_areas.to_crs(crs_projected)
    return gdf_areas


def assign_images_to_areas(df_all_images: pd.DataFrame, gdf_areas: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Spatially joins image coordinates to the zones.
    Images outside any zone are dropped. Overlaps are resolved by taking the first matched zone.
    """
    # Create a GeoDataFrame of points from the corrected coordinates
    geometry = [Point(xy) for xy in zip(df_all_images['X_corr'], df_all_images['Y_corr'])]
    gdf_points = gpd.GeoDataFrame(df_all_images, geometry=geometry, crs=gdf_areas.crs)

    # Perform spatial join (sjoin). 'how="inner"' drops points outside any zone.
    # We drop duplicates on 'Path' to keep only the first matched zone if they overlap.
    gdf_joined = gpd.sjoin(gdf_points, gdf_areas, how="inner", predicate="intersects")
    gdf_joined = gdf_joined.drop_duplicates(subset=['Path'], keep='first')

    return pd.DataFrame(gdf_joined.drop(columns=['geometry', 'index_right']))


def run_positioning(conf, logger):
    logger.info('-' * 70)

    # Find all passage directories
    paths_file = conf.save_dir / "paths.txt"
    passage_dirs = []
    with paths_file.open('r', encoding='utf-8') as file:
        for line in file:
            cleaned_path_str = line.strip().strip('"').strip("'")
            if not cleaned_path_str:
                continue
            current_dir = Path(cleaned_path_str)
            leaf_dirs = get_leaf_jpg_directories(current_dir)
            passage_dirs.extend(leaf_dirs)

    # Correct position of each image from each passage
    all_passage_data = []
    for passage_dir in passage_dirs:
        passage_name = passage_dir.name
        logger.info(f"Processing passage {passage_name}")
        df_coords = process_passage(conf, passage_dir, logger)
        if df_coords is not None and not df_coords.empty:
            # Add absolute image path to facilitate downstream routing
            df_coords['Path'] = df_coords['Image'].apply(lambda x: str(passage_dir / x))
            df_coords['Directory'] = passage_dir
            all_passage_data.append(df_coords)
        else:
            logger.error(f"Failed to process passage {passage_name}")
            continue
    df_all_images = pd.concat(all_passage_data, ignore_index=True)

    # Assign each image to a zone, drop out-of-bounds images
    areas_file = conf.save_dir / "areas.csv"
    gdf_areas = load_and_project_areas(areas_file, conf.crs_gps, conf.crs_projected)
    df_routed = assign_images_to_areas(df_all_images, gdf_areas)
    # Create mapping dictionary: { absolute_image_path: zone_name }, to be used during inference to save the predictions
    routing_map = dict(zip(df_routed['Path'], df_routed['Area']))
    # Identify and log empty areas using sets for efficiency
    all_areas = set(gdf_areas['Area'].unique())
    routed_areas = set(df_routed['Area'].unique())
    empty_areas = all_areas - routed_areas
    logger.info('-' * 70)
    for empty_area in empty_areas:
        logger.warning(f"Area '{empty_area}' is empty (no images assigned)")
    logger.info('-' * 70)

    logger.info(f"Total number of images: {len(df_all_images)}")
    logger.info(f"Number of images assigned to an area: {len(df_routed)}")
    logger.info('-' * 70)

    return routing_map, df_routed


def run_inference(conf, logger, routing_map):
    conf.eval_data_dir = routing_map.keys()
    datamodule = ImageDataModule(conf, logger)
    model = get_model(
        task='inference',
        conf=conf,
        ckpt_path=conf.ckpt_path,
        output_dir=conf.save_dir,
        routing_map=routing_map
    )
    trainer = pl.Trainer(
        logger=False,
        accelerator="gpu" if conf.use_gpu else "cpu",
        devices=1,
        precision="16-mixed",
        deterministic=conf.deterministic
    )

    trainer.predict(model, datamodule=datamodule)


def run_postprocessing(conf, logger, df_routed):
    logger.info('-' * 70)
    if df_routed.empty:
        logger.warning("All areas are empty. No postprocessing.")
        return

    for zone_name, df_zone in df_routed.groupby('Area'):
        logger.info(f"Generating outputs for zone: {zone_name}")

        zone_out_dir = conf.save_dir / zone_name

        # Reconstruct the expected 'all_passage_data' but only with the images belonging to this zone.
        zone_passage_data = []
        for passage_dir, df_passage_part in df_zone.groupby('Directory'):
            zone_passage_data.append((df_passage_part, passage_dir))

        # Generate detections (GeoJSON)
        conf.detection_dir = zone_out_dir / "detections"
        process_detections(conf, zone_passage_data, zone_out_dir, logger)

        # Generate Mosaic (COG)
        generate_global_cog(conf, zone_passage_data, zone_out_dir, logger)

        # Generate Statistics and Histograms
        compute_statistics(conf, zone_out_dir, logger)
        generate_histograms(zone_out_dir, logger)

        logger.info('-' * 70)


def main():
    # Initialization
    logging_conf()
    pytorch_perf()
    logger = logging.getLogger('Pipeline')
    conf = get_one_conf(logger)
    conf.save_dir = Path(conf.save_dir)
    pl.seed_everything(conf.seed, workers=True)

    timer = CustomTimer()
    timer.start()

    # Image positions correction and spatial filtering
    routing_map, df_routed = run_positioning(conf, logger)
    for zone_name in df_routed['Area'].unique():
        zone_out_dir = conf.save_dir / zone_name / "detections"
        zone_out_dir.mkdir(parents=True, exist_ok=True)

    # Inference
    if conf.inference_flag:
        run_inference(conf, logger, routing_map)

    # Build COG and postprocess detections
    if conf.postprocessing_flag:
        run_postprocessing(conf, logger, df_routed)

    timer.stop(logger, len_dataset=len(df_routed))

    peak_memory_gb = torch.cuda.max_memory_allocated() / 1024 ** 3
    logger.info(f'Peak GPU memory allocated: {peak_memory_gb:.2f} GB')


if __name__ == '__main__':
    main()
