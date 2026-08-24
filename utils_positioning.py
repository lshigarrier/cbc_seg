import numpy as np
import pyproj
import cv2
import math
import rasterio
import json
from pathlib import Path
from rasterio.transform import Affine
from rasterio.windows import Window
from collections import defaultdict
from shapely.geometry import Polygon, mapping
from shapely.ops import unary_union
from shapely.ops import transform as shapely_transform
from typing import Dict, List, Any


def export_qgis_style(active_class2show, qml_path, logger):
    """
        Generates a QGIS style file (.qml) for the detected classes.
        Only includes classes that are both requested for visualization and actually present in the data.

        Args:
            active_class2show (dict): Dictionary mapping present class names to their RGB color tuples.
            qml_path (pathlib.Path): Path where the .qml file will be saved.
            logger (logging.Logger): Logger instance for outputting information.
        """
    categories_xml = []
    symbols_xml = []

    for idx, (cls_name, color) in enumerate(active_class2show.items()):
        color_str = f"{color[0]},{color[1]},{color[2]},255"

        # language=text
        categories_xml.append(
            f'<category symbol="{idx}" value="{cls_name}" label="{cls_name}"/>'
        )
        # language=text
        symbols_xml.append(f"""
        <symbol name="{idx}" type="fill" force_rhr="0" alpha="1" clip_to_extent="1">
          <layer pass="0" class="SimpleFill" locked="0">
            <prop k="color" v="{color_str}"/>
            <prop k="style" v="solid"/>
            <prop k="outline_color" v="0,0,0,255"/>
            <prop k="outline_style" v="no"/>
            <prop k="outline_width" v="0.26"/>
          </layer>
        </symbol>""")

    # language=text
    qml_content = f"""<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.0.0">
  <renderer-v2 type="categorizedSymbol" attr="class" enableorderby="1">
    <categories>
      {"".join(categories_xml)}
    </categories>
    <symbols>
      {"".join(symbols_xml)}
    </symbols>
    <orderby>
      <orderByClause asc="1" nullsFirst="0">"priority"</orderByClause>
    </orderby>
  </renderer-v2>
</qgis>"""

    qml_path.write_text(qml_content, encoding='utf-8')
    logger.info(f"  Generated QGIS style file with priority ordering at {qml_path}")


def process_detections(conf, all_passage_data, out_dir, logger, area_name=None):
    logger.info("  Starting detection processing, projection, and export")

    detection_dir = Path(conf.detection_dir)

    geojson_path_all = out_dir / "detections.geojson"
    if area_name:
        geojson_path_vis = out_dir / f"{area_name}_detections.geojson"
        qml_path = out_dir / f"{area_name}_detections.qml"
    else:
        geojson_path_vis = out_dir / "visible_detections.geojson"
        qml_path = out_dir / "visible_detections.qml"

    image_lookup: Dict[str, Dict[str, float]] = {}
    for df_coords, passage_dir in all_passage_data:
        folder_name = passage_dir.name
        for _, row in df_coords.iterrows():
            image_name = Path(row['Image']).stem
            key = f"{folder_name}_{image_name}"
            image_lookup[key] = {
                'X_corr': row['X_corr'],
                'Y_corr': row['Y_corr'],
                'HDT_corr': row['HDT_corr']
            }

    class_polygons: Dict[str, List[Polygon]] = defaultdict(list)

    for json_path in detection_dir.glob("*.json"):
        with json_path.open('r', encoding='utf-8') as f:
            data = json.load(f)

        image_path_str = data.get("imagePath", json_path.stem)
        parts = image_path_str.split('_')
        if len(parts) < 2:
            continue

        folder_name = "_".join(parts[:-1])
        image_name = parts[-1].split('.')[0]

        key = f"{folder_name}_{image_name}"
        lookup_data = image_lookup.get(key)
        if lookup_data is None:
            logger.warning(f"  Cannot find image ({folder_name}, {image_name})")
            continue

        x_corr = lookup_data['X_corr']
        y_corr = lookup_data['Y_corr']
        hdt_corr = lookup_data['HDT_corr']

        h = data["imageHeight"]
        w = data["imageWidth"]
        pixel_to_meter = conf.span_width / w

        # Apply the -pi/2 display rotation
        effective_hdt = hdt_corr - (np.pi / 2.0)
        cos_hdt = np.cos(effective_hdt)
        sin_hdt = np.sin(effective_hdt)

        for shape_dict in data.get("shapes", []):
            label = shape_dict.get("label")
            if not label:
                continue

            pts = np.array(shape_dict["points"], dtype=np.float64)
            if len(pts) < 3:
                continue

            # Origin is top-left, rotation is applied at the center.
            # Y is inverted to match the mosaic's upward-pointing local Y axis.
            dx_pix = pts[:, 0] - w / 2.0
            dy_pix = h / 2.0 - pts[:, 1]

            dx_m = dx_pix * pixel_to_meter
            dy_m = dy_pix * pixel_to_meter

            # Rotate by effective vehicle heading and translate to global position
            x_global = x_corr + (dx_m * cos_hdt - dy_m * sin_hdt)
            y_global = y_corr + (dx_m * sin_hdt + dy_m * cos_hdt)

            # Store the polygon in the projected CRS (meters)
            proj_pts = np.column_stack((x_global, y_global))
            class_polygons[label].append(Polygon(proj_pts))

    logger.info("  Merging overlapping polygons by class and projecting to WGS84")

    # Initialize the transformer and the shapely transform wrapper
    transformer = pyproj.Transformer.from_crs(conf.crs_projected, "EPSG:4326", always_xy=True)

    features_all: List[Dict[str, Any]] = []
    features_vis: List[Dict[str, Any]] = []
    for cls_name, polys in class_polygons.items():
        if not polys:
            continue

        # Apply buffer(0) to fix microscopic self-intersections before merging
        valid_polys = [p.buffer(0) for p in polys]

        # Merge in the projected CRS
        merged_poly = unary_union(valid_polys)

        # Transform the merged geometry to WGS84
        merged_poly_wgs84 = shapely_transform(transformer.transform, merged_poly)
        geometries = [merged_poly_wgs84] if merged_poly_wgs84.geom_type == 'Polygon' else merged_poly_wgs84.geoms

        # Determine the priority integer based on the config list.
        try:
            priority_val = conf.priority_list.index(cls_name)
        except ValueError:
            priority_val = -1

        for geom in geometries:
            feat = {
                "type": "Feature",
                "properties": {
                    "class": cls_name,
                    "priority": priority_val
                },
                "geometry": mapping(geom)
            }

            # All features go into the statistics file
            features_all.append(feat)

            # Only visual classes go into the QGIS file
            if cls_name in conf.class2show:
                features_vis.append(feat)

    # Save the complete dataset for statistics
    with geojson_path_all.open('w', encoding='utf-8') as f:
        json.dump({"type": "FeatureCollection", "features": features_all}, f)

    # Save the filtered dataset for visualization
    with geojson_path_vis.open('w', encoding='utf-8') as f:
        json.dump({"type": "FeatureCollection", "features": features_vis}, f)

    active_class2show = {
        cls_name: conf.class2show[cls_name]
        for cls_name in class_polygons.keys()
        if cls_name in conf.class2show
    }

    export_qgis_style(active_class2show, qml_path, logger)
    logger.info(f"  Saved visualization GeoJSON to {geojson_path_vis} and complete GeoJSON to {geojson_path_all}")


def to_win_coords(u, v, args):
    w_low, h_low, pixel_to_meter, x_c, y_c, hdt, min_gx, max_gy, col_off, row_off = args

    dx_pix = u - w_low / 2.0
    dy_pix = h_low / 2.0 - v
    dx_m = dx_pix * pixel_to_meter
    dy_m = dy_pix * pixel_to_meter

    gx = x_c + dy_m * math.cos(hdt) + dx_m * math.sin(hdt)
    gy = y_c + dy_m * math.sin(hdt) - dx_m * math.cos(hdt)

    c = (gx - min_gx) / pixel_to_meter - col_off
    r = (max_gy - gy) / pixel_to_meter - row_off
    return [c, r]


def generate_global_cog(conf, all_passage_data, out_dir, logger, area_name=None):
    """
    Creates a single global mosaic directly on disk by iterating over spatial windows,
    blending overlapping images, and computing global statistics to avoid aux.xml generation.
    """
    if not all_passage_data:
        logger.warning("  No data provided to generate mosaic.")
        return

    # Sample first valid image to deduce dimensions and resolution
    sample_df, sample_dir = all_passage_data[0]
    sample_img_path = sample_dir / sample_df.iloc[0]['Image']
    sample_img = cv2.imread(str(sample_img_path))

    w_low = int(sample_img.shape[1] * conf.downscale_factor)
    h_low = int(sample_img.shape[0] * conf.downscale_factor)
    pixel_to_meter = conf.span_width / w_low

    # 1. Prepare image metadata and compute global bounds
    min_gx, max_gx, min_gy, max_gy = float('inf'), float('-inf'), float('inf'), float('-inf')

    image_metadata = []

    # Relative corner coordinates in meters
    corners_img = np.array([
        [-w_low / 2.0, h_low / 2.0],  # Top-Left
        [w_low / 2.0, h_low / 2.0],  # Top-Right
        [w_low / 2.0, -h_low / 2.0],  # Bottom-Right
        [-w_low / 2.0, -h_low / 2.0]  # Bottom-Left
    ])
    dx_m_arr = corners_img[:, 0] * pixel_to_meter
    dy_m_arr = corners_img[:, 1] * pixel_to_meter

    for df, passage_dir in all_passage_data:
        for _, row in df.iterrows():
            img_path = passage_dir / row['Image']
            if not img_path.exists():
                continue

            x_c, y_c, hdt = row['X_corr'], row['Y_corr'], row['HDT_corr']

            # Global geographic coordinates of the 4 corners
            corners_gx = x_c + dy_m_arr * math.cos(hdt) + dx_m_arr * math.sin(hdt)
            corners_gy = y_c + dy_m_arr * math.sin(hdt) - dx_m_arr * math.cos(hdt)

            img_min_gx, img_max_gx = corners_gx.min(), corners_gx.max()
            img_min_gy, img_max_gy = corners_gy.min(), corners_gy.max()

            min_gx = min(min_gx, img_min_gx)
            max_gx = max(max_gx, img_max_gx)
            min_gy = min(min_gy, img_min_gy)
            max_gy = max(max_gy, img_max_gy)

            image_metadata.append({
                'path': img_path,
                'x_c': x_c, 'y_c': y_c, 'hdt': hdt,
                'min_gx': img_min_gx, 'max_gx': img_max_gx,
                'min_gy': img_min_gy, 'max_gy': img_max_gy
            })

    width_px = int(math.ceil((max_gx - min_gx) / pixel_to_meter))
    height_px = int(math.ceil((max_gy - min_gy) / pixel_to_meter))

    global_transform = Affine.translation(min_gx, max_gy) * Affine.scale(pixel_to_meter, -pixel_to_meter)

    profile = {
        'driver': 'GTiff',
        'height': height_px,
        'width': width_px,
        'count': 3,
        'dtype': 'uint8',
        'crs': conf.crs_projected,
        'transform': global_transform,
        'nodata': 0,
        'tiled': True,
        'blockxsize': 256,
        'blockysize': 256,
        'compress': 'lzw',
        'interleave': 'pixel'
    }

    logger.info(f"  Allocating global GeoTIFF ({width_px}x{height_px} pixels)")
    if area_name:
        out_cog_path = out_dir / f"{area_name}_mosaic.tif"
    else:
        out_cog_path = out_dir / "mosaic.tif"
    with rasterio.open(out_cog_path, 'w', **profile) as _:
        pass  # Create empty structure

    # Mask template shrunk by 1 pixel to prevent interpolation edge artifacts
    mask_low = np.zeros((h_low, w_low), dtype=np.uint8)
    mask_low[1:-1, 1:-1] = 255

    # 2. Iterate over spatial windows
    window_size = conf.merge_window_size

    with rasterio.open(out_cog_path, 'r+') as dst:
        for row_off in range(0, height_px, window_size):
            logger.info(f'    Processing row {row_off//window_size + 1}/{height_px//window_size + 1}')
            for col_off in range(0, width_px, window_size):
                win_w = min(window_size, width_px - col_off)
                win_h = min(window_size, height_px - row_off)

                # Window bounding box in global coordinates
                win_min_gx = min_gx + (col_off * pixel_to_meter)
                win_max_gx = win_min_gx + (win_w * pixel_to_meter)
                win_max_gy = max_gy - (row_off * pixel_to_meter)
                win_min_gy = win_max_gy - (win_h * pixel_to_meter)

                # Filter images intersecting this window
                intersecting_images = [
                    meta for meta in image_metadata
                    if not (meta['max_gx'] < win_min_gx or meta['min_gx'] > win_max_gx or
                            meta['max_gy'] < win_min_gy or meta['min_gy'] > win_max_gy)
                ]

                if not intersecting_images:
                    continue

                # Accumulators for blending
                sum_arr = np.zeros((3, win_h, win_w), dtype=np.float32)
                count_arr = np.zeros((win_h, win_w), dtype=np.uint16)

                for meta in intersecting_images:
                    img = cv2.imread(str(meta['path']))
                    if img is None:
                        continue

                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_low = cv2.resize(img, (w_low, h_low))

                    x_c, y_c, hdt = meta['x_c'], meta['y_c'], meta['hdt']
                    to_win_coords_args = (w_low, h_low, pixel_to_meter, x_c, y_c, hdt, min_gx, max_gy, col_off, row_off)

                    src_pts = np.float32([[w_low / 2, h_low / 2], [w_low, h_low / 2], [w_low / 2, 0]])
                    dst_pts = np.float32([to_win_coords(w_low / 2, h_low / 2, to_win_coords_args),
                                          to_win_coords(w_low, h_low / 2, to_win_coords_args),
                                          to_win_coords(w_low / 2, 0, to_win_coords_args)])

                    M_warp = cv2.getAffineTransform(src_pts, dst_pts)

                    warped = cv2.warpAffine(img_low, M_warp, (win_w, win_h), flags=cv2.INTER_LINEAR)
                    warped_mask = cv2.warpAffine(mask_low, M_warp, (win_w, win_h), flags=cv2.INTER_NEAREST)

                    warped_chw = np.moveaxis(warped, -1, 0)
                    valid_mask = warped_mask > 0

                    # Accumulate valid pixels
                    for b in range(3):
                        sum_arr[b, valid_mask] += warped_chw[b, valid_mask]
                    count_arr[valid_mask] += 1

                # Compute average for the window
                avg_arr = np.zeros((3, win_h, win_w), dtype=np.uint8)
                valid_pixels = count_arr > 0

                for b in range(3):
                    avg_arr[b, valid_pixels] = np.clip(sum_arr[b, valid_pixels] / count_arr[valid_pixels], 0,
                                                       255).astype(np.uint8)

                dst.write(avg_arr, window=Window(col_off, row_off, win_w, win_h))

    logger.info(f"  Global mosaic generation complete: {out_cog_path}")
