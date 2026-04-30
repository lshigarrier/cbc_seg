import logging
import math
import cv2
import numpy as np
import pandas as pd
import pyproj
import rasterio
import json
from pathlib import Path
from collections import defaultdict
from rasterio.transform import Affine
from rasterio.windows import Window
from shapely.geometry import Polygon, mapping
from shapely.ops import unary_union
from shapely.ops import transform as shapely_transform

from utils import logging_conf, get_conf, CustomTimer


def read_uge_gps(file_path):
    """
    Load the Images.txt file into a pandas DataFrame.
    Returns a DataFrame strictly with columns: ['Image', 'Latitude', 'Longitude']
    """
    df = pd.read_csv(file_path, sep='\t')
    required_columns = ['Image', 'Latitude', 'Longitude']
    return df[required_columns]


def compute_projection_and_orientation(conf, df):
    """
        Projects GPS coordinates to a local CRS, computes symmetrically smoothed headings,
        and applies a local spatial offset to deduce the true camera position.
        """
    transformer = pyproj.Transformer.from_crs(conf.crs_gps, conf.crs_projected, always_xy=True)

    # Vectorized coordinate transformation
    x_gps, y_gps = transformer.transform(df['Longitude'].values, df['Latitude'].values)

    # Vectorized raw heading computation
    dx = np.diff(x_gps)
    dy = np.diff(y_gps)
    theta_raw = np.arctan2(dy, dx)

    # Handle the last point by duplicating the second-to-last heading
    if len(theta_raw) > 0:
        theta_raw = np.append(theta_raw, theta_raw[-1])
    else:
        theta_raw = np.array([0.0])

    # Unwrap angles to avoid discontinuities at pi / -pi before smoothing
    theta_unwrapped = np.unwrap(theta_raw)

    # Build a symmetric triangular window for weighted averaging
    window_size = conf.heading_smoothing_window
    if window_size % 2 == 0:
        window_size += 1
    half_window = window_size // 2

    # Weights decay linearly from the center (e.g., [1, 2, 3, 2, 1] for size 5)
    weights = np.concatenate((np.arange(1, half_window + 2), np.arange(half_window, 0, -1)))
    weights = weights / weights.sum()

    # Pad the boundaries using the edge values to maintain output length and handle ends properly
    theta_padded = np.pad(theta_unwrapped, half_window, mode='edge')
    theta_smoothed = np.convolve(theta_padded, weights, mode='valid')

    # Apply spatial offset using the smoothed vehicle heading
    # Standard orientation: forward vector is (cos(theta), sin(theta))
    # cam_offset_y is assumed to be forward (longitudinal), cam_offset_x is right (lateral)
    x_cam = x_gps + (conf.cam_offset_y * np.cos(theta_smoothed)) + (conf.cam_offset_x * np.sin(theta_smoothed))
    y_cam = y_gps + (conf.cam_offset_y * np.sin(theta_smoothed)) - (conf.cam_offset_x * np.cos(theta_smoothed))

    df['X_raw'] = x_cam
    df['Y_raw'] = y_cam

    # Wrap back to [-pi, pi] range
    df['Theta_raw'] = (theta_smoothed + np.pi) % (2 * np.pi) - np.pi

    return df


def extract_orb_features(conf, image):
    orb = cv2.ORB_create(nfeatures=conf.orb_nfeatures)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    return keypoints, descriptors


def compute_relative_transform(conf, kp_curr, des_curr, theta_curr, kp_prev, des_prev, theta_prev, pixel_to_meter):
    if des_curr is None or des_prev is None or len(des_curr) < 2 or len(des_prev) < 2:
        return None, None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des_curr, des_prev, k=2)

    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < conf.orb_ratio_threshold * n.distance:
                good_matches.append(m)

    valid_matches = []
    for m in good_matches:
        pt_curr = kp_curr[m.queryIdx].pt
        pt_prev = kp_prev[m.trainIdx].pt

        # dy_pt is the vertical pixel displacement for this specific point
        dy_pt = pt_curr[1] - pt_prev[1]

        # Positive step means forward motion
        if 0 < dy_pt:
            valid_matches.append(m)

    if len(valid_matches) < conf.min_match_count:
        return None, None

    src_pts = np.float32([kp_curr[m.queryIdx].pt for m in valid_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_prev[m.trainIdx].pt for m in valid_matches]).reshape(-1, 1, 2)

    matrix, inliers = cv2.estimateAffinePartial2D(
        src_pts,
        dst_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=conf.ransac_threshold
    )

    if matrix is None or inliers is None:
        return None, None

    inlier_ratio = np.sum(inliers) / len(src_pts)
    if inlier_ratio < conf.min_inlier_ratio:
        return None, None

    scale = math.hypot(matrix[0, 0], matrix[1, 0])
    if abs(scale - 1.0) > conf.max_scale_deviation:
        return None, None

    # Build 3x3 homogeneous matrix for the scene transformation
    h_scene = np.eye(3)
    h_scene[0:2, :] = matrix

    # Invert the matrix to compute the actual camera motion
    try:
        h_cam = np.linalg.inv(h_scene)
    except np.linalg.LinAlgError:
        return None, None

    # Extract rigid camera translation (pixels) and rotation (radians)
    dx = h_cam[0, 2]
    dy = h_cam[1, 2]
    dtheta = math.atan2(h_cam[1, 0], h_cam[0, 0])

    # Lateral drift check (orthogonal distance to kinematic axis)
    if abs(dx) * pixel_to_meter > conf.max_lateral_drift:
        return None, None

    # Validation against GPS prior
    expected_dtheta = theta_curr - theta_prev
    angle_diff = dtheta - expected_dtheta
    angle_diff = math.atan2(math.sin(angle_diff), math.cos(angle_diff))

    if abs(angle_diff) > conf.max_angle_deviation_rad:
        return None, None

    return (dx, dy), dtheta


def process_passage(conf, passage_dir, logger):
    txt_path = passage_dir / 'Images.txt'

    # Format selection logic
    if txt_path.exists():
        read_gps = read_uge_gps
        data_path = txt_path
    # elif (passage_dir / "AnotherFormat.csv").exists():
    #     read_gps = read_other_format_gps
    #     data_path = passage_dir / "AnotherFormat.csv"
    else:
        logger.warning(f"No GPS file found in {passage_dir}")
        return None, None

    # Load and process geographic data
    df_initial = read_gps(data_path)
    df = compute_projection_and_orientation(conf, df_initial)

    corrected_data = []

    kp_prev = None
    des_prev = None
    x_prev = 0.0
    y_prev = 0.0
    theta_prev = 0.0
    theta_curr = 0.0
    pixel_to_meter = 0.0
    group_start_idx = 0

    # Inner function to apply rigid transformation to the current group
    def correct_rigid_group(start_idx, end_idx):
        if start_idx >= end_idx:
            return

        # Extract corrected and raw values
        x_c = np.array([corrected_data[i]['X_corr'] for i in range(start_idx, end_idx)])
        y_c = np.array([corrected_data[i]['Y_corr'] for i in range(start_idx, end_idx)])
        th_c = np.array([corrected_data[i]['Theta_corr'] for i in range(start_idx, end_idx)])

        # Raw values matching the group indices
        x_r = df['X_raw'].values[start_idx:end_idx]
        y_r = df['Y_raw'].values[start_idx:end_idx]
        th_r = df['Theta_raw'].values[start_idx:end_idx]

        # 1. Compute optimal theta offset
        d_th = th_r - th_c
        mean_d_theta = math.atan2(np.mean(np.sin(d_th)), np.mean(np.cos(d_th)))

        # 2. Apply rotation around the first point of the group (pivot)
        x_pivot, y_pivot = x_c[0], y_c[0]
        cos_th, sin_th = math.cos(mean_d_theta), math.sin(mean_d_theta)

        x_rot = x_pivot + (x_c - x_pivot) * cos_th - (y_c - y_pivot) * sin_th
        y_rot = y_pivot + (x_c - x_pivot) * sin_th + (y_c - y_pivot) * cos_th
        th_rot = th_c + mean_d_theta

        # 3. Compute optimal X, Y translation offsets on the rotated points
        offset_x = np.mean(x_r - x_rot)
        offset_y = np.mean(y_r - y_rot)

        # 4. Apply final translation and update dictionaries
        x_final = x_rot + offset_x
        y_final = y_rot + offset_y

        for i, global_i in enumerate(range(start_idx, end_idx)):
            corrected_data[global_i]['X_corr'] = x_final[i]
            corrected_data[global_i]['Y_corr'] = y_final[i]
            corrected_data[global_i]['Theta_corr'] = th_rot[i]

    for idx, row in df.iterrows():
        img_path = passage_dir / row['Image']
        if not img_path.exists():
            logger.warning(f"Image missing: {img_path}")
            continue

        img_full = cv2.imread(str(img_path))
        img_stitch = cv2.resize(img_full, (0, 0), fx=conf.stitch_downscale_factor, fy=conf.stitch_downscale_factor)
        kp_curr, des_curr = extract_orb_features(conf, img_stitch)

        if idx == 0:
            x_curr = row['X_raw']
            y_curr = row['Y_raw']
            theta_curr = row['Theta_raw']
            pixel_to_meter = conf.span_width / img_stitch.shape[1]
        else:
            # Calculate pure kinematic prediction
            x_kin = x_prev + math.cos(theta_prev) * conf.vehicle_step
            y_kin = y_prev + math.sin(theta_prev) * conf.vehicle_step

            # Vector from previous point to current GPS point
            dx_gps = row['X_raw'] - x_prev
            dy_gps = row['Y_raw'] - y_prev

            # Orthogonal distance: Y-component after rotating by -theta_prev
            lateral_dist_gps = abs(-dx_gps * math.sin(theta_prev) + dy_gps * math.cos(theta_prev))

            # Overwrite GPS if lateral drift from kinematic axis is too high
            if lateral_dist_gps > conf.max_gps_kin_dist:
                df.at[idx, 'X_raw'] = x_kin
                df.at[idx, 'Y_raw'] = y_kin
                row['X_raw'] = x_kin
                row['Y_raw'] = y_kin

            trans, dtheta = compute_relative_transform(conf,
                                                       kp_curr, des_curr, theta_curr,
                                                       kp_prev, des_prev, theta_prev,
                                                       pixel_to_meter)

            if trans is not None:
                dx, dy = trans
                forward_step = dy * pixel_to_meter
                lateral_step = dx * pixel_to_meter
                x_curr = x_prev + forward_step * math.cos(theta_prev) - lateral_step * math.sin(theta_prev)
                y_curr = y_prev + forward_step * math.sin(theta_prev) + lateral_step * math.cos(theta_prev)
                theta_curr = theta_prev + dtheta
            else:
                x_curr = row['X_raw']
                y_curr = row['Y_raw']
                theta_curr = row['Theta_raw']
                # End of rigid group. Correct the previous block and start a new one.
                correct_rigid_group(group_start_idx, len(corrected_data))
                group_start_idx = len(corrected_data)

        corrected_data.append({
            'Image': row['Image'],
            'X_corr': x_curr,
            'Y_corr': y_curr,
            'Theta_corr': theta_curr
        })

        kp_prev = kp_curr
        des_prev = des_curr
        x_prev = x_curr
        y_prev = y_curr
        theta_prev = theta_curr

    correct_rigid_group(group_start_idx, len(corrected_data))

    df_out = pd.DataFrame(corrected_data)
    return df_out


def export_qgis_style(conf, qml_path, logger):
    categories_xml = []
    symbols_xml = []

    for idx, (cls_name, color) in enumerate(conf.class2show.items()):
        color_str = f"{color[0]},{color[1]},{color[2]},255"

        categories_xml.append(
            f'<category symbol="{idx}" value="{cls_name}" label="{cls_name}"/>'
        )
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
    logger.info(f"Generated QGIS style file with priority ordering at {qml_path}")


def process_detections(conf, all_passage_data, out_dir, logger):
    logger.info("Starting detection processing, projection, and export")

    detection_dir = Path(conf.detection_dir)

    geojson_path_all = out_dir / "detections.geojson"
    geojson_path_vis = out_dir / "visible_detections.geojson"
    qml_path = out_dir / "visible_detections.qml"

    image_lookup = {}
    for df_coords, passage_dir in all_passage_data:
        folder_name = passage_dir.name
        for _, row in df_coords.iterrows():
            image_name = Path(row['Image']).stem
            image_lookup[(folder_name, image_name)] = {
                'X_corr': row['X_corr'],
                'Y_corr': row['Y_corr'],
                'Theta_corr': row['Theta_corr']
            }

    class_polygons = defaultdict(list)

    for json_path in detection_dir.glob("*.json"):
        with json_path.open('r', encoding='utf-8') as f:
            data = json.load(f)

        image_path_str = data.get("imagePath", json_path.stem)
        parts = image_path_str.split('_')
        if len(parts) < 2:
            continue

        folder_name = "_".join(parts[:-1])
        image_name = parts[-1].split('.')[0]

        lookup_data = image_lookup.get((folder_name, image_name))
        if lookup_data is None:
            logger.warning(f"Cannot find image ({folder_name}, {image_name})")
            continue

        x_corr = lookup_data['X_corr']
        y_corr = lookup_data['Y_corr']
        theta_corr = lookup_data['Theta_corr']

        h = data["imageHeight"]
        w = data["imageWidth"]
        pixel_to_meter = conf.span_width / w

        # Apply the -pi/2 display rotation
        effective_theta = theta_corr - (np.pi / 2.0)
        cos_theta = np.cos(effective_theta)
        sin_theta = np.sin(effective_theta)

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
            x_global = x_corr + (dx_m * cos_theta - dy_m * sin_theta)
            y_global = y_corr + (dx_m * sin_theta + dy_m * cos_theta)

            # Store the polygon in the projected CRS (meters)
            proj_pts = np.column_stack((x_global, y_global))
            class_polygons[label].append(Polygon(proj_pts))

    logger.info("Merging overlapping polygons by class and projecting to WGS84")

    # Initialize the transformer and the shapely transform wrapper
    transformer = pyproj.Transformer.from_crs(conf.crs_projected, "EPSG:4326", always_xy=True)

    features_all = []
    features_vis = []
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

    export_qgis_style(conf, qml_path, logger)
    logger.info(f"Saved visualization GeoJSON to {geojson_path_vis} and complete GeoJSON to {geojson_path_all}")


def generate_global_cog(conf, all_passage_data, out_dir, logger):
    """
    Creates a single global mosaic directly on disk by iterating over spatial windows,
    blending overlapping images, and computing global statistics to avoid aux.xml generation.
    """
    if not all_passage_data:
        logger.warning("No data provided to generate mosaic.")
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

            x_c, y_c, theta = row['X_corr'], row['Y_corr'], row['Theta_corr']

            # Global geographic coordinates of the 4 corners
            corners_gx = x_c + dy_m_arr * math.cos(theta) + dx_m_arr * math.sin(theta)
            corners_gy = y_c + dy_m_arr * math.sin(theta) - dx_m_arr * math.cos(theta)

            img_min_gx, img_max_gx = corners_gx.min(), corners_gx.max()
            img_min_gy, img_max_gy = corners_gy.min(), corners_gy.max()

            min_gx = min(min_gx, img_min_gx)
            max_gx = max(max_gx, img_max_gx)
            min_gy = min(min_gy, img_min_gy)
            max_gy = max(max_gy, img_max_gy)

            image_metadata.append({
                'path': img_path,
                'x_c': x_c, 'y_c': y_c, 'theta': theta,
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

    logger.info(f"Allocating global GeoTIFF ({width_px}x{height_px} pixels)")
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
            logger.info(f'Processing row {row_off//window_size + 1}/{height_px//window_size + 1}')
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

                    x_c, y_c, theta = meta['x_c'], meta['y_c'], meta['theta']

                    def to_win_coords(u, v):
                        dx_pix = u - w_low / 2.0
                        dy_pix = h_low / 2.0 - v
                        dx_m = dx_pix * pixel_to_meter
                        dy_m = dy_pix * pixel_to_meter

                        gx = x_c + dy_m * math.cos(theta) + dx_m * math.sin(theta)
                        gy = y_c + dy_m * math.sin(theta) - dx_m * math.cos(theta)

                        c = (gx - min_gx) / pixel_to_meter - col_off
                        r = (max_gy - gy) / pixel_to_meter - row_off
                        return [c, r]

                    src_pts = np.float32([[w_low / 2, h_low / 2], [w_low, h_low / 2], [w_low / 2, 0]])
                    dst_pts = np.float32([to_win_coords(w_low / 2, h_low / 2),
                                          to_win_coords(w_low, h_low / 2),
                                          to_win_coords(w_low / 2, 0)])

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

    logger.info(f"Global mosaic generation complete: {out_cog_path}")


def main():
    logging_conf()
    logger = logging.getLogger('Mosaic')
    conf = get_conf(logger, verbose=False)

    out_dir = Path(conf.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    input_dir = Path(conf.input_dir)

    timer = CustomTimer()
    timer.start()

    paths_file = input_dir / "paths.txt"
    if paths_file.is_file():
        logger.info(f"Found {paths_file.name}, reading passage directories from file")
        with paths_file.open('r', encoding='utf-8') as f:
            passage_dirs = [Path(line.strip().strip('\'"')) for line in f if line.strip()]
    else:
        logger.info(f"No paths.txt found in {input_dir}, iterating over subdirectories")
        passage_dirs = list(input_dir.iterdir())

    all_passage_data = []

    for passage_dir in passage_dirs:
        if not passage_dir.is_dir():
            continue

        passage_name = passage_dir.name
        out_csv = out_dir / f"{passage_name}.csv"

        if conf.read_csv and out_csv.exists():
            logger.info(f"CSV found for {passage_name}")
            df_coords = pd.read_csv(out_csv)
        else:
            logger.info(f'Processing passage {passage_name}')
            df_coords = process_passage(conf, passage_dir, logger)
            if df_coords is None or df_coords.empty:
                logger.error(f"Failed to process passage {passage_name}")
                continue
            df_coords.to_csv(out_csv, index=False)

        all_passage_data.append((df_coords, passage_dir))

    if conf.generate_detections:
        process_detections(conf, all_passage_data, out_dir, logger)

    if conf.generate_mosaic:
        generate_global_cog(conf, all_passage_data, out_dir, logger)

    timer.stop(logger, show_time_per_image=False)

if __name__ == '__main__':
    main()
