import logging
import math
import cv2
import numpy as np
import pandas as pd
import pyproj
import rasterio
from pathlib import Path
from rasterio.transform import Affine

from utils import logging_conf, get_conf


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


def compute_relative_transform(conf, kp_curr, des_curr, theta_curr, kp_prev, des_prev, theta_prev):
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

    if len(good_matches) < conf.min_match_count:
        return None, None

    src_pts = np.float32([kp_curr[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_prev[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    matrix, inliers = cv2.estimateAffinePartial2D(
        src_pts,
        dst_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=conf.ransac_threshold
    )

    if matrix is None:
        return None, None

    # Build 3x3 homogeneous matrix for the scene transformation
    h_scene = np.eye(3)
    h_scene[0:2, :] = matrix

    # Invert the matrix to compute the actual camera motion
    h_cam = np.linalg.inv(h_scene)

    # Extract rigid camera translation (pixels) and rotation (radians)
    dx = h_cam[0, 2]
    dy = h_cam[1, 2]
    dtheta = math.atan2(h_cam[1, 0], h_cam[0, 0])

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
    low_res_images = []

    kp_prev = None
    des_prev = None
    x_prev = 0.0
    y_prev = 0.0
    theta_prev = 0.0
    theta_curr = 0.0
    pixel_to_meter = 0.0

    for idx, row in df.iterrows():
        img_path = passage_dir / row['Image']
        if not img_path.exists():
            logger.warning(f"Image missing: {img_path}")
            continue

        img_full = cv2.imread(str(img_path))
        img_low = cv2.resize(img_full, (0, 0), fx=conf.downscale_factor, fy=conf.downscale_factor)
        low_res_images.append((img_low, idx))

        img_stitch = cv2.resize(img_full, (0, 0), fx=conf.stitch_downscale_factor, fy=conf.stitch_downscale_factor)
        kp_curr, des_curr = extract_orb_features(conf, img_stitch)

        if idx == 0:
            x_curr = row['X_raw']
            y_curr = row['Y_raw']
            theta_curr = row['Theta_raw']
            pixel_to_meter = conf.span_width / img_stitch.shape[1]
        else:
            logger.info(f"Processing image {row['Image']}")

            trans, dtheta = compute_relative_transform(conf, kp_curr, des_curr, theta_curr, kp_prev, des_prev, theta_prev)

            if trans is not None:
                dx, dy = trans
                forward_step = dy * pixel_to_meter
                lateral_step = dx * pixel_to_meter
                x_curr = x_prev + forward_step * math.cos(theta_prev) - lateral_step * math.sin(theta_prev)
                y_curr = y_prev + forward_step * math.sin(theta_prev) + lateral_step * math.cos(theta_prev)
                theta_curr = theta_prev + dtheta
            else:
                logger.info(f"Stitching failed for {row['Image']}, using fallback")
                x_kin = x_prev + math.cos(theta_prev) * conf.vehicle_step
                y_kin = y_prev + math.sin(theta_prev) * conf.vehicle_step
                x_curr = conf.weight_gps_v_kin * row['X_raw'] + (1 - conf.weight_gps_v_kin) * x_kin
                y_curr = conf.weight_gps_v_kin * row['Y_raw'] + (1 - conf.weight_gps_v_kin) * y_kin
                theta_curr = row['Theta_raw']

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

    df_out = pd.DataFrame(corrected_data)
    return df_out, low_res_images


def load_low_res_images(conf, df_coords, passage_dir, logger):
    low_res_images = []

    for idx, row in df_coords.iterrows():
        img_path = passage_dir / row['Image']

        if not img_path.exists():
            logger.warning(f"Image not found at {img_path}")
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning(f"Failed to decode image at {img_path}")
            continue

        # Resize to match the logic previously handled in process_passage
        img_low = cv2.resize(img, (0, 0), fx=conf.downscale_factor, fy=conf.downscale_factor)
        low_res_images.append((img_low, idx))

    return low_res_images


def create_oriented_mosaic(conf, df_coords, low_res_images, out_tif_path, logger):
    # 1. Compute global passage heading to build an Oriented Bounding Box (OBB)
    x0, y0 = df_coords.iloc[0]['X_corr'], df_coords.iloc[0]['Y_corr']
    xn, yn = df_coords.iloc[-1]['X_corr'], df_coords.iloc[-1]['Y_corr']
    alpha = math.atan2(yn - y0, xn - x0)

    cos_a = math.cos(-alpha)
    sin_a = math.sin(-alpha)

    # 2. Transform coordinates to local passage-aligned frame
    x_prime_list = []
    y_prime_list = []

    for _, row in df_coords.iterrows():
        dx = row['X_corr'] - x0
        dy = row['Y_corr'] - y0
        x_prime = dx * cos_a - dy * sin_a
        y_prime = dx * sin_a + dy * cos_a
        x_prime_list.append(x_prime)
        y_prime_list.append(y_prime)

    # Calculate spatial resolution and dynamic bounding box buffer
    sample_img = low_res_images[0][0]
    h, w = sample_img.shape[:2]
    pixel_to_meter = conf.span_width / w

    # Maximum physical distance from image center to any corner in meters
    max_radius_meters = math.hypot(w * pixel_to_meter, h * pixel_to_meter) / 2.0

    min_x_prime = min(x_prime_list) - max_radius_meters
    max_x_prime = max(x_prime_list) + max_radius_meters
    min_y_prime = min(y_prime_list) - max_radius_meters
    max_y_prime = max(y_prime_list) + max_radius_meters

    pixel_to_meter = conf.span_width / low_res_images[0][0].shape[1]

    width_px = int((max_x_prime - min_x_prime) / pixel_to_meter)
    height_px = int((max_y_prime - min_y_prime) / pixel_to_meter)

    mosaic_canvas = np.zeros((height_px, width_px, 3), dtype=np.float32)
    weight_canvas = np.zeros((height_px, width_px, 1), dtype=np.float32)

    # 3. Paste images into the local bounding box
    for (img_low, idx), x_prime, y_prime in zip(low_res_images, x_prime_list, y_prime_list):
        theta_global = df_coords.iloc[idx]['Theta_corr']
        theta_local = theta_global - alpha - np.pi/2

        cx_px = (x_prime - min_x_prime) / pixel_to_meter
        cy_px = (max_y_prime - y_prime) / pixel_to_meter

        h, w = img_low.shape[:2]

        # OpenCV rotation is counter-clockwise for negative angles, adapt as needed
        M = cv2.getRotationMatrix2D((w / 2, h / 2), math.degrees(theta_local), 1.0)
        M[0, 2] += cx_px - w / 2
        M[1, 2] += cy_px - h / 2

        warped_img = cv2.warpAffine(img_low.astype(np.float32), M, (width_px, height_px))
        mask = cv2.warpAffine(np.ones((h, w), dtype=np.float32), M, (width_px, height_px))

        mosaic_canvas += warped_img
        weight_canvas += mask[..., np.newaxis]

    np.divide(mosaic_canvas, weight_canvas, out=mosaic_canvas, where=weight_canvas > 0)
    mosaic_canvas = mosaic_canvas.astype(np.uint8)

    # 4. Construct Affine transform mapping pixel (col, row) to global (X, Y)
    transform = (
            Affine.translation(x0, y0) *
            Affine.rotation(math.degrees(alpha)) *
            Affine.translation(min_x_prime, max_y_prime) *
            Affine.scale(pixel_to_meter, -pixel_to_meter)
    )

    with rasterio.open(
            out_tif_path,
            'w',
            driver='GTiff',
            height=mosaic_canvas.shape[0],
            width=mosaic_canvas.shape[1],
            count=3,
            dtype=mosaic_canvas.dtype,
            crs=conf.crs_projected,
            transform=transform,
    ) as dest:
        dest.write(mosaic_canvas[:, :, 2], 1)
        dest.write(mosaic_canvas[:, :, 1], 2)
        dest.write(mosaic_canvas[:, :, 0], 3)

    logger.info(f"Mosaic created at {out_tif_path}")


def merge_all_passages(conf, tif_paths, out_path, logger):
    """
    Merges multiple passage GeoTIFFs into a single global GeoTIFF.
    Overlapping regions are averaged.
    """
    # 1. Compute global bounds and extract base resolution
    min_x, min_y, max_x, max_y = float('inf'), float('inf'), float('-inf'), float('-inf')
    pixel_to_meter = None

    for path in tif_paths:
        with rasterio.open(path) as src:
            bounds = src.bounds
            min_x = min(min_x, bounds.left)
            min_y = min(min_y, bounds.bottom)
            max_x = max(max_x, bounds.right)
            max_y = max(max_y, bounds.top)

            if pixel_to_meter is None:
                pixel_to_meter = src.transform.a

    # 2. Compute global dimensions
    width_px = int(np.ceil((max_x - min_x) / pixel_to_meter))
    height_px = int(np.ceil((max_y - min_y) / pixel_to_meter))

    global_transform = Affine.translation(min_x, max_y) * Affine.scale(pixel_to_meter, -pixel_to_meter)

    # Use float32 for accumulation to prevent overflow
    global_canvas = np.zeros((3, height_px, width_px), dtype=np.float32)
    global_weight = np.zeros((1, height_px, width_px), dtype=np.float32)

    # 3. Accumulate data
    for path in tif_paths:
        with rasterio.open(path) as src:
            data = src.read()  # Shape: (3, H, W)

            # Compute offsets in the global canvas
            col_offset = int(round((src.bounds.left - min_x) / pixel_to_meter))
            row_offset = int(round((max_y - src.bounds.top) / pixel_to_meter))

            h, w = data.shape[1], data.shape[2]

            # Valid pixels mask (assuming black/0 is nodata)
            valid_mask = np.any(data > 0, axis=0)[np.newaxis, ...]

            global_canvas[:, row_offset:row_offset + h, col_offset:col_offset + w] += (data * valid_mask)
            global_weight[:, row_offset:row_offset + h, col_offset:col_offset + w] += valid_mask

    # 4. Average and cast back to uint8
    np.divide(global_canvas, global_weight, out=global_canvas, where=global_weight > 0)
    global_canvas = global_canvas.astype(np.uint8)

    # 5. Write out the final merged raster
    with rasterio.open(
            out_path,
            'w',
            driver='GTiff',
            height=height_px,
            width=width_px,
            count=3,
            dtype=global_canvas.dtype,
            crs=conf.crs_projected,
            transform=global_transform,
            compress='lzw'
    ) as dest:
        dest.write(global_canvas)

    logger.info(f"Global merged mosaic created at {out_path}")


def main():
    logging_conf()
    logger = logging.getLogger('Mosaic')
    conf = get_conf(logger)

    out_dir = Path(conf.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    input_dir = Path(conf.input_dir)

    generated_tifs = []

    for passage_dir in input_dir.iterdir():
        if not passage_dir.is_dir():
            continue

        passage_name = passage_dir.name
        out_csv = out_dir / f"{passage_name}.csv"
        out_tif = out_dir / f"{passage_name}.tif"

        '''
        if out_csv.exists():
            logger.info(f"CSV found for {passage_name}")
            df_coords = pd.read_csv(out_csv)
            low_res_images = load_low_res_images(conf, df_coords, passage_dir, logger)
        else:
        '''
        logger.info(f'Processing passage {passage_name}')
        df_coords, low_res_images = process_passage(conf, passage_dir, logger)
        if df_coords is None or df_coords.empty or not low_res_images:
            logger.error(f"Failed to process passage {passage_name}")
            continue
        df_coords.to_csv(out_csv, index=False)

        logger.info(f"Creating mosaic for {passage_name}")
        create_oriented_mosaic(conf, df_coords, low_res_images, out_tif, logger)
        generated_tifs.append(out_tif)

    if generated_tifs:
        logger.info("Merging all passages into a single global mosaic...")
        merged_out_path = out_dir / "mosaic.tif"
        merge_all_passages(conf, generated_tifs, merged_out_path, logger)

if __name__ == '__main__':
    main()
