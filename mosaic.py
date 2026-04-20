import pandas as pd
import numpy as np
import logging
import cv2
import rasterio
from rasterio.transform import Affine
from pathlib import Path
from pyproj import Transformer
from scipy.optimize import minimize

from utils import get_conf, logging_conf


def load_uge_passage_data(images_txt_path):
    """
    Load the Images.txt file into a pandas DataFrame.
    Returns a DataFrame strictly with columns: ['Image', 'Latitude', 'Longitude']
    """
    # Assuming tab-separated values based on your snippet
    df = pd.read_csv(images_txt_path, sep='\t')
    # Define the strict subset of columns required
    required_columns = ['Image', 'Latitude', 'Longitude']
    # Extract only the required columns
    df = df[required_columns]
    return df


def convert_to_local_crs(df, src_epsg=4326, dst_epsg=2154):
    """
    Convert Latitude/Longitude to local metric coordinates (X, Y).
    """
    # always_xy=True forces the order to be (Longitude, Latitude) -> (X, Y)
    transformer = Transformer.from_crs(
        f"EPSG:{src_epsg}",
        f"EPSG:{dst_epsg}",
        always_xy=True
    )

    # pyproj expects (lon, lat) when always_xy=True
    lon = df['Longitude'].values
    lat = df['Latitude'].values

    x, y = transformer.transform(lon, lat)

    df['X'] = x
    df['Y'] = y

    return df


def compute_robust_orientations(df, alpha=0.3):
    """
    Compute raw orientations from consecutive images, then apply a
    reverse Exponential Moving Average (EMA) to smooth them.
    alpha: Weight of the current observation.
    """
    x = df['X'].values
    y = df['Y'].values
    n = len(df)

    raw_theta = np.zeros(n)

    # 1. Compute raw orientations (difference between next and current)
    for i in range(n - 1):
        dx = x[i + 1] - x[i]
        dy = y[i + 1] - y[i]
        raw_theta[i] = np.arctan2(dy, dx)

    # Last image gets the same orientation as the penultimate
    if n > 1:
        raw_theta[-1] = raw_theta[-2]

    # 2. Reverse Exponential Moving Average (EMA)
    smoothed_theta = np.zeros(n)
    smoothed_theta[-1] = raw_theta[-1]  # Start from the end

    for i in range(n - 2, -1, -1):
        # We must calculate the angular difference carefully to avoid wrap-around issues
        # (e.g., so that the average of 359 deg and 1 deg is 0, not 180)
        diff = (raw_theta[i] - smoothed_theta[i + 1] + np.pi) % (2 * np.pi) - np.pi

        # Apply EMA: S_i = S_{i+1} + alpha * difference
        smoothed_theta[i] = smoothed_theta[i + 1] + alpha * diff

        # Normalize back to [-pi, pi] just to keep values clean
        smoothed_theta[i] = (smoothed_theta[i] + np.pi) % (2 * np.pi) - np.pi

    df['Theta_rad'] = smoothed_theta
    return df


def process_passage(passage_dir, conf, logger):
    """Process a single passage folder."""
    images_txt_path = passage_dir / "Images.txt"
    if images_txt_path.exists():
        load_passage_data = load_uge_passage_data
    else:
        logger.warning(f"No GPS file in {passage_dir}")
        return None

    df = load_passage_data(images_txt_path)
    df = convert_to_local_crs(df, dst_epsg=conf.epsg)
    df = compute_robust_orientations(df, alpha=0.75)

    return df


def apply_2d_hanning(image_crop):
    """
    Applies a 2D Hanning window to a cropped image region.
    This fades the edges to zero, preventing sharp edge artifacts during FFT.
    """
    # Ensure image is grayscale (or float for math)
    if len(image_crop.shape) == 3:
        crop_gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
    else:
        crop_gray = image_crop.copy()

    h, w = crop_gray.shape

    # Create 1D hanning windows for height and width
    han_h = np.hanning(h)
    han_w = np.hanning(w)

    # Outer product creates the 2D window
    window_2d = np.outer(han_h, han_w)

    # Apply window
    windowed_crop = crop_gray.astype(np.float32) * window_2d
    return windowed_crop


def compute_phase_correlation_sweep(crop_prev, crop_curr, conf):
    """
    Sweeps through a range of angles, rotating the current crop and computing
    phase correlation against the previous crop to find the best translation and rotation.

    Returns:
        best_dx, best_dy, best_angle, best_response
    """
    # 1. Prepare the fixed previous image
    windowed_prev = apply_2d_hanning(crop_prev)

    best_response = -1.0
    best_dx, best_dy, best_angle = 0.0, 0.0, 0.0

    h, w = crop_curr.shape[:2]
    center = (w // 2, h // 2)

    # Generate array of angles (e.g., -10.0 to 10.0 inclusive)
    num_steps = int((conf.angle_max - conf.angle_min) / conf.angle_step) + 1
    angles = np.linspace(conf.angle_min, conf.angle_max, num_steps)

    for angle in angles:
        # 2. Rotate the current crop
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # We use BORDER_REPLICATE so the edges stretch out instead of introducing black triangles
        rotated_curr = cv2.warpAffine(
            crop_curr, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )

        # 3. Apply Hanning window to the cleanly rotated image
        windowed_curr = apply_2d_hanning(rotated_curr)

        # 4. Compute Phase Correlation
        # cv2.phaseCorrelate takes floats and returns: shift (dx, dy) and response (0 to 1)
        shift, response = cv2.phaseCorrelate(windowed_prev, windowed_curr)

        # 5. Track the maximum response
        if response > best_response:
            best_response = response
            best_dx = shift[0]
            best_dy = shift[1]
            best_angle = angle

    return best_dx, best_dy, best_angle, best_response


def compute_step_optimization(
        prev_state, curr_gps_state, vision_target,
        crop_center_prev, crop_center_curr, image_center,
        meters_per_pixel, vehicle_step,
        weight_gps=1.0, weight_vision=10.0, weight_kinematic=2.0
):
    # Offsets from image center to crop centers (in meters)
    offset_x_prev = (crop_center_prev[0] - image_center[0]) * meters_per_pixel
    offset_y_prev = (crop_center_prev[1] - image_center[1]) * meters_per_pixel

    offset_x_curr = (crop_center_curr[0] - image_center[0]) * meters_per_pixel
    offset_y_curr = (crop_center_curr[1] - image_center[1]) * meters_per_pixel

    # Initial guess uses smoothed GPS theta
    initial_guess = np.array([curr_gps_state['X'], curr_gps_state['Y'], curr_gps_state['Theta']])

    # Helper to properly compute angular difference (-pi to pi)
    def angle_diff(t1, t2):
        return np.arctan2(np.sin(t1 - t2), np.cos(t1 - t2))

    def loss_function(params):
        X, Y, Theta = params

        # 1. GPS Loss (Compare X, Y, Theta to current GPS)
        loss_gps = (X - curr_gps_state['X']) ** 2 + (Y - curr_gps_state['Y']) ** 2
        loss_gps += angle_diff(Theta, curr_gps_state['Theta']) ** 2

        # 2. Kinematic Loss (Compare X, Y, Theta to expected forward step)
        cos_prev, sin_prev = np.cos(prev_state['Theta']), np.sin(prev_state['Theta'])
        kin_x = prev_state['X'] + vehicle_step * cos_prev
        kin_y = prev_state['Y'] + vehicle_step * sin_prev

        loss_kinematic = (X - kin_x) ** 2 + (Y - kin_y) ** 2
        loss_kinematic += angle_diff(Theta, prev_state['Theta']) ** 2

        # 3. Vision Loss
        loss_vision = 0.0
        if vision_target is not None:
            # Step A: Find Prev Crop Center
            prev_crop_x = prev_state['X'] + (offset_x_prev * cos_prev - offset_y_prev * sin_prev)
            prev_crop_y = prev_state['Y'] + (offset_x_prev * sin_prev + offset_y_prev * cos_prev)

            # Step B: Apply vision translation to find expected Curr Crop Center
            # (Phase correlation dx, dy are relative to prev image's unrotated axes)
            dx_m = vision_target['dx'] * meters_per_pixel
            dy_m = vision_target['dy'] * meters_per_pixel

            curr_crop_x = prev_crop_x + (dx_m * cos_prev - dy_m * sin_prev)
            curr_crop_y = prev_crop_y + (dx_m * sin_prev + dy_m * cos_prev)

            # Step C: Subtract current offset to find Expected Curr Image Center
            expected_theta = prev_state['Theta'] + np.deg2rad(vision_target['dtheta'])
            cos_exp, sin_exp = np.cos(expected_theta), np.sin(expected_theta)

            expected_X = curr_crop_x - (offset_x_curr * cos_exp - offset_y_curr * sin_exp)
            expected_Y = curr_crop_y - (offset_x_curr * sin_exp + offset_y_curr * cos_exp)

            # Compute Vision Loss terms
            err_x = X - expected_X
            err_y = Y - expected_Y
            err_theta = angle_diff(Theta, expected_theta)

            v_weight = weight_vision * vision_target['weight']
            loss_vision = v_weight * (err_x ** 2 + err_y ** 2 + err_theta ** 2)

        return (weight_gps * loss_gps) + (weight_kinematic * loss_kinematic) + loss_vision

    # Run the optimization
    result = minimize(loss_function, initial_guess, method='BFGS')

    return result.x[0], result.x[1], result.x[2]


def compute_passage_mosaic(passage_df, images_dir, conf, logger):
    """
    Main function to compute the mosaic for a single passage.

    Parameters:
    - passage_df: DataFrame containing ['Image', 'Latitude', 'Longitude', 'X', 'Y', 'Theta_rad']
    - images_dir: Path to the directory containing the images
    """

    optimized_positions = []

    # --- 1) Initialize the first image ---
    # We fix the first image at its starting GPS position and initial heading
    initial_state = {
        'Image': passage_df.iloc[0]['Image'],
        'X': passage_df.iloc[0]['X'],
        'Y': passage_df.iloc[0]['Y'],
        'Theta': passage_df.iloc[0]['Theta_rad']
    }
    optimized_positions.append(initial_state)

    # Load the very first image to start the loop
    prev_img_path = images_dir / passage_df.iloc[0]['Image']
    prev_img = cv2.imread(prev_img_path)
    if prev_img is None:
        raise FileNotFoundError(f"Could not load image: {prev_img_path}")

    img_height, img_width = prev_img.shape[:2]
    image_center = (img_width / 2.0, img_height / 2.0)

    # Calculate crop centers in full image pixel coordinates
    crop_center_prev = (img_width / 2.0, conf.overlap_height_pixels / 2.0)
    crop_center_curr = (img_width / 2.0, img_height - (conf.overlap_height_pixels / 2.0))

    # --- 2) Loop through successive images ---
    for i in range(1, len(passage_df)):
        curr_row = passage_df.iloc[i]
        logger.info(f"Processing image {curr_row['Image']}")
        curr_img_path = images_dir / curr_row['Image']
        curr_img = cv2.imread(curr_img_path)

        if curr_img is None:
            logger.warning(f"Warning: Could not load {curr_img_path}. Skipping.")
            continue

        prev_state = optimized_positions[i - 1]

        curr_gps_state = {
            'X': curr_row['X'],
            'Y': curr_row['Y'],
            'Theta': curr_row['Theta_rad']
        }

        # --- 3) Windowing the images on the approximate overlap area ---
        # Assuming vehicle moves forward:
        # The content at the TOP of the previous image is seen at the BOTTOM of the current image.
        # In numpy (OpenCV), row 0 is the top of the image.

        # Top of prev_img
        crop_prev = prev_img[0: conf.overlap_height_pixels, :]

        # Bottom of curr_img
        crop_curr = curr_img[img_height - conf.overlap_height_pixels: img_height, :]

        # 4) Compute Phase Correlation (Rotation sweep)
        dx, dy, dtheta, response = compute_phase_correlation_sweep(crop_prev, crop_curr, conf)

        # Define a threshold to accept the match (e.g., 0.05 or 0.1 depending on texture)
        vision_target = None
        if response > conf.response_thr:
            # Convert the local dx, dy, dtheta into global X, Y, Theta targets for the optimizer
            vision_target = {
                'dx': dx, 'dy': dy, 'dtheta': dtheta, 'weight': response
            }
        else:
            logger.warning(f"Vision match failed for image {i + 1} (Response: {response:.4f}). Relying on Kinematics/GPS.")

        # 5) Scipy Optimization
        opt_X, opt_Y, opt_Theta = compute_step_optimization(
            prev_state=prev_state,
            curr_gps_state=curr_gps_state,
            vision_target=vision_target,
            crop_center_prev=crop_center_prev,
            crop_center_curr=crop_center_curr,
            image_center=image_center,
            meters_per_pixel=conf.span_width/img_width,
            vehicle_step=conf.vehicle_step,
            weight_gps=conf.weight_gps,
            weight_vision=conf.weight_vision,
            weight_kinematic=conf.weight_kinematic
        )

        optimized_positions.append({
            'Image': curr_row['Image'],
            'X': opt_X,
            'Y': opt_Y,
            'Theta': opt_Theta
        })

        # Update prev_img for the next iteration to avoid reloading
        prev_img = curr_img

    return pd.DataFrame(optimized_positions)


def create_geotiffs(csv_path, images_dir, output_dir, logger, span_width, scale_factor, epsg):
    """
    Converts images to georeferenced TIFFs using the optimized CSV coordinates.
    """
    df = pd.read_csv(csv_path)

    tiff_list = []

    for idx, row in df.iterrows():
        img_name = row['Image']
        img_path = images_dir / img_name
        logger.info(f'Creating GeoTIFF for image {img_name}')

        # Load and compress the image
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning(f"Skipping {img_name}, could not load.")
            continue

        # Downscale
        new_w = int(img.shape[1] * scale_factor)
        new_h = int(img.shape[0] * scale_factor)
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Convert BGR (OpenCV) to RGB (Rasterio expects RGB)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

        # Rasterio expects bands first: (Bands, Height, Width)
        img_bands = np.transpose(img_rgb, (2, 0, 1))

        # --- Calculate Affine Transform ---
        X, Y, Theta = row['X'], row['Y'], row['Theta']

        # Meters per pixel (on the compressed image)
        mpp = span_width / new_w

        # Convert Theta from radians to degrees
        theta_deg = np.degrees(Theta)

        corrected_angle = theta_deg - 90

        # Chain the transforms: Translate to GPS -> Rotate -> Scale (Flip Y) -> Center image
        transform = (
                Affine.translation(X, Y) *
                Affine.rotation(corrected_angle) *
                Affine.scale(mpp, -mpp) *
                Affine.translation(-new_w / 2, -new_h / 2)
        )

        # Save to GeoTIFF
        out_tiff = output_dir / f"{Path(img_name).stem}.tif"

        with rasterio.open(
                out_tiff, 'w',
                driver='GTiff',
                height=new_h,
                width=new_w,
                count=3,  # 3 bands for RGB
                dtype=img_bands.dtype,
                crs=f'EPSG:{epsg}',
                transform=transform,
                nodata=0  # Makes absolute black transparent if needed
        ) as dst:
            dst.write(img_bands)

        tiff_list.append(str(out_tiff))


def main():
    logging_conf()
    logger = logging.getLogger('Mosaic')
    conf = get_conf(logger)

    out_dir = Path(conf.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    input_dir = Path(conf.input_dir)

    # Iterate through subdirectories
    for passage_dir in input_dir.iterdir():
        if passage_dir.is_dir():
            logger.info(f'Processing passage {passage_dir.name}')
            df_processed = process_passage(passage_dir, conf, logger)
            if df_processed is not None:
                out_passage_dir = out_dir / passage_dir.name
                out_passage_dir.mkdir(parents=True, exist_ok=True)
                df_optimized = compute_passage_mosaic(df_processed, passage_dir, conf, logger)
                # Save to CSV
                out_csv = out_passage_dir / f"{passage_dir.name}.csv"
                df_optimized.to_csv(out_csv, index=False)
                create_geotiffs(out_csv, passage_dir, out_passage_dir, logger,
                                conf.span_width, conf.scale_factor, conf.epsg)
                break


if __name__ == '__main__':
    main()
