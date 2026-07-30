import pandas as pd
import numpy as np
import scipy.interpolate as si
import logging
import pyproj
from pathlib import Path

from utils import logging_conf, get_conf, CustomTimer
from utils_positioning import process_detections, generate_global_cog


def read_uge_gps(file_path):
    """
    Load the Images.txt file into a pandas DataFrame.
    Returns a DataFrame strictly with columns: ['Image', 'Latitude', 'Longitude']
    """
    df = pd.read_csv(file_path, sep='\t')
    required_columns = ['Image', 'Latitude', 'Longitude']
    return df[required_columns]


def read_xlsx_gps(file_path):
    """
    Load the Image.xlsx file into a pandas DataFrame.
    Returns a DataFrame strictly with columns: ['Image', 'Latitude', 'Longitude']
    """
    df = pd.read_excel(file_path)
    required_columns = ['Image', 'Latitude', 'Longitude']
    return df[required_columns]


def project_coordinates(conf, df):
    """
    Project GPS coordinates (Latitude/Longitude) to local coordinates (X/Y).
    """
    transformer = pyproj.Transformer.from_crs(conf.crs_gps, conf.crs_projected, always_xy=True)
    df['X_proj'], df['Y_proj'] = transformer.transform(df['Longitude'].values, df['Latitude'].values)
    return df


def compute_heading(dx, dy):
    """
    Computation of the heading
    """
    hdt = np.arctan2(dy, dx)
    # Normalization of the heading between 0 and 2*pi
    return hdt % (2 * np.pi)


def log_image_step(x, y, logger):
    logger.info(f"  Image step mean: {np.mean(np.hypot(np.diff(x), np.diff(y))):.3f} m")
    logger.info(f"  Image step std: {np.std(np.hypot(np.diff(x), np.diff(y))):.3f} m")


def smooth_and_resample_trajectory(conf, x_proj, y_proj, num_images, logger):
    """
    Smooth the trajectory using a B-Spline.
    Resample uniformly and compute the orientation using the tangent vector (first derivative).
    """
    # Remove consecutive duplicates (vehicle not moving)
    # Keep the first point, then keep only the points whose distance to the previous one is > 0
    diffs = np.hypot(np.diff(x_proj), np.diff(y_proj))
    keep = np.insert(diffs > 1e-3, 0, True)  # Tolerance: 1 mm
    x_proj = x_proj[keep]
    y_proj = y_proj[keep]

    # Compute the parametric spline
    smoothing_factor = len(x_proj) * conf.gps_error_variance
    result = si.splprep([x_proj, y_proj], s=smoothing_factor)
    tck = result[0]

    if conf.method == 'step' or conf.method == 'range':
        # Arc length parameterization
        # Dense sampling of the curve to compute the arc length
        u_dense = np.linspace(0, 1, 150 * num_images)
        x_dense, y_dense = si.splev(u_dense, tck)

        # Euclidean distance between successive dense points and cumulative distance
        dx_dense = np.diff(x_dense)
        dy_dense = np.diff(y_dense)
        distances = np.hypot(dx_dense, dy_dense)
        cum_length = np.insert(np.cumsum(distances), 0, 0.0)

        if conf.method == 'step':
            # Creation of target distances based strictly on the vehicle step
            target_lengths = np.arange(num_images) * conf.vehicle_step
            # Interpolation to find the parameters 'u' corresponding to target lengths
            # The argument fill_value="extrapolate" allows to extend the curve if the theoretical distance exceeds the real distance measured by the GPS.
            interp_func = si.interp1d(cum_length, u_dense, fill_value="extrapolate")
            u_target = interp_func(target_lengths)

        else:  # conf.method == 'range'
            total_length = cum_length[-1]
            # Uniform distribution over the entire available length
            target_lengths = np.linspace(0, total_length, num_images)
            # Creation of the vector of uniformly spaced parameters (no need to extrapolate here)
            # This ensures a distribution such that step = total_length / (num_images - 1)
            interp_func = si.interp1d(cum_length, u_dense)
            u_target = interp_func(target_lengths)

    else:
        raise NotImplementedError

    # Final evaluation of smoothed positions and of the derivative
    x_smooth, y_smooth = si.splev(u_target, tck, der=0)
    log_image_step(x_smooth, y_smooth, logger)
    dx, dy = si.splev(u_target, tck, der=1)
    hdt = compute_heading(dx, dy)

    return x_smooth, y_smooth, hdt


def apply_camera_offset(conf, x_smooth, y_smooth, hdt):
    """
    Apply the spatial offset to get the real position of the camera.
    """
    # Standard orientation: forward vector is (cos(hdt), sin(hdt))
    # cam_offset_y is assumed to be forward (longitudinal), cam_offset_x is right (lateral)
    x_corr = x_smooth + (conf.cam_offset_y * np.cos(hdt)) + (conf.cam_offset_x * np.sin(hdt))
    y_corr = y_smooth + (conf.cam_offset_y * np.sin(hdt)) - (conf.cam_offset_x * np.cos(hdt))

    return x_corr, y_corr


def process_passage(conf, passage_dir, logger):
    """
    Process an entire passage directory : reading, projection, smoothing, and computation of corrections.
    """
    if (passage_dir / 'Images.txt').exists():
        read_gps = read_uge_gps
        data_path = passage_dir / 'Images.txt'
    elif (passage_dir / "Image.xlsx").exists():
        read_gps = read_xlsx_gps
        data_path = passage_dir / "Image.xlsx"
    else:
        raise NotImplementedError(f"  No GPS file found in {passage_dir}")

    # Loading GPS data
    df = read_gps(data_path)

    # Projection in cartesian coordinates
    df = project_coordinates(conf, df)

    if conf.apply_correction:
        # Smoothing, resampling and computation of the headings
        num_images = len(df)
        x_smooth, y_smooth, df['HDT_corr'] = (smooth_and_resample_trajectory(
            conf,
            df['X_proj'].values,
            df['Y_proj'].values,
            num_images,
            logger
        ))
        # Apply the camera offset with respect to the GPS receiver
        df['X_corr'], df['Y_corr'] = apply_camera_offset(conf, x_smooth, y_smooth, df['HDT_corr'].values)

    else:
        df['X_corr'], df['Y_corr'] = df['X_proj'].copy(), df['Y_proj'].copy()
        log_image_step(df['X_corr'].values, df['Y_corr'].values, logger)
        dx = df['X_corr'].shift(-1) - df['X_corr']
        dy = df['Y_corr'].shift(-1) - df['Y_corr']
        df['HDT_corr'] = compute_heading(dx, dy)

    return df


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
        passage_name = passage_dir.name
        out_csv = out_dir / f"{passage_name}.csv"

        if conf.read_csv and out_csv.exists():
            logger.info(f"  CSV found for {passage_name}")
            df_coords = pd.read_csv(out_csv)
        else:
            logger.info(f"  Processing passage {passage_name}")
            df_coords = process_passage(conf, passage_dir, logger)
            if df_coords is None or df_coords.empty:
                logger.error(f"  Failed to process passage {passage_name}")
                continue
            df_coords.to_csv(out_csv, index=False)

        all_passage_data.append((df_coords, passage_dir))

    if conf.generate_detections:
        process_detections(conf, all_passage_data, out_dir, logger)

    if conf.generate_cog:
        generate_global_cog(conf, all_passage_data, out_dir, logger)

    timer.stop(logger, show_time_per_image=False)

if __name__ == '__main__':
    main()
