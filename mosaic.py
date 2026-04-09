import pandas as pd
import numpy as np
import logging
import cv2
from pathlib import Path
from pyproj import Transformer

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


def initialize_all_passages(conf, logger):
    """Iterate over all passages in the input directory and initialize their data."""
    out_dir = Path(conf.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    input_dir = Path(conf.input_dir)

    passage_data = {}

    # Iterate through subdirectories
    for passage_dir in input_dir.iterdir():
        if passage_dir.is_dir():
            df_processed = process_passage(passage_dir, conf, logger)
            if df_processed is not None:
                passage_data[passage_dir.name] = df_processed

    return passage_data


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


def compute_passage_mosaic(passage_df, images_dir, overlap_height_pixels, logger):
    """
    Main function to compute the mosaic for a single passage.

    Parameters:
    - passage_df: DataFrame containing ['Image', 'Latitude', 'Longitude', 'X', 'Y', 'Theta_rad']
    - images_dir: Path to the directory containing the images
    - overlap_height_pixels: Expected vertical overlap between two successive images in pixels
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

    # --- 2) Loop through successive images ---
    for i in range(1, len(passage_df)):
        curr_row = passage_df.iloc[i]
        curr_img_path = images_dir / curr_row['Image']
        curr_img = cv2.imread(curr_img_path)

        if curr_img is None:
            logger.warning(f"Warning: Could not load {curr_img_path}. Skipping.")
            continue

        prev_state = optimized_positions[i - 1]

        # --- 3) Windowing the images on the approximate overlap area ---
        # Assuming vehicle moves forward:
        # The content at the TOP of the previous image is seen at the BOTTOM of the current image.
        # In numpy (OpenCV), row 0 is the top of the image.

        # Top of prev_img
        crop_prev = prev_img[0: overlap_height_pixels, :]

        # Bottom of curr_img
        crop_curr = curr_img[img_height - overlap_height_pixels: img_height, :]

        # Apply the Hanning window to prepare for phase correlation
        windowed_prev = apply_2d_hanning(crop_prev)
        windowed_curr = apply_2d_hanning(crop_curr)

        # [NEXT STEPS TO BE ADDED HERE]
        # 4) Compute Phase Correlation (Rotation sweep)
        # 5) Scipy Optimization

        # Update prev_img for the next iteration to avoid reloading
        prev_img = curr_img

        # (Temporary placeholder to allow the loop to run if tested now)
        optimized_positions.append({
            'Image': curr_row['Image'],
            'X': curr_row['X'],
            'Y': curr_row['Y'],
            'Theta': prev_state['Theta_rad']
        })

    return pd.DataFrame(optimized_positions)


def main():
    logging_conf()
    logger = logging.getLogger('Mosaic')
    conf = get_conf(logger)
    passage_data = initialize_all_passages(conf, logger)
    for passage_name in passage_data:
        logger.info(passage_data[passage_name].head())


if __name__ == '__main__':
    main()
