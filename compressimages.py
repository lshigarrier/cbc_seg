import logging
import cv2
import concurrent.futures
from pathlib import Path

from utils import get_conf, logging_conf


def parse_uge_gps(folder_path):
    """
    Parses the 'Images.txt' GPS format for a given folder.
    Returns a dictionary mapping the original image name to a tuple: (lon, lat, alt).
    """
    gps_file = folder_path / "Images.txt"
    gps_mapping = {}

    if not gps_file.exists():
        return gps_mapping

    with gps_file.open('r') as f:
        lines = f.readlines()

    # Skip header
    for line in lines[1:]:
        parts = line.strip().split()
        if len(parts) >= 5:
            lat = parts[2]
            lon = parts[3]
            img_name = parts[4]
            # WebODM requires (X/Longitude, Y/Latitude, Z/Altitude). Dummy altitude = 50.0.
            gps_mapping[img_name] = (lon, lat, "50.0")

    return gps_mapping


def process_image(img_path, conf, new_name, coords, logger):
    """
    Reads, resizes, and saves a single image.
    Returns the formatted string for geo.txt if successful.
    """
    try:
        out_path = conf.output_dir / new_name

        # 1. Read and Compress Image
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning(f"Warning: OpenCV could not read {img_path}")
            return None

        new_width = int(img.shape[1] * conf.scale_factor)
        new_height = int(img.shape[0] * conf.scale_factor)
        resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), conf.jpeg_quality]
        cv2.imwrite(str(out_path), resized_img, encode_params)

        # 2. Return the geo.txt line
        lon, lat, alt = coords
        return f"{new_name} {lon} {lat} {alt}\n"

    except Exception as e:
        logger.error(f"Error processing {img_path}: {e}")
        return None


def main():
    logging_conf()
    logger = logging.getLogger('Compress')
    conf = get_conf(logger)

    conf.input_dir = Path(conf.input_dir)
    conf.output_dir = Path(conf.output_dir)
    conf.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Find all images and group them by their parent folder
    image_paths = sorted(Path(conf.input_dir).rglob('*.jpg'))

    folders_dict = {}
    for img_path in image_paths:
        folders_dict.setdefault(img_path.parent, []).append(img_path)

    logger.info(f"Found {len(image_paths)} images across {len(folders_dict)} folders.")

    # 2. Prepare the exact tasks for the thread pool
    tasks = []

    # Select the correct GPS file parser
    parser_func = parse_uge_gps

    for folder, imgs in folders_dict.items():
        # Parse GPS file exactly once per folder
        gps_mapping = parser_func(folder)

        for img_path in imgs:
            if img_path.name not in gps_mapping:
                logger.warning(f"No GPS data found for {img_path.name}. Skipping.")
                continue

            # Create the unique flat filename: PassageFolder_ImageName.jpg
            new_name = f"{folder.name}_{img_path.name}"
            coords = gps_mapping[img_path.name]

            # Queue up the task parameters
            tasks.append((img_path, conf, new_name, coords, logger))

    logger.info(f"Starting compression for {len(tasks)} images...")

    # 3. Execute tasks concurrently
    geo_lines = ["EPSG:4326\n"]  # WebODM header

    with concurrent.futures.ThreadPoolExecutor(max_workers=conf.thread_workers) as executor:
        # Submit all tasks
        futures = [executor.submit(process_image, *task_args) for task_args in tasks]

        # Gather results as they finish
        for future in concurrent.futures.as_completed(futures):
            geo_line = future.result()
            if geo_line is not None:
                geo_lines.append(geo_line)

    # 4. Write the final unified geo.txt file
    geo_txt_path = conf.output_dir / "geo.txt"
    with geo_txt_path.open('w') as f:
        f.writelines(geo_lines)

    logger.info(f"Processing complete. Wrote {len(geo_lines) - 1} GPS entries to geo.txt.")


if __name__ == '__main__':
    main()
