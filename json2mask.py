import logging
import json
import numpy as np
import cv2
import concurrent.futures
from pathlib import Path

from utils import get_conf, logging_conf


def get_priority(shape, priority_list):
    label = shape['label']
    return priority_list.index(label)


def process_json_to_mask(conf, json_path, logger):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    img_height = data.get('imageHeight')
    img_width = data.get('imageWidth')

    if not img_height or not img_width:
        logger.warning(f"Skipping {json_path}: Missing imageHeight/Width")
        return

    # Initialize a blank mask (Background = 0)
    final_mask = np.zeros((img_height, img_width), dtype=np.uint8)

    # Sort shapes based on priority list
    shapes = data.get('shapes', [])
    shapes.sort(key=lambda s: get_priority(s, conf.priority_list))

    for shape in shapes:
        label = shape['label']
        points = np.array(shape['points'], dtype=np.int32)

        # Get the boundary settings for this specific class
        b_settings = conf.boundary_settings.get(label, conf.boundary_settings.get("default"))
        out_pixels = b_settings["outside"]
        in_pixels = b_settings["inside"]

        # Create a temporary blank canvas for this specific polygon
        temp_canvas = np.zeros((img_height, img_width), dtype=np.uint8)
        cv2.fillPoly(temp_canvas, [points], 1)

        # 1. Calculate Outside Area (Dilate)
        if out_pixels > 0:
            # Kernel size: 2 * radius + 1 ensures the center anchor is exact
            k_out = np.ones((2 * out_pixels + 1, 2 * out_pixels + 1), np.uint8)
            dilated = cv2.dilate(temp_canvas, k_out, iterations=1)
        else:
            dilated = temp_canvas.copy()  # No outside halo

        # 2. Calculate Inside Area (Erode)
        if in_pixels > 0:
            k_in = np.ones((2 * in_pixels + 1, 2 * in_pixels + 1), np.uint8)
            eroded = cv2.erode(temp_canvas, k_in, iterations=1)
        else:
            eroded = temp_canvas.copy()  # No inside halo

        # 3. Draw the Masks
        # First, apply the ignore_index to the entire dilated area (Core + Outside + Inside boundary)
        final_mask[dilated == 1] = conf.ignore_index

        # Second, overwrite the shrunken core with the actual class ID
        # Because it's eroded, it leaves the ignore_index behind on the inner boundary!
        final_mask[eroded == 1] = conf.class_mapping[label]

    # Save the mask
    out_path = json_path.with_name(f"{json_path.stem}_mask.png")
    cv2.imwrite(out_path, final_mask)


def main():
    logging_conf()
    logger = logging.getLogger('J2M')
    conf = get_conf(logger)

    # Find all JSON files recursively
    json_files = list(Path(conf.data_dir).rglob("*.json"))

    logger.info(f"Found {len(json_files)} JSON files. Starting conversion...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=conf.thread_workers) as executor:
        futures = [executor.submit(process_json_to_mask, conf, json_path, logger) for json_path in json_files]

        # Wait for tasks to complete and catch any potential errors
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logger.error(f"An error occurred during processing: {e}")

    logger.info("Done converting annotations to masks!")


if __name__ == "__main__":
    main()
