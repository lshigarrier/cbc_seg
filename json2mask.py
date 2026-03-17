import logging
import json
import numpy as np
import cv2
from pathlib import Path

from utils import get_conf


def get_priority(shape, priority_list):
    label = shape['label']
    return priority_list.index(label)


def process_json_to_mask(conf, json_path, logger):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    img_height = data.get('imageHeight')
    img_width = data.get('imageWidth')

    if not img_height or not img_width:
        logger.info(f"Skipping {json_path}: Missing imageHeight/Width")
        return

    # Initialize a blank mask (Background = 0)
    final_mask = np.zeros((img_height, img_width), dtype=np.uint8)

    # Sort shapes based on priority list
    shapes = data.get('shapes', [])
    shapes.sort(key=lambda s: get_priority(s, conf.priority_list))

    kernel = np.ones((conf.dilation_kernel_size, conf.dilation_kernel_size), np.uint8)

    for shape in shapes:
        points = np.array(shape['points'], dtype=np.int32)

        # Create a temporary blank canvas for this specific polygon
        temp_canvas = np.zeros((img_height, img_width), dtype=np.uint8)
        cv2.fillPoly(temp_canvas, [points], 1)

        # 1. Create the Soft Boundary (Halo)
        # Dilate the polygon to make it thicker
        dilated_canvas = cv2.dilate(temp_canvas, kernel, iterations=1)

        # Apply the 255 halo to the final mask where the dilated canvas is 1
        final_mask[dilated_canvas == 1] = conf.ignore_index

        # 2. Draw the actual class core
        # Overwrite the exact polygon area with the actual class ID
        label = shape['label']
        final_mask[temp_canvas == 1] = conf.class_mapping[label]

    # Save the mask
    out_path = json_path.with_name(f"{json_path.stem}_mask.png")
    cv2.imwrite(out_path, final_mask)


def main():
    logger = logging.getLogger('J2M')
    conf = get_conf(logger)

    # Find all JSON files recursively
    json_files = list(Path(conf.data_dir).rglob("*.json"))

    logger.info(f"Found {len(json_files)} JSON files.")

    for json_path in json_files:
        process_json_to_mask(conf, json_path, logger)

    logger.info("Done converting annotations to masks!")


if __name__ == "__main__":
    main()
