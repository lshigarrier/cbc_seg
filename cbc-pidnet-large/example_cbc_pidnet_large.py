"""
This script demonstrates how to run inference on a single image using the CBCPIDNetL model.
It instantiates the model class, processes a single image, generates the semantic segmentation mask,
and saves the blended overlay image to an output directory.

Inputs:
    - Path to the input image (input_image_path)
    - Path to the output directory (output_dir)
    - Path to the model weights (model_path)

Outputs:
    - Saves the overlay image (original image + semi-transparent mask) in the output directory.

Example command to run the script:
    python example_cbc_pidnet_large.py
"""

import time
import cv2
import torch
import numpy as np
from pathlib import Path

from cbc_pidnet_large import CBCPIDNetL


def generate_color_palette() -> np.ndarray:
    """Generates a color palette for mask visualization."""
    return np.array([
    [0, 0, 0],
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 255],
    [255, 255, 0],
    [255, 0, 255],
    [0, 255, 255],
    [255, 128, 0],
    [255, 0, 128],
    [0, 255, 128],
    [128, 255, 0],
    [128, 0, 255],
    [0, 128, 255],
    [128, 0, 0],
    [0, 128, 0],
    [0, 0, 128],
    [128, 128, 0],
    [128, 0, 128],
    [0, 128, 128]
    ], dtype=np.uint8)


def overlay_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Overlays a discrete mask onto an image with alpha blending.

    Args:
        image: Original RGB image array (H, W, 3).
        mask: 2D array of predicted class indices (H, W).
        alpha: Transparency factor.

    Returns:
        The blended RGB image.
    """
    palette = generate_color_palette()

    color_mask = palette[mask]
    blended = cv2.addWeighted(image, 1 - alpha, color_mask, alpha, 0)

    fg_mask = (mask != 0)[..., None]
    final_out = np.where(fg_mask, blended, image)

    return final_out


def main():
    input_image_path = Path("C:/Users/Admin/Data/test_images/NR_ADP_2514015.CA1_CA1_ADP_2514015_00000011220.jpg")
    output_dir = Path("C:/Users/Admin/Code/cbc_seg/cbc-pidnet-large/example_output")
    model_path = Path("C:/Users/Admin/Code/cbc_seg/cbc-pidnet-large/model/cbc-pidnet-large.pt")

    output_dir.mkdir(parents=True, exist_ok=True)

    start_perf_count = time.perf_counter()

    model = CBCPIDNetL(
        model_path=model_path,
        patch_size=1024,
        patch_per_row=7,
        patch_per_col=3,
        patch_overlap=0.5,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # Inference
    mask_tensor = model(input_image_path, is_bgr=False, return_logits=False)

    # Move the resulting mask to CPU and convert to numpy
    mask_np = mask_tensor.squeeze().cpu().numpy().astype(np.uint8)

    # Read original image using OpenCV (BGR) and convert to RGB
    orig_img_bgr = cv2.imread(str(input_image_path))
    orig_img_rgb = cv2.cvtColor(orig_img_bgr, cv2.COLOR_BGR2RGB)

    blended_rgb = overlay_mask(orig_img_rgb, mask_np, alpha=0.5)

    # Convert back to BGR for saving with OpenCV
    blended_bgr = cv2.cvtColor(blended_rgb, cv2.COLOR_RGB2BGR)

    # Save output
    output_path = output_dir / f"overlay_{input_image_path.name}"
    cv2.imwrite(str(output_path), blended_bgr)

    elapsed_perf_count = time.perf_counter() - start_perf_count
    print(f"Perf counter: {elapsed_perf_count:.4f} s")


if __name__ == "__main__":
    main()
