"""
Test script for CBCPIDNetL Inference Module.

This script loads a set of images from a directory and runs inference
using 5 different input formats (path, numpy single, tensor single,
numpy batch, tensor batch). It overlays the predicted segmentation masks
onto the original images and saves the results in designated subdirectories.

Inputs:
    - input_dir: Path to the directory containing input images.
    - output_dir: Path to the directory where results will be saved.
    - model_path: Path to the pre-trained TorchScript model.

Outputs:
    - Generates 5 subdirectories inside output_dir containing the blended
      images (original image + semi-transparent mask overlay).

Example command:
    python test_cbc_pidnet_large.py
"""

from pathlib import Path
from typing import List

import time
import cv2
import numpy as np
import torch

from cbc_pidnet_large import CBCPIDNetL


class CustomTimer:

    def __init__(self):
        self.start_perf_count = None

    def start(self):
        self.start_perf_count = time.perf_counter()

    def stop(self, len_dataset):
        elapsed_perf_count = time.perf_counter() - self.start_perf_count
        print(f"Perf counter: {elapsed_perf_count:.2f} s")
        print(f"  Time per image: {elapsed_perf_count / len_dataset * 1000:.2f} ms")


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


def save_result(output_path: Path, image: np.ndarray, mask: torch.Tensor):
    """Blends the mask with the image and saves it to the disk."""
    mask_np = mask.cpu().numpy().astype(np.uint8)
    blended = overlay_mask(image, mask_np)

    # Convert RGB back to BGR for OpenCV saving
    blended_bgr = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), blended_bgr)


def load_image_rgb(path: Path) -> np.ndarray:
    """Loads an image from disk and converts it to RGB."""
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def test_path_single(model: torch.nn.Module, image_paths: List[Path], output_dir: Path):
    """Variant 1: Inference using file paths one by one."""
    out_sub_dir = output_dir / "1_path_single"
    out_sub_dir.mkdir(parents=True, exist_ok=True)

    for path in image_paths:
        mask = model(path, return_logits=False)
        orig_img = load_image_rgb(path)
        save_result(out_sub_dir / path.name, orig_img, mask)


def test_numpy_single(model: torch.nn.Module, image_paths: List[Path], output_dir: Path):
    """Variant 2: Inference using numpy arrays one by one (batch_size=1)."""
    out_sub_dir = output_dir / "2_numpy_single"
    out_sub_dir.mkdir(parents=True, exist_ok=True)

    for path in image_paths:
        img_np = load_image_rgb(path)
        # Passing numpy array (H, W, 3)
        mask = model(img_np, is_bgr=False, return_logits=False)
        save_result(out_sub_dir / path.name, img_np, mask)


def test_tensor_single(model: torch.nn.Module, image_paths: List[Path], output_dir: Path):
    """Variant 3: Inference using torch tensors one by one (batch_size=1)."""
    out_sub_dir = output_dir / "3_tensor_single"
    out_sub_dir.mkdir(parents=True, exist_ok=True)

    for path in image_paths:
        img_np = load_image_rgb(path)
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)  # (3, H, W)

        mask = model(img_tensor, return_logits=False)
        save_result(out_sub_dir / path.name, img_np, mask)


def test_numpy_batch(model: torch.nn.Module, image_paths: List[Path], output_dir: Path, batch_size: int = 2):
    """Variant 4: Inference using numpy arrays in batches (batch_size=2)."""
    out_sub_dir = output_dir / "4_numpy_batch"
    out_sub_dir.mkdir(parents=True, exist_ok=True)

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]

        # Load and stack into (B, H, W, 3)
        orig_images = [load_image_rgb(p) for p in batch_paths]
        batch_np = np.stack(orig_images, axis=0)

        masks = model(batch_np, is_bgr=False, return_logits=False)

        # Masks shape should be (B, H, W)
        for j, path in enumerate(batch_paths):
            save_result(out_sub_dir / path.name, orig_images[j], masks[j])


def test_tensor_batch(model: torch.nn.Module, image_paths: List[Path], output_dir: Path, batch_size: int = 2):
    """Variant 5: Inference using torch tensors in batches (batch_size=2)."""
    out_sub_dir = output_dir / "5_tensor_batch"
    out_sub_dir.mkdir(parents=True, exist_ok=True)

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]

        orig_images = [load_image_rgb(p) for p in batch_paths]
        # Create tensor (B, 3, H, W)
        tensors = [torch.from_numpy(img).permute(2, 0, 1) for img in orig_images]
        batch_tensor = torch.stack(tensors, dim=0)

        masks = model(batch_tensor, return_logits=False)

        for j, path in enumerate(batch_paths):
            save_result(out_sub_dir / path.name, orig_images[j], masks[j])


def main():
    model_path = Path("C:/Users/Admin/Code/cbc_seg/cbc-pidnet-large/model/cbc-pidnet-large.pt")
    input_dir = Path("C:/Users/Admin/Data/test_images")
    output_dir = Path("C:/Users/Admin/Code/cbc_seg/cbc-pidnet-large/test_output")

    # Gather all image paths
    image_paths = [p for p in input_dir.iterdir() if p.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    if not image_paths:
        print(f"No images found in {input_dir}")
        return

    # Sort for consistent batching
    image_paths.sort()

    print(f"Found {len(image_paths)} images. Loading model...")
    model = CBCPIDNetL(
        model_path=model_path,
        patch_size=1024,
        patch_per_row=7,
        patch_per_col=3,
        patch_overlap=0.5,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    timer = CustomTimer()

    print("Running Variant 1: Path Single")
    timer.start()
    test_path_single(model, image_paths, output_dir)
    timer.stop(len_dataset=len(image_paths))

    print("Running Variant 2: Numpy Single")
    timer.start()
    test_numpy_single(model, image_paths, output_dir)
    timer.stop(len_dataset=len(image_paths))

    print("Running Variant 3: Tensor Single")
    timer.start()
    test_tensor_single(model, image_paths, output_dir)
    timer.stop(len_dataset=len(image_paths))

    print("Running Variant 4: Numpy Batch (size=2)")
    timer.start()
    test_numpy_batch(model, image_paths, output_dir, batch_size=2)
    timer.stop(len_dataset=len(image_paths))

    print("Running Variant 5: Tensor Batch (size=2)")
    timer.start()
    test_tensor_batch(model, image_paths, output_dir, batch_size=2)
    timer.stop(len_dataset=len(image_paths))

    peak_memory_gb = torch.cuda.max_memory_allocated() / 1024 ** 3
    print(f'Peak GPU memory allocated: {peak_memory_gb:.2f} GB')

    print("All tests completed successfully.")


if __name__ == "__main__":
    main()
