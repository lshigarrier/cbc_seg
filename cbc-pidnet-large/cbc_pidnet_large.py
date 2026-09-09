"""
End-to-end Inference Module for PIDNet.

This module encapsulates the pre-processing (cropping), inference,
and post-processing (stitching with cosine blending) of images
using a pre-trained ONNX model. It supports both single images
and batches of images.

Inputs:
    - image: Can be a file path (str or pathlib.Path) or a numpy array.
             If a numpy array is provided, it can be a single image or a batch.
    - is_bgr: Boolean indicating if the numpy array is in BGR format (default: False).
    - return_logits: Boolean indicating whether to return the raw logits instead of the argmax mask (default: False).

Outputs:
    - A numpy array containing either the segmentation mask (if return_logits=False)
      or the raw logits (if return_logits=True).
"""

import math
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


class CBCPIDNetL:
    def __init__(
            self,
            model_path: str | Path,
            patch_size: int = 1024,
            patch_per_row: int = 7,
            patch_per_col: int = 3,
            patch_overlap: float = 0.5,
            device: str = 'cpu'
    ):
        super().__init__()
        self.patch_size = patch_size
        self.patch_per_row = patch_per_row
        self.patch_per_col = patch_per_col
        self.patch_overlap = patch_overlap

        # Configure ONNX Runtime execution providers
        providers = ['CPUExecutionProvider']
        if device == 'cuda' or device == 'gpu':
            # ONNX Runtime will fallback to CPU if CUDA is not available/installed properly
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

        # Load the ONNX model
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        # Dynamically get the input name for the ONNX graph
        self.input_name = self.session.get_inputs()[0].name

        # ImageNet normalization parameters
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

    def _prepare_image(self, image: str | Path | np.ndarray, is_bgr: bool) -> np.ndarray:
        """Converts various input types to a standardized Numpy batch array (B, C, H, W)."""
        if isinstance(image, (str, Path)):
            image_np = cv2.imread(str(image), cv2.IMREAD_COLOR)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            image_array = np.transpose(image_np, (2, 0, 1))[np.newaxis, ...]  # (1, 3, H, W)
        elif isinstance(image, np.ndarray):
            if is_bgr:
                image = image[..., ::-1].copy()
            image_array = image.copy()
            if image_array.ndim == 3:  # (H, W, C)
                image_array = np.transpose(image_array, (2, 0, 1))[np.newaxis, ...]  # (1, 3, H, W)
            elif image_array.ndim == 4:  # (B, H, W, C)
                image_array = np.transpose(image_array, (0, 3, 1, 2))  # (B, 3, H, W)
            else:
                raise ValueError("Numpy array must be 3D or 4D")
        else:
            raise NotImplementedError("Input must be a file path or a Numpy array.")

        if image_array.dtype == np.uint8:
            image_array = image_array.astype(np.float32) / 255.0

            # Apply test transform (ImageNet normalization)
        image_array = (image_array - self.mean) / self.std

        return image_array

    def _extract_patches(self, img_array: np.ndarray):
        """Crops the batch into overlapping patches and resizes them to patch_size."""
        B, C, H, W = img_array.shape
        Pw = W / (1 + (self.patch_per_row - 1) * (1 - self.patch_overlap)) if self.patch_per_row > 1 else W
        Ph = H / (1 + (self.patch_per_col - 1) * (1 - self.patch_overlap)) if self.patch_per_col > 1 else H
        num_patches = self.patch_per_col * self.patch_per_row

        patches_array = np.empty((num_patches, B, C, self.patch_size, self.patch_size), dtype=np.float32)
        boxes = []
        p_idx = 0

        for i in range(self.patch_per_col):
            for j in range(self.patch_per_row):
                y1 = int(i * Ph * (1 - self.patch_overlap))
                x1 = int(j * Pw * (1 - self.patch_overlap))

                y2 = min(int(y1 + Ph) if i < self.patch_per_col - 1 else H, H)
                x2 = min(int(x1 + Pw) if j < self.patch_per_row - 1 else W, W)

                h, w = y2 - y1, x2 - x1

                patches_array[p_idx] = np.transpose(
                    cv2.resize(
                        np.transpose(img_array[:, :, y1:y2, x1:x2], (2, 3, 0, 1)).reshape(h, w, B * C),
                        (self.patch_size, self.patch_size),
                        interpolation=cv2.INTER_CUBIC
                    ).reshape(self.patch_size, self.patch_size, B, C),
                    (2, 3, 0, 1)
                )

                boxes.append((y1, y2, x1, x2))
                p_idx += 1

        # Transpose to (B, num_patches, C, P, P) and flatten to (B * num_patches, C, P, P)
        patches_flat = np.transpose(patches_array, (1, 0, 2, 3, 4)).reshape(
            B * num_patches, C, self.patch_size, self.patch_size
        )

        return patches_flat, boxes, H, W, B

    @staticmethod
    def _stitch_logits(patch_logits: np.ndarray, boxes: list, H: int, W: int, B:int) -> np.ndarray:
        """Reconstructs the full image logits from the patches using 2D cosine/hann window blending for a batch."""
        _, C, H_out, W_out = patch_logits.shape
        num_patches = len(boxes)

        # Reshape logits back to (B, num_patches, C, H_out, W_out)
        patch_logits = patch_logits.reshape(B, num_patches, C, H_out, W_out)

        img_acc = np.zeros((B, C, H, W), dtype=np.float32)
        weight_acc = np.zeros((1, 1, H, W), dtype=np.float32)

        # Cache for cosine windows to avoid redundant heavy math operations on CPU
        window_cache = {}

        for p_idx in range(num_patches):
            y1, y2, x1, x2 = boxes[p_idx]
            h_box, w_box = y2 - y1, x2 - x1

            # Get or compute cosine window
            if (h_box, w_box) not in window_cache:
                wy = np.cos(np.linspace(-math.pi / 2, math.pi / 2, h_box, dtype=np.float32))
                wx = np.cos(np.linspace(-math.pi / 2, math.pi / 2, w_box, dtype=np.float32))
                window_cache[(h_box, w_box)] = (np.outer(wy, wx)[np.newaxis, np.newaxis, :, :] + 1e-5)

            img_acc[:, :, y1:y2, x1:x2] += np.transpose(
                cv2.resize(
                    np.transpose(patch_logits[:, p_idx], (2, 3, 0, 1)).reshape(H_out, W_out, B * C),
                    (w_box, h_box),
                    interpolation=cv2.INTER_CUBIC
                ).reshape(h_box, w_box, B, C),
                (2, 3, 0, 1)
            ) * window_cache[(h_box, w_box)]

            weight_acc[:, :, y1:y2, x1:x2] += window_cache[(h_box, w_box)]

        img_acc /= weight_acc
        return img_acc

    def __call__(
            self,
            image: str | Path | np.ndarray,
            is_bgr: bool = False,
            return_logits: bool = False
    ) -> np.ndarray:

        # 1. Pre-process input
        img_array = self._prepare_image(image, is_bgr)

        # 2. Crop into patches
        patches, boxes, H, W, B = self._extract_patches(img_array)

        # 3. Inference using ONNX Runtime
        # The input must be a float32 contiguous numpy array (ensure it is contiguous)
        patches = np.ascontiguousarray(patches, dtype=np.float32)
        patch_logits = self.session.run(None, {self.input_name: patches})[0]

        # 4. Post-process (Stitching)
        full_logits = self._stitch_logits(patch_logits, boxes, H, W, B)

        # 5. Output
        # Squeeze batch dimension if input was not a batch initially, keeping it intuitive
        argmax_dim = 1
        if full_logits.shape[0] == 1 and not (isinstance(image, np.ndarray) and image.ndim == 4):
            full_logits = full_logits.squeeze(0)
            argmax_dim = 0
        if return_logits:
            return full_logits
        else:
            # Mask generation over batch
            mask = np.argmax(full_logits, axis=argmax_dim)
            return mask
