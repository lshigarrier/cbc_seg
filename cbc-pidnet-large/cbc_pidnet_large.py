"""
End-to-end Inference Module for PIDNet.

This module encapsulates the pre-processing (cropping), inference,
and post-processing (stitching with cosine blending) of images
using a pre-trained TorchScript model. It supports both single images
and batches of images.

Inputs:
    - image: Can be a file path (str or pathlib.Path), a numpy array, or a torch.Tensor.
             If a tensor or numpy array is provided, it can be a single image or a batch.
    - is_bgr: Boolean indicating if the numpy array is in BGR format (default: False).
    - return_logits: Boolean indicating whether to return the raw logits instead of the argmax mask (default: False).

Outputs:
    - A torch.Tensor containing either the segmentation mask (if return_logits=False)
      or the raw logits (if return_logits=True). The output is on the same device as the model.

Example command:
    import torch
    from pathlib import Path
    from cbc_pidnet_large import CBCPIDNetL

    model_path = Path('models/cbc-pidnet-large.pt')
    model = CBCPIDNetL(
        model_path=model_path,
        patch_size=1024,
        patch_per_row=7,
        patch_per_col=3,
        patch_overlap=0.5,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # Run inference directly on a batch tensor
    dummy_batch = torch.rand(4, 3, 2160, 4096)
    masks = model(dummy_batch, return_logits=False)
"""

import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CBCPIDNetL(nn.Module):
    def __init__(
            self,
            model_path: str | Path,
            patch_size: int = 1024,
            patch_per_row: int = 7,
            patch_per_col: int = 3,
            patch_overlap: float = 0.5,
            device: str | int | torch.device = 'cuda'
    ):
        super().__init__()
        self.patch_size = patch_size
        self.patch_per_row = patch_per_row
        self.patch_per_col = patch_per_col
        self.patch_overlap = patch_overlap
        self.device = torch.device(device)

        # ImageNet normalization parameters
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # Load the TorchScript model
        self.model = torch.jit.load(str(model_path), map_location=self.device)
        self.model.eval()

    def _prepare_image(self, image: str | Path | np.ndarray | torch.Tensor, is_bgr: bool) -> torch.Tensor:
        """Converts various input types to a standardized PyTorch batch tensor (B, C, H, W)."""
        if isinstance(image, (str, Path)):
            image_np = cv2.imread(str(image), cv2.IMREAD_COLOR)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)
        elif isinstance(image, np.ndarray):
            if is_bgr:
                image = image[..., ::-1].copy()
            image_tensor = torch.from_numpy(image)
            if image_tensor.ndim == 3:
                image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
            elif image_tensor.ndim == 4:
                image_tensor = image_tensor.permute(0, 3, 1, 2)
        elif isinstance(image, torch.Tensor):
            image_tensor = image.clone()
            if image_tensor.ndim == 3:
                image_tensor = image_tensor.unsqueeze(0)
        else:
            raise NotImplementedError

        image_tensor = image_tensor.to(self.device)

        if image_tensor.dtype == torch.uint8:
            image_tensor = image_tensor.float() / 255.0

        # Apply test transform (ImageNet normalization)
        image_tensor = (image_tensor - self.mean.to(self.device)) / self.std.to(self.device)

        return image_tensor

    def _extract_patches(self, img_tensor: torch.Tensor):
        """Crops the batch into overlapping patches and resizes them to patch_size."""
        B, C, H, W = img_tensor.shape
        Pw = W / (1 + (self.patch_per_row - 1) * (1 - self.patch_overlap)) if self.patch_per_row > 1 else W
        Ph = H / (1 + (self.patch_per_col - 1) * (1 - self.patch_overlap)) if self.patch_per_col > 1 else H

        patches = []
        boxes = []

        for i in range(self.patch_per_col):
            for j in range(self.patch_per_row):
                y1 = int(i * Ph * (1 - self.patch_overlap))
                x1 = int(j * Pw * (1 - self.patch_overlap))

                y2 = int(y1 + Ph) if i < self.patch_per_col - 1 else H
                x2 = int(x1 + Pw) if j < self.patch_per_row - 1 else W

                y2 = min(y2, H)
                x2 = min(x2, W)

                patch = img_tensor[:, :, y1:y2, x1:x2]  # (B, C, h, w)

                # Resize to target patch_size
                patch_resized = F.interpolate(
                    patch,
                    size=(self.patch_size, self.patch_size),
                    mode='bicubic',
                    align_corners=False
                )

                patches.append(patch_resized)
                boxes.append((y1, y2, x1, x2))

        # Stack patches -> (B, num_patches, C, P, P)
        patches_tensor = torch.stack(patches, dim=1)
        # Flatten for inference -> (B * num_patches, C, P, P)
        patches_flat = patches_tensor.reshape(B * len(boxes), C, self.patch_size, self.patch_size)

        return patches_flat, boxes, H, W, B

    def _stitch_logits(self, patch_logits: torch.Tensor, boxes: list, H: int, W: int, B:int) -> torch.Tensor:
        """Reconstructs the full image logits from the patches using 2D cosine/hann window blending for a batch."""
        _, C, H_out, W_out = patch_logits.shape
        num_patches = len(boxes)

        # Reshape logits back to (B, num_patches, C, H_out, W_out)
        patch_logits = patch_logits.view(B, num_patches, C, H_out, W_out)

        img_acc = torch.zeros((B, C, H, W), device=self.device, dtype=torch.float32)
        weight_acc = torch.zeros((1, 1, H, W), device=self.device, dtype=torch.float32)

        for p_idx in range(num_patches):
            y1, y2, x1, x2 = boxes[p_idx]
            h_box, w_box = y2 - y1, x2 - x1

            patch = patch_logits[:, p_idx]  # (B, C, H_out, W_out)
            resized_patch = F.interpolate(
                patch,
                size=(h_box, w_box),
                mode='bicubic',
                align_corners=False
            )

            wy = torch.cos(torch.linspace(-math.pi / 2, math.pi / 2, h_box, device=self.device))
            wx = torch.cos(torch.linspace(-math.pi / 2, math.pi / 2, w_box, device=self.device))
            window = torch.ger(wy, wx).unsqueeze(0).unsqueeze(0) + 1e-5 # (1, 1, h_box, w_box)

            img_acc[:, :, y1:y2, x1:x2] += resized_patch * window
            weight_acc[:, :, y1:y2, x1:x2] += window

        full_logits = img_acc / weight_acc
        return full_logits

    @torch.no_grad()
    def forward(
            self,
            image: str | Path | np.ndarray | torch.Tensor,
            is_bgr: bool = False,
            return_logits: bool = False
    ) -> torch.Tensor:

        # 1. Pre-process input
        img_tensor = self._prepare_image(image, is_bgr)

        # 2. Crop into patches
        patches, boxes, H, W, B = self._extract_patches(img_tensor)

        # 3. Inference (batch process all patches)
        patch_logits = self.model(patches)

        # 4. Post-process (Stitching)
        full_logits = self._stitch_logits(patch_logits, boxes, H, W, B)

        # 5. Output
        # Squeeze batch dimension if input was not a batch initially, keeping it intuitive
        argmax_dim = 1
        if full_logits.size(0) == 1 and not (isinstance(image, torch.Tensor) and image.ndim == 4) and not (
                isinstance(image, np.ndarray) and image.ndim == 4):
            full_logits = full_logits.squeeze(0)
            argmax_dim = 0
        if return_logits:
            return full_logits
        else:
            # Mask generation over batch
            mask = torch.argmax(full_logits, dim=argmax_dim)
            return mask
