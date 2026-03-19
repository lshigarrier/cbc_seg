import math
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.io import read_image
from pathlib import Path

from augmentation import get_train_transform, get_test_transform


class ImageDataset(Dataset):

    def __init__(
            self,
            folder: Path | str,
            patch_per_row: int = 7,
            patch_per_col:int = 3,
            patch_size: int = 1024,
            patch_overlap: float = 0.5
    ) -> None:
        super().__init__()
        self.paths = sorted(Path(folder).rglob('*.jpg'))
        self.patch_per_row = patch_per_row
        self.patch_per_col = patch_per_col
        self.patch_per_img = patch_per_row * patch_per_col
        self.patch_size = patch_size
        assert 0.0 <= patch_overlap < 1.0, 'patch_overlap must be in [0, 1)'
        self.patch_overlap = patch_overlap
        self.transform = get_test_transform()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(
            self,
            item: int
    ) -> tuple[torch.Tensor, str, torch.Tensor, tuple[tuple[int, int, int, int], ...]]:
        path = self.paths[item]
        img = read_image(str(path))
        name = path.stem

        _, H, W = img.shape
        Pw = W / (1 + (self.patch_per_row - 1) * (1 - self.patch_overlap)) if self.patch_per_row > 1 else W
        Ph = H / (1 + (self.patch_per_col - 1) * (1 - self.patch_overlap)) if self.patch_per_col > 1 else H

        patches = []
        boxes = []

        for i in range(self.patch_per_col):
            for j in range(self.patch_per_row):
                # Calculate the start (y1, x1) and end (y2, x2) indices for the current patch
                y1 = int(i * Ph * (1 - self.patch_overlap))
                x1 = int(j * Pw * (1 - self.patch_overlap))

                # For the last row/column, force it to extend all the way to H/W to avoid rounding gaps
                y2 = int(y1 + Ph) if i < self.patch_per_col - 1 else H
                x2 = int(x1 + Pw) if j < self.patch_per_row - 1 else W

                y2 = min(y2, H)
                x2 = min(x2, W)

                patch = img[:, y1:y2, x1:x2]
                patch = F.interpolate(
                    patch.unsqueeze(0),
                    size=(self.patch_size, self.patch_size),
                    mode='bicubic',
                    align_corners=False
                )
                patches.append(patch)
                boxes.append((y1, y2, x1, x2))

        patches = torch.cat(patches, dim=0)

        return img, name, self.transform(patches), tuple(boxes)

    def stitch(self, patches: torch.Tensor, boxes: tuple[tuple[int, int, int, int], ...]) -> list[torch.Tensor]:
        # patches : (num_images * patch_per_img, channels, patch_size, patch_size)
        # boxes : contains (y1, y2, x1, x2) for each patch
        num_patches = patches.shape[0]
        assert num_patches % self.patch_per_img == 0
        num_images = num_patches // self.patch_per_img

        device = patches.device
        dtype = patches.dtype
        C = patches.shape[1]

        all_images = []

        for b in range(num_images):
            # Extract the bounding boxes for the current image
            img_boxes = boxes[b * self.patch_per_img: (b + 1) * self.patch_per_img]

            # Reconstruct the original full image sizes from the bounding boxes
            H = max(box[1] for box in img_boxes)
            W = max(box[3] for box in img_boxes)

            # Create accumulators for the blended image and the blending weights
            img_acc = torch.zeros((C, H, W), device=device, dtype=torch.float32)
            weight_acc = torch.zeros((1, H, W), device=device, dtype=torch.float32)

            for p_idx in range(self.patch_per_img):
                global_p_idx = b * self.patch_per_img + p_idx
                y1, y2, x1, x2 = img_boxes[p_idx]
                h_box, w_box = y2 - y1, x2 - x1

                # Resize patch back to its original bounding box size
                patch = patches[global_p_idx: global_p_idx + 1]
                resized_patch = F.interpolate(patch, size=(h_box, w_box), mode='bicubic', align_corners=False).squeeze(0)

                # Create a 2D cosine/hann window (peaks at 1 in the middle, decays to 0 at the edges)
                wy = torch.cos(torch.linspace(-math.pi / 2, math.pi / 2, h_box, device=device))
                wx = torch.cos(torch.linspace(-math.pi / 2, math.pi / 2, w_box, device=device))
                window = torch.ger(wy, wx).unsqueeze(0) + 1e-5  # Add a tiny epsilon to prevent division by zero

                # Accumulate the weighted patch and the weights
                img_acc[:, y1:y2, x1:x2] += resized_patch * window
                weight_acc[:, y1:y2, x1:x2] += window

            # Normalize the accumulated image by the sum of weights (this performs the smooth blending)
            reconstructed = img_acc / weight_acc
            all_images.append(reconstructed.to(dtype))

        return all_images


def image_collate_fn(batch: list) ->  tuple[tuple[torch.Tensor, ...], tuple[str, ...], torch.Tensor, tuple[tuple[int, int, int, int], ...]]:
    all_imgs = tuple(item[0] for item in batch)
    all_names = tuple(item[1] for item in batch)
    # item[2] is a tensor of shape (patch_per_img, C, patch_size, patch_size)
    # torch.cat creates a flat batch: (batch_size * patch_per_img, C, patch_size, patch_size)
    all_patches = torch.cat([item[2] for item in batch], dim=0)
    # item[3] is a tuple of bounding boxes: ((y1, y2, x1, x2), ...)
    # sum(..., ()) cleanly flattens the list of tuples into one giant tuple of boxes
    all_boxes = sum((item[3] for item in batch), ())
    return all_imgs, all_names, all_patches, all_boxes


class ImageMaskDataset(ImageDataset):

    def __init__(
            self,
            folder: Path | str,
            patch_per_row: int = 4,
            patch_per_col:int = 2,
            patch_size: int = 1024,
            patch_overlap: float = 0.
    ) -> None:
        super().__init__(
            folder,
            patch_per_row,
            patch_per_col,
            patch_size,
            patch_overlap
        )
        self.paths = sorted(Path(folder).rglob('*_mask.png'))

    def get_class_image_counts(self, class_mapping: dict, ignore_index: int = 255) -> dict:
        """
        Calculates how many annotated images contain each class.
        """
        # Initialize counts class mapping names
        counts = {cls_name: 0 for cls_name in class_mapping.keys()}
        # Reverse mapping to get class names from indices
        index_to_name = {v: k for k, v in class_mapping.items()}

        for path in self.paths:
            mask = read_image(str(path))
            # Find unique values in the mask tensor
            unique_classes = torch.unique(mask).tolist()

            for cls_idx in unique_classes:
                if cls_idx == ignore_index:
                    continue
                if cls_idx in index_to_name:
                    counts[index_to_name[cls_idx]] += 1

        return counts

    def __getitem__(
            self,
            item: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        path = self.paths[item]
        mask = read_image(str(path))
        img = read_image(str(path.with_name(f"{path.stem[:-5]}.jpg")))

        _, H, W = img.shape
        Pw = W / (1 + (self.patch_per_row - 1) * (1 - self.patch_overlap)) if self.patch_per_row > 1 else W
        Ph = H / (1 + (self.patch_per_col - 1) * (1 - self.patch_overlap)) if self.patch_per_col > 1 else H

        patches = []
        mask_patches = []

        for i in range(self.patch_per_col):
            for j in range(self.patch_per_row):
                # Calculate the start (y1, x1) and end (y2, x2) indices for the current patch
                y1 = int(i * Ph * (1 - self.patch_overlap))
                x1 = int(j * Pw * (1 - self.patch_overlap))

                # For the last row/column, force it to extend all the way to H/W to avoid rounding gaps
                y2 = int(y1 + Ph) if i < self.patch_per_col - 1 else H
                x2 = int(x1 + Pw) if j < self.patch_per_row - 1 else W

                y2 = min(y2, H)
                x2 = min(x2, W)

                patch = img[:, y1:y2, x1:x2]
                patch = F.interpolate(
                    patch.unsqueeze(0),
                    size=(self.patch_size, self.patch_size),
                    mode='bicubic',
                    align_corners=False
                )
                patches.append(patch)
                mask_patch = mask[:, y1:y2, x1:x2]
                mask_patch = F.interpolate(
                    mask_patch.unsqueeze(0),
                    size=(self.patch_size, self.patch_size),
                    mode='nearest'
                )
                mask_patches.append(mask_patch)

        patches = tv_tensors.Image(torch.cat(patches, dim=0))
        mask_patches = tv_tensors.Mask(torch.cat(mask_patches, dim=0).squeeze(1))

        patches, mask_patches = self.transform(patches, mask_patches)

        return patches.as_subclass(torch.Tensor), mask_patches.as_subclass(torch.Tensor).long()


def image_mask_collate_fn(batch: list) ->  tuple[torch.Tensor, torch.Tensor]:
    # item[0] is a tensor of shape (patch_per_img, C, patch_size, patch_size)
    # torch.cat creates a flat batch: (batch_size * patch_per_img, C, patch_size, patch_size)
    all_patches = torch.cat([item[0] for item in batch], dim=0)
    all_masks = torch.cat([item[1] for item in batch], dim=0)
    return all_patches, all_masks


class TrainImageMaskDataset(ImageMaskDataset):
    def __init__(
            self,
            folder: Path | str,
            patch_per_img: int = 4,
            patch_size: int = 1024,
    ) -> None:
        super().__init__(folder=folder, patch_size=patch_size)
        self.patch_per_img = patch_per_img
        self.transform = get_train_transform(patch_size)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        path = self.paths[item]
        mask = read_image(str(path))
        img = read_image(str(path.with_name(f"{path.stem[:-5]}.jpg")))

        img = tv_tensors.Image(img)
        mask = tv_tensors.Mask(mask.squeeze(0))  # [H, W]

        patches = []
        mask_patches = []

        # Generate N independent random crops from the same full image
        for _ in range(self.patch_per_img):
            p, m = self.transform(img, mask)
            patches.append(p)
            mask_patches.append(m)

        # Stack them into [patch_per_img, C, H, W]
        patches = torch.stack(patches, dim=0)
        mask_patches = torch.stack(mask_patches, dim=0)

        # Return as standard tensors
        return patches.as_subclass(torch.Tensor), mask_patches.as_subclass(torch.Tensor).long()
