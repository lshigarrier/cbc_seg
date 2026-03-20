import torch
import pytorch_lightning as pl
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from data.augmentation import get_train_transform, get_test_transform


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

    def __getitem__(self, item):
        path = self.paths[item]

        # Read with OpenCV and convert BGR to RGB
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        name = path.stem

        H, W, _ = img.shape
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

                patch = img[y1:y2, x1:x2, :]
                patch = cv2.resize(patch, (self.patch_size, self.patch_size), interpolation=cv2.INTER_CUBIC)
                patches.append(patch['image'])
                boxes.append((y1, y2, x1, x2))

        patches = np.stack(patches, axis=0)
        patches = torch.from_numpy(patches).permute(0, 3, 1, 2)

        return img , name, self.transform(patches), tuple(boxes)


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
            *args,
            **kwargs
    ) -> None:
        super().__init__(folder, *args, **kwargs)
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
            # Read mask as grayscale numpy array
            mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            # Find unique values in the mask tensor
            unique_classes = np.unique(mask).tolist()

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

        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(str(path.with_name(f"{path.stem[:-5]}.jpg")), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        H, W, _ = img.shape
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

                patch = img[y1:y2, x1:x2, :]
                mask_patch = mask[y1:y2, x1:x2]

                # Resize Image (Cubic) and Mask (Nearest)
                patch = cv2.resize(patch, (self.patch_size, self.patch_size), interpolation=cv2.INTER_CUBIC)
                mask_patch = cv2.resize(mask_patch, (self.patch_size, self.patch_size), interpolation=cv2.INTER_NEAREST)

                patches.append(patch)
                mask_patches.append(mask_patch)

        patches = np.stack(patches, axis=0)
        mask_patches = np.stack(mask_patches, axis=0)
        patches = torch.from_numpy(patches).permute(0, 3, 1, 2)
        mask_patches = torch.from_numpy(mask_patches)

        return self.transform(patches), mask_patches.long()


def image_mask_collate_fn(batch: list) ->  tuple[torch.Tensor, torch.Tensor]:
    # item[0] is a tensor of shape (patch_per_img, C, patch_size, patch_size)
    # torch.cat creates a flat batch: (batch_size * patch_per_img, C, patch_size, patch_size)
    all_patches = torch.cat([item[0] for item in batch], dim=0)
    all_masks = torch.cat([item[1] for item in batch], dim=0)
    return all_patches, all_masks


class TrainImageMaskDataset(ImageMaskDataset):
    def __init__(
            self,
            *args,
            patch_per_img: int = 4,
            patch_size: int = 1024,
            ignore_index: int = 255,
            **kwargs
    ) -> None:
        super().__init__(*args, patch_size=patch_size, **kwargs)
        self.patch_per_img = patch_per_img
        self.transform = get_train_transform(patch_size, ignore_index)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        path = self.paths[item]

        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(str(path.with_name(f"{path.stem[:-5]}.jpg")), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        patches = []
        mask_patches = []

        # Generate N independent random crops from the same full image
        for _ in range(self.patch_per_img):
            augmented = self.transform(image=img, mask=mask)
            patches.append(augmented['image'])
            mask_patches.append(augmented['mask'])

        patches = torch.stack(patches, dim=0)
        mask_patches = torch.stack(mask_patches, dim=0)

        return patches, mask_patches.long()


class ImageDataModule(pl.LightningDataModule):
    def __init__(self, conf, logger):
        super().__init__()
        self.conf = conf
        self.logger = logger
        self.train_dataset = None
        self.predict_dataset = None

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = TrainImageMaskDataset(
                folder=self.conf.data_dir,
                patch_per_img=self.conf.patch_per_img,
                patch_size=self.conf.patch_size,
                ignore_index=self.conf.ignore_index
            )
            class_counts = self.train_dataset.get_class_image_counts(self.conf.class_mapping, self.conf.ignore_index)
            self.logger.info('-' * 70)
            self.logger.info(f"Training on {len(self.train_dataset)} annotated images")
            self.logger.info("Number of images containing each class:")
            for cls_name, count in class_counts.items():
                self.logger.info(f"  - {cls_name}: {count}")
            self.logger.info('-' * 70)

        if stage == 'predict' or stage is None:
            self.predict_dataset = ImageDataset(
                folder=self.conf.eval_data_dir,
                patch_per_row=self.conf.patch_per_row,
                patch_per_col=self.conf.patch_per_col,
                patch_size=self.conf.patch_size,
                patch_overlap=self.conf.patch_overlap
            )
            self.logger.info('-' * 70)
            self.logger.info(f"Inference on {len(self.predict_dataset)} images")
            self.logger.info('-' * 70)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.conf.batch_size,
            shuffle=True,
            num_workers=self.conf.num_workers,
            persistent_workers=True,
            pin_memory=self.conf.use_gpu,
            collate_fn=image_mask_collate_fn
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.conf.num_workers,
            persistent_workers=True,
            pin_memory=self.conf.use_gpu,
            collate_fn=image_collate_fn
        )
