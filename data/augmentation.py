import torch
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.transforms import v2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def get_train_transform(img_size: int, ignore_index: int = 255):
    return A.Compose([
        # --- Spatial & Cropping ---
        A.RandomResizedCrop(
            size=(img_size, img_size),
            scale=(0.05, 0.5),
            ratio=(0.8, 1.2),
            interpolation=cv2.INTER_CUBIC,
            p=1.0
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.GridDistortion(num_steps=5, distort_limit=0.2, interpolation=cv2.INTER_CUBIC, p=0.2),

        # --- Domain Specific (Pavement/Airport) ---
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
        A.RandomShadow(num_shadows_limit=(1, 3), shadow_dimension=5, p=0.2),
        A.RandomSunFlare(src_radius=150, src_color=(255, 255, 255), p=0.1),

        # --- Photometric ---
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02, p=0.7),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        A.Posterize(num_bits=5, p=0.08),
        A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.12),

        # --- Noise & Erasing ---
        A.GaussNoise(std_range=(0.05, 0.2), p=0.2),
        A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(0.05, 0.2),
            hole_width_range=(0.05, 0.2),
            fill='random',
            fill_mask=ignore_index,
            p=0.3
        ),

        # --- Formatting ---
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])


def get_test_transform():
    return v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
