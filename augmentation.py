import random
import torch
from torchvision.transforms import v2
from torchvision.transforms import InterpolationMode
from torchvision import tv_tensors

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


class AddGaussianNoise(v2.Transform):
    """Add Gaussian noise to a tensor in [0,1] with prob p. Safely ignores Masks."""
    def __init__(self, sigma: float = 0.02, p: float = 0.2):
        super().__init__()
        self.sigma = sigma
        self.p = p

    def _transform(self, inpt, params):
        # Only apply to Image tensors, ignore Mask tensors
        if isinstance(inpt, tv_tensors.Image) or (isinstance(inpt, torch.Tensor) and not isinstance(inpt, tv_tensors.Mask)):
            if random.random() < self.p:
                return (inpt + torch.randn_like(inpt) * self.sigma).clamp(0.0, 1.0)
        return inpt


def get_train_transform(img_size: int):
    return v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),

        v2.RandomResizedCrop(
            size=(img_size, img_size),
            scale=(0.05, 0.5),
            ratio=(0.8, 1.2),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ),

        # --- photometric (automatically ignored by masks) ---
        v2.RandomApply([v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02)], p=0.7),
        v2.RandomApply([v2.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.2),
        v2.RandomPosterize(bits=5, p=0.08),
        v2.RandomAdjustSharpness(sharpness_factor=1.5, p=0.12),
        v2.RandomAutocontrast(p=0.12),
        v2.RandomEqualize(p=0.08),

        # Erasing handles mask ignoring automatically in v2
        v2.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random'),

        v2.ToDtype(torch.float32, scale=True),
        AddGaussianNoise(sigma=0.02, p=0.2),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_test_transform():
    return v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
