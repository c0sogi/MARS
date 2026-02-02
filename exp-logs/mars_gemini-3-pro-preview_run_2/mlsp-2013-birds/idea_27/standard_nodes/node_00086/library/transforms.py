import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Handle import path for ImageOnlyTransform across different albumentations versions
try:
    from albumentations import ImageOnlyTransform
except ImportError:
    from albumentations.core.transforms_interface import ImageOnlyTransform

from library.config import Config


class CyclicRoll(ImageOnlyTransform):
    """
    Apply cyclic rolling along the time axis (width) for data augmentation.
    This simulates the temporal invariance of bird calls within the recording window.
    """

    def __init__(self, shift_limit=0.5, always_apply=False, p=0.5):
        super(CyclicRoll, self).__init__(always_apply, p)
        self.shift_limit = shift_limit

    def apply(self, img, shift_factor=0, **params):
        # img is H x W x C
        # Axis 1 is Width (Time)
        h, w = img.shape[:2]
        shift = int(w * shift_factor)
        return np.roll(img, shift, axis=1)

    def get_params(self):
        # Generate a random float between -shift_limit and +shift_limit
        return {"shift_factor": np.random.uniform(-self.shift_limit, self.shift_limit)}

    def get_transform_init_args_names(self):
        return ("shift_limit",)


def cyclic_time_roll(image, shift_percent):
    """
    Deterministic cyclic shift for Test-Time Augmentation (TTA).

    Args:
         image (np.ndarray): Image of shape (H, W, C).
         shift_percent (float): Percentage of width to shift (0.0 to 1.0).

    Returns:
         np.ndarray: Shifted image.
    """
    # Handle both HWC and HW formats
    if len(image.shape) == 3:
        axis = 1
        w = image.shape[1]
    else:
        axis = 1
        w = image.shape[1]

    shift = int(w * shift_percent)
    return np.roll(image, shift, axis=axis)


def mixup_data(x, y, alpha=0.4, device="cuda"):
    """
    Applies Mixup augmentation to a batch of data.

    Args:
        x (torch.Tensor): Batch of images.
        y (torch.Tensor): Batch of labels.
        alpha (float): Mixup parameter.
        device (str): Device to move indices to.

    Returns:
        mixed_x (torch.Tensor): Mixed images.
        y_a (torch.Tensor): Original labels.
        y_b (torch.Tensor): Permuted labels.
        lam (float): Lambda mixing coefficient.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def get_transforms(mode="train", img_height=224, img_width=448):
    """
    Returns the Albumentations composition of transforms.

    Args:
        mode (str): 'train', 'val', or 'test'.
        img_height (int): Target height (Frequency).
        img_width (int): Target width (Time).

    Returns:
        A.Compose: Composed transforms.
    """

    # Normalization constants for ImageNet
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if mode == "train":
        return A.Compose(
            [
                A.Resize(height=img_height, width=img_width),
                # Cyclic Time-Rolling (Augmentation)
                # Randomly shift up to 50% of the width
                CyclicRoll(shift_limit=0.5, p=0.5),
                # SpecAugment Simulation using CoarseDropout
                # 1. Time Masking (Vertical Strips)
                # We force min_height to be large to ensure vertical strips
                A.CoarseDropout(
                    max_holes=8,
                    max_height=img_height,
                    max_width=int(img_width * 0.1),
                    min_holes=1,
                    min_height=int(img_height * 0.8),  # Almost full height
                    min_width=4,
                    fill_value=0,
                    p=0.5,
                ),
                # 2. Frequency Masking (Horizontal Strips)
                # We force min_width to be large to ensure horizontal strips
                A.CoarseDropout(
                    max_holes=8,
                    max_height=int(img_height * 0.1),
                    max_width=img_width,
                    min_holes=1,
                    min_height=4,
                    min_width=int(img_width * 0.8),  # Almost full width
                    fill_value=0,
                    p=0.5,
                ),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    else:
        # Val / Test
        return A.Compose(
            [
                A.Resize(height=img_height, width=img_width),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
