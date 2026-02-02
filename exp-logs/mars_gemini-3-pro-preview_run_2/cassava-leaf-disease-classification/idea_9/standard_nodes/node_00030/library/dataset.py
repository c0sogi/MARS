import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import CFG


class CassavaDataset(Dataset):
    """
    Dataset class for Cassava Leaf Disease Classification.
    Uses PIL for image loading to ensure consistency and native handling.
    """

    def __init__(self, df, transform=None, output_label=True):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe containing 'file_path' and 'label'.
            transform (albumentations.Compose): Augmentation pipeline.
            output_label (bool): Whether to return labels (True for train/val, False for test).
        """
        self.df = df
        self.transform = transform
        self.output_label = output_label
        self.file_paths = df["file_path"].values

        # Handle labels if they exist and are required
        if self.output_label:
            self.labels = df["label"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Construct full file path
        # file_path in df is relative (e.g., "train_images/xyz.jpg")
        path = os.path.join(CFG.input_dir, self.file_paths[idx])

        # Load image using PIL as required
        try:
            image = Image.open(path).convert("RGB")
            image = np.array(image)
        except Exception as e:
            # Fallback for corrupt images (though dataset is assumed clean)
            # Return a blank image to prevent crashing
            image = np.zeros((600, 800, 3), dtype=np.uint8)

        # Apply augmentations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Return image and label (if requested)
        if self.output_label:
            label = self.labels[idx]
            return image, label

        return image


def get_transforms(phase, size):
    """
    Returns the Albumentations transformation pipeline.

    Args:
        phase (str): 'train', 'valid', or 'test'.
        size (int): Input resolution (e.g., 224, 384).
    """
    # ImageNet normalization statistics
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if phase == "train":
        return A.Compose(
            [
                # Geometric Augmentations
                A.RandomResizedCrop(size=(size, size), scale=(0.08, 1.0)),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.2, rotate_limit=20, p=0.5
                ),
                # Photometric Augmentations (Simulating RandAugment)
                A.OneOf(
                    [
                        A.HueSaturationValue(
                            hue_shift_limit=0.2,
                            sat_shift_limit=0.2,
                            val_shift_limit=0.2,
                            p=0.9,
                        ),
                        A.RandomBrightnessContrast(
                            brightness_limit=0.2, contrast_limit=0.2, p=0.9
                        ),
                        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.9),
                    ],
                    p=0.5,
                ),
                # Regularization
                A.CoarseDropout(
                    max_holes=8,
                    max_height=int(size * 0.1),
                    max_width=int(size * 0.1),
                    min_holes=4,
                    min_height=4,
                    min_width=4,
                    fill_value=0,
                    p=0.5,
                ),
                # Normalization & Tensor Conversion
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )

    elif phase in ["valid", "test"]:
        return A.Compose(
            [
                # Resize to slightly larger than target, then center crop
                # This preserves aspect ratio better than simple resize
                A.Resize(height=int(size * 1.14), width=int(size * 1.14)),
                A.CenterCrop(height=size, width=size),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )

    else:
        # Default fallback
        return A.Compose(
            [
                A.Resize(height=size, width=size),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )


class Mixup:
    """
    Implements Mixup and CutMix augmentation logic for batches.
    """

    def __init__(
        self,
        mixup_alpha=1.0,
        cutmix_alpha=1.0,
        prob=0.5,
        switch_prob=0.5,
        num_classes=5,
    ):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
        self.switch_prob = switch_prob
        self.num_classes = num_classes

    def __call__(self, batch_x, batch_y):
        """
        Args:
            batch_x (Tensor): Batch of images [B, C, H, W]
            batch_y (Tensor): Batch of labels [B]

        Returns:
            mixed_x (Tensor): Mixed images
            mixed_y (Tensor): Soft targets [B, NumClasses]
        """
        # Always convert to one-hot first for soft target consistency
        batch_y_onehot = F.one_hot(batch_y, num_classes=self.num_classes).float()

        # Determine if we apply mixing
        if np.random.rand() > self.prob:
            return batch_x, batch_y_onehot

        batch_size = batch_x.size(0)
        indices = torch.randperm(batch_size, device=batch_x.device)

        shuffled_x = batch_x[indices]
        shuffled_y = batch_y_onehot[indices]

        # Decide between Mixup and CutMix
        if np.random.rand() < self.switch_prob:
            # CutMix
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)

            bbx1, bby1, bbx2, bby2 = self.rand_bbox(batch_x.size(), lam)

            # Adjust lambda to match exact pixel ratio
            lam = 1 - (
                (bbx2 - bbx1) * (bby2 - bby1) / (batch_x.size(2) * batch_x.size(3))
            )

            # Create mixed image
            mixed_x = batch_x.clone()
            mixed_x[:, :, bbx1:bbx2, bby1:bby2] = shuffled_x[:, :, bbx1:bbx2, bby1:bby2]

            # Mix labels
            mixed_y = lam * batch_y_onehot + (1 - lam) * shuffled_y

        else:
            # Mixup
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)

            mixed_x = lam * batch_x + (1 - lam) * shuffled_x
            mixed_y = lam * batch_y_onehot + (1 - lam) * shuffled_y

        return mixed_x, mixed_y

    def rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        # Uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2
