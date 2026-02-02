import os
import cv2
import math
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config


# --- Transforms ---
def get_transforms(img_size, mode="train"):
    """
    Returns the Albumentations transformation pipeline.

    Args:
        img_size (int): Target image size (height and width).
        mode (str): 'train', 'valid', or 'test'.
    """
    if mode == "train":
        return A.Compose(
            [
                A.RandomResizedCrop(
                    height=img_size, width=img_size, scale=Config.CROP_SCALE
                ),
                A.Transpose(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.5
                ),
                A.HueSaturationValue(
                    hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5
                ),
                A.CoarseDropout(
                    max_holes=8,
                    max_height=int(img_size * 0.1),
                    max_width=int(img_size * 0.1),
                    p=0.5,
                ),
                A.Normalize(mean=Config.MEAN, std=Config.STD),
                ToTensorV2(),
            ]
        )
    else:
        # Validation and Test: Resize to target size deterministically
        return A.Compose(
            [
                A.Resize(height=img_size, width=img_size),
                A.Normalize(mean=Config.MEAN, std=Config.STD),
                ToTensorV2(),
            ]
        )


# --- Dataset ---
class CassavaDataset(Dataset):
    """
    Dataset class for Cassava Leaf Disease Classification.
    Reads images based on metadata paths.
    """

    def __init__(self, df, transform=None, mode="train"):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata (image_id, label, file_path).
            transform (albumentations.Compose): Transformation pipeline.
            mode (str): 'train', 'valid', or 'test'.
        """
        self.df = df
        self.transform = transform
        self.mode = mode

        # Pre-compute full paths to avoid string concatenation in loop
        # Metadata paths are relative to ./input, e.g., "train_images/img.jpg"
        self.file_paths = [
            os.path.join("./input", path) for path in df["file_path"].values
        ]

        # Store labels if available
        if "label" in df.columns:
            self.labels = df["label"].values
        else:
            self.labels = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]

        # Read image using OpenCV
        img = cv2.imread(file_path)
        if img is None:
            # Fallback or error raising
            raise FileNotFoundError(f"Image not found at {file_path}")

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented["image"]

        # Return logic based on mode
        if self.mode == "test":
            # For test, we return image and image_id to create the submission
            return img, self.df.iloc[idx]["image_id"]
        else:
            # For train/val, return image and label
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return img, label


# --- Collate Function (MixUp / CutMix) ---
class MixupCollate:
    """
    Collate function that applies MixUp or CutMix to a batch of data.
    """

    def __init__(
        self,
        mixup_alpha=Config.MIXUP_ALPHA,
        cutmix_alpha=Config.CUTMIX_ALPHA,
        prob=Config.MIXUP_PROB,
        num_classes=Config.NUM_CLASSES,
    ):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
        self.num_classes = num_classes

    def __call__(self, batch):
        """
        Args:
            batch: List of tuples (image, label)
        """
        images, labels = zip(*batch)
        images = torch.stack(images)
        labels = torch.stack(labels)

        # One-hot encode labels
        batch_size = images.size(0)
        # Use the device of the input images (likely CPU here, moved to GPU later)
        one_hot_labels = torch.zeros(batch_size, self.num_classes, device=images.device)
        one_hot_labels.scatter_(1, labels.view(-1, 1), 1)

        # Decide whether to apply augmentation
        if np.random.rand() > self.prob:
            return images, one_hot_labels

        # Decide MixUp vs CutMix (50/50 split if active)
        use_cutmix = np.random.rand() > 0.5

        if use_cutmix:
            return self.cutmix(images, one_hot_labels)
        else:
            return self.mixup(images, one_hot_labels)

    def mixup(self, images, labels):
        batch_size = images.size(0)
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)

        index = torch.randperm(batch_size)

        mixed_images = lam * images + (1 - lam) * images[index, :]
        mixed_labels = lam * labels + (1 - lam) * labels[index, :]

        return mixed_images, mixed_labels

    def cutmix(self, images, labels):
        batch_size = images.size(0)
        W, H = images.size(3), images.size(2)

        lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)

        # Generate bounding box
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        index = torch.randperm(batch_size)

        # Apply CutMix to images
        mixed_images = images.clone()
        mixed_images[:, :, bby1:bby2, bbx1:bbx2] = images[
            index, :, bby1:bby2, bbx1:bbx2
        ]

        # Adjust lambda to exact area ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        mixed_labels = lam * labels + (1 - lam) * labels[index, :]

        return mixed_images, mixed_labels
