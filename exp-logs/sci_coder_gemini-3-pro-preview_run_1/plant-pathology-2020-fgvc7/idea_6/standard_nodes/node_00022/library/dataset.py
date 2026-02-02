import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import get_logger

logger = get_logger("dataset")


def get_transforms(data: str):
    """
    Returns the Albumentations transformation pipeline based on the data split.

    Args:
        data (str): One of 'train', 'valid', or 'test'.

    Returns:
        A.Compose: The transformation pipeline.
    """
    if data == "train":
        return A.Compose(
            [
                A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.1, rotate_limit=30, p=0.5
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.5
                ),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    elif data == "valid" or data == "test":
        return A.Compose(
            [
                A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    else:
        raise ValueError(f"Unknown data split: {data}")


class AppleDataset(Dataset):
    """
    PyTorch Dataset for Apple Disease Detection.
    Reads image paths and labels from the provided metadata DataFrame.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        transform: A.Compose = None,
        data_root: str = Config.INPUT_DIR,
    ):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata (file_paths and targets).
            transform (A.Compose): Albumentations transforms to apply.
            data_root (str): Root directory where images are stored.
        """
        self.df = df
        self.transform = transform
        self.data_root = data_root

        # Pre-extract file paths and labels to avoid overhead in __getitem__
        self.file_paths = self.df["file_path"].values

        # Check if targets exist (they won't for test set)
        self.targets = None
        if set(Config.CLASS_LABELS).issubset(self.df.columns):
            self.targets = self.df[Config.CLASS_LABELS].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Construct full image path
        # file_path in metadata is relative, e.g., "images/Train_0.jpg"
        # data_root is "./input"
        path = os.path.join(self.data_root, self.file_paths[idx])

        # Load image
        image = cv2.imread(path)
        if image is None:
            # Fallback or error handling; usually shouldn't happen with verified metadata
            raise FileNotFoundError(f"Image not found at {path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Prepare return dictionary
        result = {"image": image, "image_id": self.df.iloc[idx]["image_id"]}

        # Add targets if available
        if self.targets is not None:
            label = torch.tensor(self.targets[idx], dtype=torch.float32)
            result["target"] = label

        return result


def rand_bbox(size, lam):
    """
    Generates a random bounding box for CutMix.

    Args:
        size (tuple): Size of the image tensor (Batch, Channel, Height, Width).
        lam (float): Lambda value sampled from beta distribution.

    Returns:
        tuple: (bbx1, bby1, bbx2, bby2) coordinates.
    """
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


def cutmix(data, targets, alpha):
    """
    Applies CutMix augmentation to a batch.

    Args:
        data (torch.Tensor): Input batch of images.
        targets (torch.Tensor): Input batch of targets.
        alpha (float): Alpha parameter for Beta distribution.

    Returns:
        tuple: (mixed_data, target_a, target_b, lam)
    """
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]

    lam = np.random.beta(alpha, alpha)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)

    # Adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))

    data[:, :, bbx1:bbx2, bby1:bby2] = shuffled_data[:, :, bbx1:bbx2, bby1:bby2]

    return data, targets, shuffled_targets, lam


def mixup(data, targets, alpha):
    """
    Applies Mixup augmentation to a batch.

    Args:
        data (torch.Tensor): Input batch of images.
        targets (torch.Tensor): Input batch of targets.
        alpha (float): Alpha parameter for Beta distribution.

    Returns:
        tuple: (mixed_data, target_a, target_b, lam)
    """
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]

    lam = np.random.beta(alpha, alpha)
    data = data * lam + shuffled_data * (1 - lam)

    return data, targets, shuffled_targets, lam
