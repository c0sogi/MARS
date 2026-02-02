import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import CFG
from library.utils import seed_everything


def rand_bbox(size, lam):
    """
    Generates a random bounding box for CutMix augmentation.

    Args:
        size (tuple): Dimensions of the image (W, H).
        lam (float): Lambda value sampled from a beta distribution.

    Returns:
        tuple: (x1, y1, x2, y2) coordinates of the bounding box.
    """
    W = size[0]
    H = size[1]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # Uniformly sample the center of the box
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def get_transforms(data, img_size):
    """
    Returns the Albumentations transformation pipeline based on the data split.

    Args:
        data (str): 'train' or 'valid'.
        img_size (int): Target image size (e.g., 384 or 512).

    Returns:
        A.Compose: The transform pipeline.
    """
    if data == "train":
        return A.Compose(
            [
                # Contextual cropping to avoid ambiguous macro crops
                A.RandomResizedCrop(size=(img_size, img_size), scale=CFG.crop_scale),
                A.Transpose(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(p=0.5),
                A.HueSaturationValue(
                    hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5
                ),
                A.CoarseDropout(p=0.5),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    elif data == "valid":
        return A.Compose(
            [
                A.Resize(height=img_size, width=img_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(height=img_size, width=img_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )


class CassavaDataset(Dataset):
    """
    Custom Dataset for Cassava Leaf Disease Classification.
    """

    def __init__(self, df, transform=None):
        self.df = df
        self.file_paths = df["file_path"].values
        # Handle labels if they exist (Train/Val), otherwise None (Test)
        self.labels = df["label"].values if "label" in df.columns else None
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Resolve full path relative to input root
        file_path = os.path.join(CFG.input_root, self.file_paths[idx])

        # Load image via OpenCV
        image = cv2.imread(file_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {file_path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Return label if available, else return a dummy label (-1)
        if self.labels is not None:
            label = torch.tensor(self.labels[idx]).long()
            return image, label
        else:
            return image, torch.tensor(-1).long()


def get_dataloaders(img_size, debug=None):
    """
    Initializes datasets and dataloaders for training, validation, and testing.

    Args:
        img_size (int): Image size for resizing (supports progressive resizing).
        debug (bool, optional): Overrides CFG.debug if provided.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Use CFG debug setting if argument not provided
    is_debug = CFG.debug if debug is None else debug

    # Load Metadata CSVs
    train_df = pd.read_csv(CFG.train_csv)
    val_df = pd.read_csv(CFG.val_csv)
    test_df = pd.read_csv(CFG.test_csv)

    # Subset data for debugging if enabled
    if is_debug:
        train_df = train_df.sample(n=100, random_state=CFG.seed).reset_index(drop=True)
        val_df = val_df.sample(n=50, random_state=CFG.seed).reset_index(drop=True)
        test_df = test_df.sample(n=50, random_state=CFG.seed).reset_index(drop=True)

    # Instantiate Datasets
    train_dataset = CassavaDataset(
        train_df, transform=get_transforms(data="train", img_size=img_size)
    )

    val_dataset = CassavaDataset(
        val_df, transform=get_transforms(data="valid", img_size=img_size)
    )

    test_dataset = CassavaDataset(
        test_df, transform=get_transforms(data="valid", img_size=img_size)
    )

    # Instantiate DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
