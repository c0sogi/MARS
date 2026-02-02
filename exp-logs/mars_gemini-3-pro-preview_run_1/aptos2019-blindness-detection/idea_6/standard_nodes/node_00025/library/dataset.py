import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config


class RetinopathyDataset(Dataset):
    """
    Custom Dataset for Diabetic Retinopathy Classification.
    Handles image loading, 'squashing' resize strategy, photometric augmentations,
    and ordinal target encoding.
    """

    def __init__(self, df, phase, transforms=None):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata (id_code, file_path, diagnosis).
            phase (str): 'train', 'val', or 'test'.
            transforms (albumentations.Compose): Augmentation pipeline.
        """
        self.df = df
        self.phase = phase
        self.transforms = transforms
        self.input_dir = Config.INPUT_DIR

        # Pre-calculate ordinal targets for train/val to speed up __getitem__
        if self.phase in ["train", "val"]:
            self.targets = self.df["diagnosis"].values
            # Create ordinal vectors: shape (N, 4)
            # Label k (0-4) -> sum of first k units is k.
            # 0 -> [0,0,0,0]
            # 1 -> [1,0,0,0]
            # 2 -> [1,1,0,0]
            # 3 -> [1,1,1,0]
            # 4 -> [1,1,1,1]
            self.ordinal_targets = np.zeros(
                (len(self.df), Config.NUM_OUTPUTS), dtype=np.float32
            )
            for i, label in enumerate(self.targets):
                if label > 0:
                    self.ordinal_targets[i, : int(label)] = 1.0

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full file path
        # Metadata file_path is relative, e.g., "train_images/xxxx.png"
        # Input dir is "./input"
        image_path = os.path.join(self.input_dir, row["file_path"])

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            # Fallback for missing images (though metadata validation passed)
            # Create a black image of target size
            image = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image_tensor = augmented["image"]
        else:
            # Fallback transform if None provided (Resize + Normalize + ToTensor)
            transform = A.Compose(
                [
                    A.Resize(height=Config.IMG_SIZE, width=Config.IMG_SIZE),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ]
            )
            image_tensor = transform(image=image)["image"]

        if self.phase in ["train", "val"]:
            target = self.ordinal_targets[idx]
            return image_tensor, torch.tensor(target, dtype=torch.float32)
        else:
            # For test phase, return image and id_code for submission
            return image_tensor, row["id_code"]


def get_transforms(phase: str):
    """
    Returns the Albumentations transformation pipeline for the specified phase.
    Implements the 'squashing' resize and photometric augmentations.
    """
    if phase == "train":
        return A.Compose(
            [
                # Squashing Strategy: Resize directly ignoring aspect ratio
                A.Resize(height=Config.IMG_SIZE, width=Config.IMG_SIZE),
                # Geometric Augmentations
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=30, p=0.5),
                # Photometric Augmentations (Domain Shift Robustness)
                A.RandomBrightnessContrast(
                    brightness_limit=Config.AUG_BRIGHTNESS,
                    contrast_limit=Config.AUG_CONTRAST,
                    p=Config.AUG_PROB,
                ),
                # Normalization and Tensor conversion
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        # Validation / Test
        return A.Compose(
            [
                A.Resize(height=Config.IMG_SIZE, width=Config.IMG_SIZE),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


def get_dataloaders(debug=False, subset_size=100):
    """
    Creates and returns DataLoaders for train, validation, and test sets.

    Args:
        debug (bool): If True, subsets the data for quick debugging.
        subset_size (int): Number of samples to use in debug mode.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_META)
    val_df = pd.read_csv(Config.VAL_META)
    test_df = pd.read_csv(Config.TEST_META)

    # Debugging: Subset data
    if debug:
        train_df = train_df.iloc[:subset_size]
        val_df = val_df.iloc[:subset_size]
        test_df = test_df.iloc[:subset_size]

    # Create Datasets
    train_dataset = RetinopathyDataset(
        train_df, phase="train", transforms=get_transforms("train")
    )

    val_dataset = RetinopathyDataset(
        val_df, phase="val", transforms=get_transforms("val")
    )

    test_dataset = RetinopathyDataset(
        test_df, phase="test", transforms=get_transforms("test")
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
