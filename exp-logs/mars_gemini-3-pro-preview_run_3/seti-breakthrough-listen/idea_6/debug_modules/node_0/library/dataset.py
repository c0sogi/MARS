import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config


class SETIDataset(Dataset):
    """
    Custom Dataset for SETI Technosignature detection.
    Handles loading .npy spectrograms, splitting into On/Off target streams,
    padding, and applying synchronized augmentations.
    """

    def __init__(self, df, transform=None):
        """
        Args:
            df (pd.DataFrame): DataFrame containing 'file_path' and 'target'.
            transform (albumentations.Compose): Augmentation pipeline.
        """
        self.df = df
        self.transform = transform
        # Pre-construct full file paths for efficiency
        self.file_paths = [
            os.path.join(Config.INPUT_DIR, p) for p in df["file_path"].values
        ]
        self.targets = df["target"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        target = self.targets[idx]

        # Load spectrogram: Shape (6, 273, 256)
        # 6 positions, 273 freq bins, 256 time steps
        try:
            img = np.load(path).astype(np.float32)
        except Exception:
            # Fallback for robustness (should not happen given metadata checks)
            img = np.zeros(Config.ORIG_SHAPE, dtype=np.float32)

        # Split into On-Target (A observations) and Off-Target (B, C, D observations)
        # On-Target indices: 0, 2, 4
        # Off-Target indices: 1, 3, 5
        on_target = img[[0, 2, 4], :, :]  # Shape: (3, 273, 256)
        off_target = img[[1, 3, 5], :, :]  # Shape: (3, 273, 256)

        # Transpose to (H, W, C) for Albumentations and Padding
        # Result: (273, 256, 3)
        on_target = np.transpose(on_target, (1, 2, 0))
        off_target = np.transpose(off_target, (1, 2, 0))

        # Pad Height (Frequency axis) to Config.IMG_HEIGHT (288)
        # Current H=273, Target H=288 -> Pad 15 pixels
        h, w, c = on_target.shape
        pad_h = Config.IMG_HEIGHT - h

        if pad_h > 0:
            # Pad at the end of the height dimension
            padding = ((0, pad_h), (0, 0), (0, 0))
            on_target = np.pad(on_target, padding, mode="constant", constant_values=0)
            off_target = np.pad(off_target, padding, mode="constant", constant_values=0)

        # Apply Synchronized Augmentations
        # We treat 'on_target' as the main image and 'off_target' as an additional target
        # to ensure the EXACT same geometric transforms are applied to both.
        if self.transform:
            augmented = self.transform(image=on_target, off_image=off_target)
            on_target = augmented["image"]
            off_target = augmented["off_image"]
        else:
            # Fallback conversion if no transform provided
            on_target = torch.from_numpy(on_target.transpose(2, 0, 1))
            off_target = torch.from_numpy(off_target.transpose(2, 0, 1))

        return on_target, off_target, torch.tensor(target, dtype=torch.float32)


def get_transforms(phase="train"):
    """
    Returns the Albumentations transform pipeline.

    Args:
        phase (str): 'train', 'val', or 'test'.
    """
    # We define 'off_image' as an additional target of type 'image'
    # This ensures geometric transforms (flips) are applied to both inputs identically.
    additional_targets = {"off_image": "image"}

    if phase == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),  # Time reversal
                A.VerticalFlip(p=0.5),  # Frequency inversion
                ToTensorV2(),  # Converts (H, W, C) -> (C, H, W)
            ],
            additional_targets=additional_targets,
        )
    else:
        return A.Compose([ToTensorV2()], additional_targets=additional_targets)


def get_train_val_loaders(debug=False):
    """
    Creates DataLoaders for training and validation sets.

    Args:
        debug (bool): If True, subsets the data for rapid testing.

    Returns:
        train_loader, val_loader
    """
    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA)
    val_df = pd.read_csv(Config.VAL_METADATA)

    if debug:
        train_df = train_df.iloc[: Config.DEBUG_SAMPLE_SIZE]
        val_df = val_df.iloc[: Config.DEBUG_SAMPLE_SIZE]

    # Instantiate Datasets
    train_dataset = SETIDataset(train_df, transform=get_transforms("train"))
    val_dataset = SETIDataset(val_df, transform=get_transforms("val"))

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

    return train_loader, val_loader


def get_test_loader():
    """
    Creates DataLoader for the test set.

    Returns:
        test_loader
    """
    test_df = pd.read_csv(Config.TEST_METADATA)

    # Instantiate Dataset
    test_dataset = SETIDataset(test_df, transform=get_transforms("test"))

    # Create DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return test_loader
