import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A

from library.config import Config


def get_transforms(data="train"):
    """
    Returns the Albumentations transformations for train, validation, or test sets.

    Args:
        data (str): One of 'train', 'valid', 'test'.

    Returns:
        albumentations.Compose: The composition of transforms.
    """
    if data == "train":
        return A.Compose(
            [
                A.Resize(Config.height, Config.width),
                A.HorizontalFlip(p=Config.hflip_prob),
                A.VerticalFlip(p=Config.vflip_prob),
            ]
        )
    elif data == "valid" or data == "test":
        return A.Compose(
            [
                A.Resize(Config.height, Config.width),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(Config.height, Config.width),
            ]
        )


class SETIDataset(Dataset):
    """
    Custom Dataset for SETI Signal Detection.
    Handles loading .npy spectrograms, vertical stacking, normalization,
    channel expansion, and augmentation.
    """

    def __init__(self, df, transform=None):
        """
        Args:
            df (pd.DataFrame): DataFrame containing 'id', 'target', and 'file_path'.
            transform (albumentations.Compose): Augmentation pipeline.
        """
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full file path
        # Metadata file_path is relative to input_root (e.g., "train/0/xxxx.npy")
        file_path = os.path.join(Config.input_root, row["file_path"])

        # Load spectrogram
        # Expected Shape: (6, 273, 256)
        try:
            img = np.load(file_path).astype(np.float32)
        except FileNotFoundError:
            # Fallback (should not happen with validated metadata)
            img = np.zeros((6, 273, 256), dtype=np.float32)

        # 1. Vertical Stacking
        # Stack the 6 panels vertically to form (1638, 256)
        # This preserves the continuity of Doppler drifting signals across the cadence
        img = np.vstack(img)

        # 2. Instance Normalization
        # Subtract mean and divide by std per sample to handle varying noise floors
        mean = np.mean(img)
        std = np.std(img)
        img = (img - mean) / (std + 1e-6)

        # 3. Prepare for Albumentations
        # Albumentations expects HWC format. Current shape is (H, W). Add channel dim.
        img = img[:, :, np.newaxis]  # (1638, 256, 1)

        # 4. Apply Augmentations
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented["image"]

        # 5. Channel Expansion
        # Replicate the single channel 3 times to match ImageNet pretrained weights (RGB)
        # Shape becomes (1638, 256, 3)
        if img.shape[2] == 1:
            img = np.concatenate([img, img, img], axis=2)

        # 6. Convert to Tensor (HWC -> CHW)
        # Resulting shape: (3, 1638, 256)
        img = torch.from_numpy(img).permute(2, 0, 1)

        # Get Target
        if "target" in row:
            target = torch.tensor(row["target"], dtype=torch.float32)
        else:
            # Default placeholder for test set
            target = torch.tensor(0.5, dtype=torch.float32)

        return img, target


def get_datasets(debug=False):
    """
    Factory function to create train, validation, and test datasets.

    Args:
        debug (bool): If True, subsamples the data for faster debugging.

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    # Load DataFrames from metadata paths defined in Config
    train_df = pd.read_csv(Config.train_csv_path)
    val_df = pd.read_csv(Config.val_csv_path)
    test_df = pd.read_csv(Config.test_csv_path)

    # Debug Mode: Subsample data
    if debug:
        train_df = train_df.sample(
            n=Config.debug_sample_size, random_state=Config.seed
        ).reset_index(drop=True)
        val_df = val_df.sample(
            n=Config.debug_sample_size, random_state=Config.seed
        ).reset_index(drop=True)
        # For test, we take min of length or debug size
        test_df = test_df.sample(
            n=min(len(test_df), Config.debug_sample_size), random_state=Config.seed
        ).reset_index(drop=True)

    # Create Dataset Instances
    train_dataset = SETIDataset(train_df, transform=get_transforms(data="train"))

    val_dataset = SETIDataset(val_df, transform=get_transforms(data="valid"))

    test_dataset = SETIDataset(test_df, transform=get_transforms(data="test"))

    return train_dataset, val_dataset, test_dataset
