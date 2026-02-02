import os
import numpy as np
import pandas as pd
import torch
import albumentations as A
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import seed_everything


class SETIDataset(Dataset):
    """
    Dataset class for SETI signal detection.
    Loads .npy spectrograms, splits into On-Target/Off-Target streams,
    applies synchronized augmentations, and pads to compatible dimensions.
    """

    def __init__(self, df, transform=None, is_test=False):
        self.df = df
        self.transform = transform
        self.is_test = is_test
        # Calculate padding required to reach Config.IMG_HEIGHT (e.g., 288 - 273 = 15)
        self.pad_h = max(0, Config.IMG_HEIGHT - Config.ORIGINAL_HEIGHT)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # Load spectrogram: Shape (6, 273, 256)
        try:
            image = np.load(file_path).astype(np.float32)
        except Exception as e:
            # Fallback for robustness (should not be triggered with valid metadata)
            image = np.zeros(
                (6, Config.ORIGINAL_HEIGHT, Config.IMG_WIDTH), dtype=np.float32
            )

        # Split into Siamese streams
        # Stream A: On-Target (indices 0, 2, 4) -> Shape: (3, 273, 256)
        # Stream B: Off-Target (indices 1, 3, 5) -> Shape: (3, 273, 256)
        stream_a = image[[0, 2, 4], :, :]
        stream_b = image[[1, 3, 5], :, :]

        # Transpose to (H, W, C) for Albumentations processing
        stream_a = np.transpose(stream_a, (1, 2, 0))
        stream_b = np.transpose(stream_b, (1, 2, 0))

        # Apply synchronized augmentations to both streams
        if self.transform:
            # 'image' is the primary target, 'image_b' is the additional target
            # This ensures geometric transformations (flips) are identical
            augmented = self.transform(image=stream_a, image_b=stream_b)
            stream_a = augmented["image"]
            stream_b = augmented["image_b"]

        # Transpose back to (C, H, W) for PyTorch
        stream_a = np.transpose(stream_a, (2, 0, 1))
        stream_b = np.transpose(stream_b, (2, 0, 1))

        # Pad Height (Frequency axis) to be divisible by 32 (e.g., 273 -> 288)
        # Padding is applied at the end of the frequency axis
        if self.pad_h > 0:
            # np.pad format: ((before_c, after_c), (before_h, after_h), (before_w, after_w))
            pad_width = ((0, 0), (0, self.pad_h), (0, 0))
            stream_a = np.pad(stream_a, pad_width, mode="constant", constant_values=0)
            stream_b = np.pad(stream_b, pad_width, mode="constant", constant_values=0)

        # Convert to PyTorch tensors
        stream_a = torch.from_numpy(stream_a).float()
        stream_b = torch.from_numpy(stream_b).float()

        # Prepare target
        if self.is_test:
            target = torch.tensor(0.5, dtype=torch.float)  # Placeholder for test set
        else:
            target = torch.tensor(row["target"], dtype=torch.float)

        return {"stream_a": stream_a, "stream_b": stream_b}, target


def get_loaders(debug=False, load_cached_data=False):
    """
    Creates and returns DataLoaders for training and validation.

    Args:
        debug (bool): If True, uses a small subset of the data for debugging.
        load_cached_data (bool): Flag included for interface consistency.
                                 Direct file loading is used for this dataset.
    """
    seed_everything(Config.SEED)

    # Load Metadata from CSVs
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)

    if debug:
        train_df = train_df.sample(
            n=min(len(train_df), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
        ).reset_index(drop=True)
        val_df = val_df.sample(
            n=min(len(val_df), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
        ).reset_index(drop=True)

    # Define Augmentations
    # Synchronized Horizontal (Time) and Vertical (Frequency) flips
    train_transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
        ],
        additional_targets={"image_b": "image"},
    )

    # Validation - No augmentation
    val_transform = None

    # Create Datasets
    train_dataset = SETIDataset(train_df, transform=train_transform)
    val_dataset = SETIDataset(val_df, transform=val_transform)

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
    Creates and returns the DataLoader for the test set.
    Used for generating submission predictions.
    """
    test_df = pd.read_csv(Config.TEST_CSV)

    # Test set uses no augmentation
    dataset = SETIDataset(test_df, transform=None, is_test=True)

    loader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return loader
