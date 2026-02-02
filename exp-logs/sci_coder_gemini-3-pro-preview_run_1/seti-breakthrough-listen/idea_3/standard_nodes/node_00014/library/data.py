import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config


# -----------------------------------------------------------------------------
# Augmentations
# -----------------------------------------------------------------------------
class Compose:
    """
    Composes several transforms together.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class RandomHorizontalFlip:
    """
    Randomly flips the input along the frequency axis (axis 2).
    Input shape: (Depth, Height, Width) -> (6, 273, 256)
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            return np.flip(img, axis=2).copy()
        return img


class RandomVerticalFlip:
    """
    Randomly flips the input along the time axis (axis 1).
    Input shape: (Depth, Height, Width) -> (6, 273, 256)
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            return np.flip(img, axis=1).copy()
        return img


class RandomFrequencyShift:
    """
    Randomly shifts the input along the frequency axis (axis 2).
    Input shape: (Depth, Height, Width) -> (6, 273, 256)
    """

    def __init__(self, max_shift=20, p=0.5):
        self.max_shift = max_shift
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            shift = np.random.randint(-self.max_shift, self.max_shift)
            return np.roll(img, shift, axis=2)
        return img


def get_transforms(mode="train"):
    """
    Returns the composition of transforms for the specified mode.
    """
    if mode == "train":
        return Compose(
            [
                RandomHorizontalFlip(p=0.5),
                RandomVerticalFlip(p=0.5),
                RandomFrequencyShift(max_shift=20, p=0.5),
            ]
        )
    else:
        return Compose([])


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
class SETIDataset(Dataset):
    """
    Dataset class for loading cadence snippets.
    Reshapes the (6, 273, 256) input into (1, 6, 273, 256) tensors for 3D CNNs.
    """

    def __init__(self, metadata_df, mode="train", transform=None):
        self.metadata = metadata_df
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        file_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # Load raw spectrogram: Shape (6, 273, 256)
        # Data is stored as float16, convert to float32
        try:
            data = np.load(file_path).astype(np.float32)
        except FileNotFoundError:
            # Handle missing files gracefully (e.g., return zeros)
            data = np.zeros((6, 273, 256), dtype=np.float32)

        # 1. Normalization: Instance Standardization
        mean = np.mean(data)
        std = np.std(data)
        if std > 1e-6:
            data = (data - mean) / std
        else:
            data = data - mean

        # 2. Augmentation (Train only)
        if self.transform:
            data = self.transform(data)

        # 3. Format for 3D CNN: (C, D, H, W) -> (1, 6, 273, 256)
        tensor = torch.from_numpy(data).unsqueeze(0)

        if self.mode == "test":
            return tensor, row["id"]
        else:
            # Target is float for BCEWithLogitsLoss
            return tensor, torch.tensor(row["target"], dtype=torch.float)


# -----------------------------------------------------------------------------
# Data Loaders
# -----------------------------------------------------------------------------
def get_dataloaders(
    train_metadata_path=Config.TRAIN_METADATA,
    val_metadata_path=Config.VAL_METADATA,
    test_metadata_path=Config.TEST_METADATA,
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    debug=False,
    debug_subset_size=Config.DEBUG_SUBSET_SIZE,
):
    """
    Creates and returns DataLoaders for train, validation, and test sets.
    """
    # Load Metadata
    train_df = pd.read_csv(train_metadata_path)
    val_df = pd.read_csv(val_metadata_path)
    test_df = pd.read_csv(test_metadata_path)

    # Debug Mode: Subset data
    if debug:
        train_df = train_df.iloc[:debug_subset_size]
        val_df = val_df.iloc[:debug_subset_size]
        # We generally keep the test set intact or handle it separately,
        # but for full pipeline debug, it can be left as is or subsetted if desired.
        # Here we leave test set full to ensure submission format verification works.

    # Get Transforms
    train_transform = get_transforms(mode="train")
    val_transform = get_transforms(mode="val")
    test_transform = get_transforms(mode="test")

    # Create Datasets
    train_dataset = SETIDataset(train_df, mode="train", transform=train_transform)
    val_dataset = SETIDataset(val_df, mode="val", transform=val_transform)
    test_dataset = SETIDataset(test_df, mode="test", transform=test_transform)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
