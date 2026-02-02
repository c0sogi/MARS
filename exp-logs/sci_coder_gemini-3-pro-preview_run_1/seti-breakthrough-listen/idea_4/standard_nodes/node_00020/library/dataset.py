import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config


class SETIDataset(Dataset):
    """
    Custom Dataset for SETI Signal Detection.
    Loads spectrogram snippets, applies normalization, and performs
    time-synchronized augmentations for the Time-Distributed CNN.
    """

    def __init__(self, metadata, mode="train"):
        """
        Args:
            metadata (pd.DataFrame): DataFrame containing 'id', 'file_path', and optionally 'target'.
            mode (str): 'train', 'val', or 'test'. Controls augmentation behavior.
        """
        self.metadata = metadata
        self.mode = mode
        self.input_dir = Config.INPUT_DIR

    def __len__(self):
        return len(self.metadata)

    def _augment_sequence(self, img):
        """
        Applies random vertical frequency shifts and horizontal flips.
        Augmentations are applied identically across all 6 time-steps to preserve
        the relative alignment of the cadence.

        Args:
            img (np.ndarray): Input image of shape (6, 273, 256).

        Returns:
            np.ndarray: Augmented image.
        """
        # Vertical Shift (Frequency Shift)
        # Assuming axis 1 (273) is the vertical dimension of the array.
        h = img.shape[1]
        # Shift range: +/- 15% of height
        max_shift = int(h * 0.15)
        shift = np.random.randint(-max_shift, max_shift)
        img = np.roll(img, shift, axis=1)

        # Horizontal Flip
        # Assuming axis 2 (256) is the horizontal dimension of the array.
        if np.random.rand() < 0.5:
            img = np.flip(img, axis=2)

        return img

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        file_path = os.path.join(self.input_dir, row["file_path"])

        # Load data
        # Data shape: (6, 273, 256)
        try:
            img = np.load(file_path).astype(np.float32)
        except Exception as e:
            # Fallback for robustness
            print(f"Error loading {file_path}: {e}")
            img = np.zeros((6, 273, 256), dtype=np.float32)

        # Instance-wise Normalization
        # Normalize each sample to have zero mean and unit variance
        mean = np.mean(img)
        std = np.std(img)
        img = (img - mean) / (std + 1e-6)

        # Apply Augmentations (Train only)
        if self.mode == "train":
            img = self._augment_sequence(img)

        # Reshape for Time-Distributed CNN
        # Input: (6, 273, 256)
        # Output: (6, 1, 273, 256) -> (Time, Channels, Height, Width)
        img = img[:, np.newaxis, :, :]

        # Convert to Tensor
        # Ensure contiguous memory for efficient tensor conversion
        img = np.ascontiguousarray(img)
        img_tensor = torch.from_numpy(img)

        # Get Target
        if "target" in row:
            target = torch.tensor(row["target"], dtype=torch.float32)
        else:
            # Dummy target for test set
            target = torch.tensor(0.5, dtype=torch.float32)

        return img_tensor, target


def get_dataloaders(
    train_metadata_path=Config.TRAIN_METADATA,
    val_metadata_path=Config.VAL_METADATA,
    test_metadata_path=Config.TEST_METADATA,
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    debug=Config.DEBUG,
    debug_sample_size=Config.DEBUG_SAMPLE_SIZE,
):
    """
    Creates and returns DataLoaders for train, validation, and test sets.

    Args:
        train_metadata_path (str): Path to training metadata CSV.
        val_metadata_path (str): Path to validation metadata CSV.
        test_metadata_path (str): Path to test metadata CSV.
        batch_size (int): Batch size for DataLoaders.
        num_workers (int): Number of worker processes for loading.
        debug (bool): If True, uses a small subset of data.
        debug_sample_size (int): Number of samples to use in debug mode.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load Metadata DataFrames
    train_df = pd.read_csv(train_metadata_path)
    val_df = pd.read_csv(val_metadata_path)
    test_df = pd.read_csv(test_metadata_path)

    # Debug Mode: Subsample data to speed up development loop
    if debug:
        train_df = train_df.sample(
            n=min(len(train_df), debug_sample_size), random_state=Config.SEED
        ).reset_index(drop=True)
        val_df = val_df.sample(
            n=min(len(val_df), debug_sample_size), random_state=Config.SEED
        ).reset_index(drop=True)
        test_df = test_df.sample(
            n=min(len(test_df), debug_sample_size), random_state=Config.SEED
        ).reset_index(drop=True)

    # Create Dataset Instances
    train_dataset = SETIDataset(train_df, mode="train")
    val_dataset = SETIDataset(val_df, mode="val")
    test_dataset = SETIDataset(test_df, mode="test")

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
