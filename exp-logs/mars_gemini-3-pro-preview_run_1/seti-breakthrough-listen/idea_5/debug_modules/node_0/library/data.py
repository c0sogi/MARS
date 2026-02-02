import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library import config, utils


class SETIDataset(Dataset):
    """
    PyTorch Dataset for SETI Technosignature Detection.

    Handles:
    - Loading .npy spectrogram files.
    - Instance-wise normalization (Zero Mean, Unit Variance).
    - Reshaping to (6, 1, 273, 256) for Time-Distributed models.
    - Deterministic augmentation across time panels (Flip, Shift).
    """

    def __init__(
        self, metadata_path, mode="train", debug=False, debug_sample_size=None
    ):
        """
        Args:
            metadata_path (str): Path to the metadata CSV file.
            mode (str): Operation mode ('train', 'val', 'test').
            debug (bool): If True, limits dataset size for debugging.
            debug_sample_size (int): Overrides config.DEBUG_SAMPLE_SIZE if provided.
        """
        self.mode = mode
        self.metadata_path = metadata_path

        # Load metadata
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        self.df = pd.read_csv(metadata_path)

        # Apply Debug Subsampling
        if debug:
            limit = (
                debug_sample_size
                if debug_sample_size is not None
                else config.DEBUG_SAMPLE_SIZE
            )
            if len(self.df) > limit:
                utils.seed_everything(config.SEED)
                self.df = self.df.sample(n=limit, random_state=config.SEED).reset_index(
                    drop=True
                )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct file path
        # Metadata file_path is relative (e.g., "train/0/xxxx.npy")
        file_path = os.path.join(config.INPUT_DIR, row["file_path"])

        # Load Spectrogram
        # Shape: (6, 273, 256) -> (Time-Panel, Frequency, Time-Bin)
        try:
            spectrogram = np.load(file_path).astype(np.float32)
        except Exception as e:
            # Fallback for robustness (should not happen with verified metadata)
            print(f"Error loading {file_path}: {e}")
            spectrogram = np.zeros((6, 273, 256), dtype=np.float32)

        # Instance-wise Normalization
        # Normalize each sample to have mean 0 and std 1
        mean = np.mean(spectrogram)
        std = np.std(spectrogram)
        spectrogram = (spectrogram - mean) / (std + 1e-6)

        # Apply Augmentations (Train only)
        if self.mode == "train":
            spectrogram = self._apply_augmentations(spectrogram)

        # Reshape for Time-Distributed CNN
        # Input: (6, 273, 256)
        # Output: (6, 1, 273, 256) -> (Time, Channel, Height, Width)
        spectrogram = spectrogram[:, np.newaxis, :, :]

        # Convert to Tensor
        data_tensor = torch.tensor(spectrogram, dtype=torch.float32)

        # Get Target
        if self.mode == "test":
            # Test set does not have targets (or has dummy ones)
            target_tensor = torch.tensor(0.0, dtype=torch.float32)
        else:
            target_tensor = torch.tensor(row["target"], dtype=torch.float32)

        return data_tensor, target_tensor

    def _apply_augmentations(self, image):
        """
        Applies augmentations identically across all 6 time panels to preserve
        the relative signal structure (Doppler drift).

        Args:
            image (np.array): Shape (6, 273, 256)

        Returns:
            np.array: Augmented image.
        """
        # 1. Random Vertical Flip (Frequency Axis = 1)
        if np.random.rand() < 0.5:
            image = np.flip(image, axis=1)

        # 2. Random Frequency Shift (Vertical Translation)
        # Shift range: +/- 10% of frequency height
        height = image.shape[1]
        max_shift = int(height * 0.1)
        shift = np.random.randint(-max_shift, max_shift + 1)

        if shift != 0:
            image = np.roll(image, shift, axis=1)

            # Mask the wrapped-around part to avoid artifacts
            if shift > 0:
                # Rolled down: mask top
                image[:, :shift, :] = 0.0
            else:
                # Rolled up: mask bottom
                image[:, shift:, :] = 0.0

        return image


def get_train_dataloader(
    batch_size=config.BATCH_SIZE, debug=config.DEBUG, num_workers=config.NUM_WORKERS
):
    """
    Returns the DataLoader for the training set.
    """
    dataset = SETIDataset(
        metadata_path=config.TRAIN_METADATA, mode="train", debug=debug
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # Important for BatchNorm/GroupNorm stability
    )


def get_val_dataloader(
    batch_size=config.BATCH_SIZE, debug=config.DEBUG, num_workers=config.NUM_WORKERS
):
    """
    Returns the DataLoader for the validation set.
    """
    dataset = SETIDataset(metadata_path=config.VAL_METADATA, mode="val", debug=debug)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


def get_test_dataloader(
    batch_size=config.BATCH_SIZE, debug=config.DEBUG, num_workers=config.NUM_WORKERS
):
    """
    Returns the DataLoader for the test set.
    """
    dataset = SETIDataset(metadata_path=config.TEST_METADATA, mode="test", debug=debug)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
