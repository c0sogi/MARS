import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import load_npy
from library.spectrogram_ops import generate_dataset_spectrograms


class VolcanoCNNDataset(Dataset):
    """
    PyTorch Dataset for the Vision Branch (EfficientNet).
    Loads pre-computed Dual-Resolution Spectrograms (20 channels) from .npy files.
    Applies Log-Scaling (log1p) to the target variable 'time_to_eruption'.
    """

    def __init__(self, metadata_df, data_dir, is_test=False):
        """
        Args:
            metadata_df (pd.DataFrame): Metadata containing 'segment_id' and 'time_to_eruption'.
            data_dir (str): Directory path containing the cached .npy spectrogram files.
            is_test (bool): If True, returns dummy targets (0.0).
        """
        self.metadata_df = metadata_df.reset_index(drop=True)
        self.data_dir = data_dir
        self.is_test = is_test

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        row = self.metadata_df.iloc[idx]
        segment_id = int(row["segment_id"])

        # Construct path to cached .npy file
        file_path = os.path.join(self.data_dir, f"{segment_id}.npy")

        # Load Spectrogram
        # Shape: (20, 256, 256) -> (Channels, H, W)
        # Data is already normalized via Global Log-Max Scaling in spectrogram_ops
        try:
            spectrogram = load_npy(file_path)
        except FileNotFoundError:
            # Fallback for missing files (should be handled by generation pipeline)
            # Return silent tensor
            spectrogram = np.zeros(
                (Config.IN_CHANNELS, Config.IMG_SIZE[0], Config.IMG_SIZE[1]),
                dtype=np.float32,
            )

        # Convert to PyTorch Tensor
        img_tensor = torch.from_numpy(spectrogram)

        # Handle Target
        if self.is_test:
            # Test set has no ground truth, return 0.0
            target = 0.0
        else:
            # Apply Log-Scaling to target: log(y + 1)
            # This compresses the dynamic range of time-to-eruption
            raw_target = row["time_to_eruption"]
            target = np.log1p(raw_target)

        target_tensor = torch.tensor(target, dtype=torch.float32)

        return img_tensor, target_tensor


def get_vision_loaders(
    train_df,
    val_df,
    test_df=None,
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    load_cached_data=True,
):
    """
    Factory function to create DataLoaders for the Vision Branch.
    Triggers the generation/caching of spectrograms via spectrogram_ops.

    Args:
        train_df (pd.DataFrame): Training metadata.
        val_df (pd.DataFrame): Validation metadata.
        test_df (pd.DataFrame, optional): Test metadata.
        batch_size (int): Batch size.
        num_workers (int): Number of DataLoader workers.
        load_cached_data (bool): Whether to use existing cached files.

    Returns:
        dict: Dictionary containing 'train', 'val', and optionally 'test' DataLoaders.
    """
    loaders = {}

    # 1. Prepare Training Data
    # Triggers generation/caching of .npy files
    train_dir = generate_dataset_spectrograms(
        train_df, "train", load_cached_data=load_cached_data
    )
    train_dataset = VolcanoCNNDataset(train_df, train_dir, is_test=False)
    loaders["train"] = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # Drop incomplete batch for training stability
    )

    # 2. Prepare Validation Data
    val_dir = generate_dataset_spectrograms(
        val_df, "val", load_cached_data=load_cached_data
    )
    val_dataset = VolcanoCNNDataset(val_df, val_dir, is_test=False)
    loaders["val"] = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # 3. Prepare Test Data (Optional)
    if test_df is not None:
        test_dir = generate_dataset_spectrograms(
            test_df, "test", load_cached_data=load_cached_data
        )
        test_dataset = VolcanoCNNDataset(test_df, test_dir, is_test=True)
        loaders["test"] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )

    return loaders
