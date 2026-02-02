import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio.transforms as T
import numpy as np
import os
from library.config import Config
from library.preprocessing import EEGPreprocessor


class EEGDataset(Dataset):
    def __init__(self, raw_data, spec_data, targets=None, mode="train"):
        """
        PyTorch Dataset for the Dual-Stream Hybrid Model.

        Args:
            raw_data (np.ndarray): Raw EEG signals. Shape (N, 2500, 19).
            spec_data (np.ndarray): Spectrograms. Shape (N, 19, 64, 256).
            targets (np.ndarray, optional): Probability targets. Shape (N, 6).
            mode (str): 'train', 'val', or 'test'. Controls augmentation.
        """
        self.raw_data = raw_data
        self.spec_data = spec_data
        self.targets = targets
        self.mode = mode

        # SpecAugment Transforms for Training
        # Applied to shape (..., Freq, Time) -> (..., 64, 256)
        if self.mode == "train":
            self.freq_masking = T.FrequencyMasking(freq_mask_param=10)
            self.time_masking = T.TimeMasking(time_mask_param=20)

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        # 1. Raw Stream
        # Input stored as (Time, Channels) -> (2500, 19)
        # Model (Conv1d) expects (Channels, Time) -> (19, 2500)
        raw_sample = self.raw_data[idx]
        raw_tensor = torch.tensor(raw_sample, dtype=torch.float32).permute(1, 0)

        # 2. Spectrogram Stream
        # Input stored as (Channels, Freq, Time) -> (19, 64, 256)
        spec_sample = self.spec_data[idx]
        spec_tensor = torch.tensor(spec_sample, dtype=torch.float32)

        # Apply SpecAugment during training
        if self.mode == "train":
            # Torchaudio masking works on the last two dimensions (Freq, Time)
            # Input shape (19, 64, 256) works correctly (broadcast or channel-wise independent)
            spec_tensor = self.freq_masking(spec_tensor)
            spec_tensor = self.time_masking(spec_tensor)

        # 3. Targets
        if self.targets is not None:
            target = torch.tensor(self.targets[idx], dtype=torch.float32)
            return raw_tensor, spec_tensor, target
        else:
            return raw_tensor, spec_tensor


def get_dataloaders(
    debug=Config.DEBUG,
    load_cached=True,
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
):
    """
    Prepares DataLoaders for training and validation.

    Args:
        debug (bool): If True, uses a small subset of data.
        load_cached (bool): If True, attempts to load pre-processed data from disk.
        batch_size (int): Batch size.
        num_workers (int): Number of workers for DataLoader.

    Returns:
        tuple: (train_loader, val_loader)
    """
    preprocessor = EEGPreprocessor()

    # --- Load Training Data ---
    # Note: get_dataset handles caching logic internally
    train_raw, train_spec, train_targets = preprocessor.get_dataset(
        metadata_path=Config.TRAIN_CSV,
        cache_data_path=Config.CACHE_TRAIN_DATA,
        cache_target_path=Config.CACHE_TRAIN_TARGETS,
        mode="train",
        load_cached=load_cached,
        debug=debug,
    )

    # --- Load Validation Data ---
    val_raw, val_spec, val_targets = preprocessor.get_dataset(
        metadata_path=Config.VAL_CSV,
        cache_data_path=Config.CACHE_VAL_DATA,
        cache_target_path=Config.CACHE_VAL_TARGETS,
        mode="val",
        load_cached=load_cached,
        debug=debug,
    )

    # --- Create Datasets ---
    train_dataset = EEGDataset(train_raw, train_spec, train_targets, mode="train")

    val_dataset = EEGDataset(val_raw, val_spec, val_targets, mode="val")

    # --- Create Loaders ---
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch for stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def get_test_dataloader(
    debug=Config.DEBUG,
    load_cached=True,
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
):
    """
    Prepares DataLoader for testing/inference.
    """
    preprocessor = EEGPreprocessor()

    # Cache path for test data (no targets)
    test_cache_path = os.path.join(Config.WORKING_DIR, "test_data.npy")

    test_raw, test_spec, _ = preprocessor.get_dataset(
        metadata_path=Config.TEST_CSV,
        cache_data_path=test_cache_path,
        cache_target_path=None,
        mode="test",
        load_cached=load_cached,
        debug=debug,
    )

    test_dataset = EEGDataset(test_raw, test_spec, targets=None, mode="test")

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return test_loader
