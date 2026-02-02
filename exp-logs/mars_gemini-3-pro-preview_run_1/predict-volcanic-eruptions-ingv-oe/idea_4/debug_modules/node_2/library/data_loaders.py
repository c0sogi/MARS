import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchaudio.transforms as T
from library.config import Config
from library.feature_engineering import TabularFeatureExtractor
from library.spectrogram_processing import SpectrogramGenerator


class SeismicCNNDataset(Dataset):
    """
    PyTorch Dataset for Seismic Spectrograms.
    Handles converting numpy arrays to tensors and applying augmentations.
    """

    def __init__(self, X, y=None, is_train=False):
        """
        Args:
            X (np.ndarray): Spectrograms of shape (N, 10, H, W).
            y (np.ndarray, optional): Targets of shape (N,).
            is_train (bool): Whether to apply augmentations.
        """
        self.X = X
        self.y = y
        self.is_train = is_train

        # SpecAugment: Time and Frequency Masking
        # Parameters are chosen to be mild to prevent destroying too much signal
        # H (Freq) = 128, W (Time) = 256
        if self.is_train:
            self.time_masking = T.TimeMasking(time_mask_param=30)
            self.freq_masking = T.FrequencyMasking(freq_mask_param=20)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Load spectrogram: (10, H, W)
        # Convert to tensor
        spec = torch.from_numpy(self.X[idx])

        # Apply Augmentations if training
        if self.is_train:
            # torchaudio transforms work on (..., freq, time)
            # Our shape is (Channels, Freq, Time), so it applies per channel correctly
            spec = self.time_masking(spec)
            spec = self.freq_masking(spec)

        if self.y is not None:
            target = torch.tensor(self.y[idx], dtype=torch.float32)
            return spec, target
        else:
            return spec


def get_tabular_data(dataset_type="train", load_cached_data=True):
    """
    Retrieves tabular features for Branch A (LightGBM).

    Args:
        dataset_type (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to use cached parquet files.

    Returns:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series or None): Target variable.
        ids (pd.Series): Segment IDs.
    """
    extractor = TabularFeatureExtractor()
    X, y, ids = extractor.get_features(
        dataset_type=dataset_type, load_cached_data=load_cached_data
    )
    return X, y, ids


def get_spectrogram_loaders(load_cached_data=True):
    """
    Retrieves DataLoaders for Branch B (CNN).

    Args:
        load_cached_data (bool): Whether to use cached numpy files.

    Returns:
        train_loader (DataLoader)
        val_loader (DataLoader)
        test_loader (DataLoader)
        test_ids (np.ndarray): IDs corresponding to test_loader samples.
    """
    generator = SpectrogramGenerator()
    batch_size = Config.CNN_BATCH_SIZE
    num_workers = Config.NUM_WORKERS
    pin_memory = torch.cuda.is_available()

    # 1. Train Set
    X_train, y_train, _ = generator.get_dataset(
        "train", load_cached_data=load_cached_data
    )
    train_dataset = SeismicCNNDataset(X_train, y_train, is_train=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # 2. Validation Set
    X_val, y_val, _ = generator.get_dataset("val", load_cached_data=load_cached_data)
    val_dataset = SeismicCNNDataset(X_val, y_val, is_train=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # 3. Test Set
    X_test, y_test, ids_test = generator.get_dataset(
        "test", load_cached_data=load_cached_data
    )
    test_dataset = SeismicCNNDataset(X_test, y_test, is_train=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader, ids_test
