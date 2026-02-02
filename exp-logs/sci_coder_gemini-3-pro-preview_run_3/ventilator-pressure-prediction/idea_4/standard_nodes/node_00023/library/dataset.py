import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from typing import Optional, Dict, Tuple, Union

from library.config import Config, set_seed
from library.features import load_and_preprocess_data, get_all_feature_names


class VentilatorDataset(Dataset):
    """
    PyTorch Dataset for the Ventilator Pressure Prediction task.
    Provides a unified feature tensor for the monolithic LSTM.
    """

    def __init__(
        self, X: np.ndarray, y: Optional[np.ndarray] = None, is_test: bool = False
    ):
        """
        Args:
            X (np.ndarray): Input features of shape (num_samples, 80, num_features).
            y (np.ndarray, optional): Target values of shape (num_samples, 80).
            is_test (bool): Flag indicating if this is the test set (no targets).
        """
        self.X = X
        self.y = y
        self.is_test = is_test

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(
        self, idx: int
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Extract the 80-step time series for this breath
        # shape: (80, num_features)
        x = torch.tensor(self.X[idx], dtype=torch.float32)

        if self.is_test:
            return x

        # Return targets if available
        # shape: (80,)
        target = torch.tensor(self.y[idx], dtype=torch.float32)
        return x, target


class DataManager:
    """
    Manages data loading, caching, and DataLoader creation.
    Strictly handles cache invalidation to ensure feature consistency.
    """

    def __init__(self):
        # Ensure working directory exists
        Config.setup()

    def clear_cache(self):
        """
        Deletes all cached .npy and scaler files to force regeneration.
        """
        files_to_remove = [
            Config.TRAIN_CACHE_X,
            Config.TRAIN_CACHE_Y,
            Config.VAL_CACHE_X,
            Config.VAL_CACHE_Y,
            Config.TEST_CACHE_X,
            Config.SCALER_PATH,
        ]

        print("Clearing data cache...")
        for file_path in files_to_remove:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except OSError as e:
                    print(f"Error deleting {file_path}: {e}")

    def get_dataloader(
        self,
        split: str,
        batch_size: int = Config.BATCH_SIZE,
        shuffle: bool = False,
        load_cached_data: bool = True,
    ) -> DataLoader:
        """
        Prepares the DataLoader for a specific split.

        Args:
            split (str): 'train', 'validation', or 'test'.
            batch_size (int): Batch size for the DataLoader.
            shuffle (bool): Whether to shuffle the data.
            load_cached_data (bool): If False, clears cache and regenerates data.

        Returns:
            DataLoader: PyTorch DataLoader containing the VentilatorDataset.
        """
        # Strict cache invalidation logic
        if not load_cached_data:
            # If we are not loading cached data, we must clear the specific cache
            # for this split (or all, to be safe regarding scaler consistency).
            # Here we rely on load_and_preprocess_data's internal regeneration logic,
            # but we can explicitly clear to be safe.
            if split == "train":
                # If regenerating train, we must invalidate everything because scaler changes
                self.clear_cache()
            else:
                # For val/test, just ensuring the specific file is refreshed is handled
                # by passing load_cached_data=False to the loader function.
                pass

        # Load and preprocess data (delegated to library.features)
        X, y = load_and_preprocess_data(split, load_cached_data=load_cached_data)

        # Debugging: Subset data if configured
        if Config.DEBUG:
            print(
                f"DEBUG MODE: Truncating {split} data to {Config.DEBUG_SAMPLE_SIZE} samples."
            )
            X = X[: Config.DEBUG_SAMPLE_SIZE]
            if y is not None:
                y = y[: Config.DEBUG_SAMPLE_SIZE]

        # Create Dataset
        is_test = split == "test"
        dataset = VentilatorDataset(X, y, is_test=is_test)

        # Create DataLoader
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True if torch.cuda.is_available() else False,
            drop_last=(
                split == "train"
            ),  # Drop last incomplete batch only for training
        )

        return loader


# Helper function to ensure seed is set when module is imported/used
set_seed(Config.SEED)
