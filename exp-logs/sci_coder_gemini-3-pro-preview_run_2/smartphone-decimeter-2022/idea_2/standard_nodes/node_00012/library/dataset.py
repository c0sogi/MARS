import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from library.config import Config
from library import preprocessing


class GNSSWindowDataset(Dataset):
    def __init__(self, mode="train", load_cached_data=True):
        """
        PyTorch Dataset for GNSS data with sliding windows.

        This dataset creates sliding windows over the time-series data grouped by trip.
        It ensures that windows do not cross trip boundaries by padding with zeros
        where data from the same trip is not available (e.g., at the start/end of a trip).

        Args:
            mode (str): 'train', 'val', or 'test'. Defines which metadata to load.
            load_cached_data (bool): If True, tries to load processed parquet files from cache.
                                     If False or file missing, re-processes raw data.
        """
        self.mode = mode
        self.window_size = Config.WINDOW_SIZE
        self.half_window = self.window_size // 2

        # 1. Load Data using the preprocessing library
        # This handles caching of the dataframe itself
        self.df = preprocessing.load_dataset(
            mode=mode, load_cached_data=load_cached_data
        )

        # 2. Feature Scaling
        # For training data, we fit the scaler. For others, we apply existing stats.
        # Note: In a proper pipeline, fit_scaler should be called explicitly on train set
        # before creating val/test datasets. We assume this order is respected or stats exist.
        if mode == "train":
            preprocessing.fit_scaler(self.df)

        self.df = preprocessing.transform_data(self.df)

        # 3. Prepare Data for Fast Access
        # Convert DataFrame columns to numpy arrays to avoid overhead during __getitem__
        self.features = self.df[Config.INPUT_FEATURES].astype(np.float32).values

        # Handle targets
        if mode in ["train", "val"]:
            # For train/val, we have ground truth deltas
            self.targets = self.df[Config.TARGET_COLUMNS].astype(np.float32).values
        else:
            # For test, create dummy targets (will not be used for loss)
            self.targets = np.zeros(
                (len(self.df), len(Config.TARGET_COLUMNS)), dtype=np.float32
            )

        # 4. Trip Boundary Management
        # We encode tripId to integers to allow fast integer comparison in __getitem__
        # trips are contiguous blocks in the dataframe because load_dataset sorts them.
        self.trip_ids = self.df["tripId"].astype("category").cat.codes.values

        print(
            f"[{mode.upper()}] Dataset ready. Size: {len(self.df)}, Window: {self.window_size}, Features: {self.features.shape[1]}"
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Returns a window of features and the target for the central timestamp.

        Returns:
            x (torch.Tensor): Shape (WINDOW_SIZE, num_features)
            y (torch.Tensor): Shape (num_targets,)
        """
        # Define the window range centered at idx
        # We want the target to correspond to the feature vector at sequence index `half_window`
        start_idx = idx - self.half_window
        end_idx = start_idx + self.window_size

        # Identify the trip this sample belongs to
        current_trip = self.trip_ids[idx]

        # Initialize the sequence buffer with zeros (padding)
        sequence = np.zeros(
            (self.window_size, self.features.shape[1]), dtype=np.float32
        )

        # Determine the valid range within the global dataframe
        # Clip to dataframe bounds
        valid_start = max(0, start_idx)
        valid_end = min(len(self.df), end_idx)

        # Check trip consistency within the valid range
        # We extract the trip_ids for the candidate slice
        candidate_trip_ids = self.trip_ids[valid_start:valid_end]

        # Create a mask of where the slice matches the current trip
        # Since trips are contiguous, this mask will form a single block of True values
        mask = candidate_trip_ids == current_trip

        if np.all(mask):
            # Optimization: If the whole slice belongs to the trip, copy directly
            # Calculate where this slice fits into the sequence buffer
            seq_start = valid_start - start_idx
            seq_end = seq_start + (valid_end - valid_start)

            sequence[seq_start:seq_end] = self.features[valid_start:valid_end]
        else:
            # Boundary case: The window crosses into another trip or out of bounds
            # We only copy the data that belongs to the current trip
            match_indices = np.flatnonzero(mask)

            if len(match_indices) > 0:
                # Get relative start/end within the candidate slice
                rel_first = match_indices[0]
                rel_last = match_indices[-1] + 1  # +1 for exclusive upper bound

                # Map back to global dataframe indices
                copy_start = valid_start + rel_first
                copy_end = valid_start + rel_last

                # Map to sequence buffer indices
                seq_start = copy_start - start_idx
                seq_end = copy_end - start_idx

                sequence[seq_start:seq_end] = self.features[copy_start:copy_end]

        # Convert to PyTorch tensors
        x_tensor = torch.from_numpy(sequence)
        y_tensor = torch.from_numpy(self.targets[idx])

        return x_tensor, y_tensor
