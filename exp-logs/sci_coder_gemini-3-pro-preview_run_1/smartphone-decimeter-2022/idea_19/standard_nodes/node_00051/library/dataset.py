import torch
from torch.utils.data import Dataset
import numpy as np
import os
from library.config import Config
from library.data_processor import DataProcessor


class GNSSHeatmapDataset(Dataset):
    """
    PyTorch Dataset for Cyclic Spatio-Temporal 2D ResUNet.

    This dataset loads preprocessed 'Sky Heatmaps' (Time x Azimuth x Channels)
    from the DataProcessor and slices them into fixed-length temporal windows.
    It handles normalization of GNSS features and prepares tensors for the model.

    Attributes:
        split (str): Dataset split ('train', 'val', 'test').
        window_size (int): Temporal length of the windows.
        stride (int): Stride for sliding window generation.
        drives (list): List of processed drive dictionaries loaded from cache.
        window_index (list): Index mapping dataset ID to (drive_idx, start_time_idx).
    """

    def __init__(self, split="train", load_cached_data=True):
        """
        Initialize the dataset.

        Args:
            split (str): One of 'train', 'val', 'test'.
            load_cached_data (bool): If True, attempts to load processed data from cache.
                                     If False or cache missing, re-processes data.
        """
        self.split = split
        self.window_size = Config.WINDOW_SIZE
        self.stride = Config.STRIDE

        # Ensure working directory exists (redundant with Config but safe)
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

        # Initialize DataProcessor and load data
        # DataProcessor handles the heavy lifting of parsing logs, binning azimuths,
        # and creating the 2D heatmap structure. It also handles caching.
        processor = DataProcessor()
        self.drives = processor.process_data(
            split=split, load_cached_data=load_cached_data, save_cache=True
        )

        # Build the index of windows to allow random access via __getitem__
        self.window_index = []
        self._prepare_windows()

        if len(self.drives) > 0:
            print(
                f"[{split.upper()}] Loaded {len(self.drives)} drives, created {len(self.window_index)} windows."
            )
        else:
            print(f"[{split.upper()}] Warning: No drives loaded.")

    def _prepare_windows(self):
        """
        Internal method to slice continuous drives into windows.
        Populates self.window_index with tuples of (drive_idx, start_index).
        """
        for drive_idx, drive_data in enumerate(self.drives):
            num_timestamps = drive_data["features"].shape[0]

            # If a drive is shorter than one window, we treat it as a single window
            # starting at 0 (it will be zero-padded in __getitem__)
            if num_timestamps <= self.window_size:
                self.window_index.append((drive_idx, 0))
                continue

            # Generate sliding windows
            # For training, we use the configured stride (overlap).
            # For testing, we also use the stride to ensure coverage,
            # though inference aggregation logic will be needed downstream.
            current_idx = 0
            while current_idx < num_timestamps:
                self.window_index.append((drive_idx, current_idx))
                current_idx += self.stride

                # Stop if the next start index is beyond the end
                # Note: The last window added might extend beyond num_timestamps;
                # __getitem__ handles this via padding.
                if current_idx >= num_timestamps:
                    break

    def _normalize_features(self, feat_window):
        """
        Normalize input features to a roughly [0, 1] range.

        Feature Channels (from DataProcessor):
        0: Max Cn0DbHz
        1: Mean Cn0DbHz
        2: Mean SvElevationDegrees
        3: SatCount (in bin)
        4: Global Total SatCount
        5: Global Mean RawPseudorangeUncertaintyMeters

        Args:
            feat_window (np.ndarray): Shape (T, Azimuth, Channels)

        Returns:
            np.ndarray: Normalized features.
        """
        # Copy to avoid modifying the cached data in memory
        norm_feat = feat_window.copy()

        # 0, 1: Cn0DbHz (Range ~10-50). Divide by 60.0
        norm_feat[..., 0] = norm_feat[..., 0] / 60.0
        norm_feat[..., 1] = norm_feat[..., 1] / 60.0

        # 2: Elevation (Range 0-90). Divide by 90.0
        norm_feat[..., 2] = norm_feat[..., 2] / 90.0

        # 3: Bin Sat Count (Range 0-5+). Divide by 10.0
        norm_feat[..., 3] = norm_feat[..., 3] / 10.0

        # 4: Global Sat Count (Range 0-40+). Divide by 50.0
        norm_feat[..., 4] = norm_feat[..., 4] / 50.0

        # 5: Uncertainty (Range 0-100s of meters).
        # Use log1p to compress dynamic range, then scale.
        # log(100) ~ 4.6. Dividing by 10 puts it roughly in [0, 1] for common errors.
        norm_feat[..., 5] = np.log1p(norm_feat[..., 5]) / 10.0

        return norm_feat

    def __len__(self):
        return len(self.window_index)

    def __getitem__(self, idx):
        """
        Retrieves a window of data.

        Returns:
            dict: {
                'features': Tensor (Channels, Time, Azimuth),
                'targets': Tensor (Time, 2) [East, North],
                'mask': Tensor (Time,) [1.0 for valid, 0.0 for padding],
                'wls_pos': Tensor (Time, 3) [Lat, Lon, Alt],
                'timestamps': Tensor (Time,) [UnixTimeMillis],
                'drive_idx': int,
                't_start': int
            }
        """
        drive_idx, start_idx = self.window_index[idx]
        drive_data = self.drives[drive_idx]

        # Raw data shapes:
        # features: (Total_Time, Azimuth, Channels)
        # targets: (Total_Time, 2)
        features_all = drive_data["features"]
        total_len = features_all.shape[0]

        # Determine slice length
        valid_len = min(self.window_size, total_len - start_idx)

        # 1. Prepare Features
        # Slice
        feat_slice = features_all[start_idx : start_idx + valid_len]

        # Initialize buffer with zeros (Padding)
        # Shape: (Window, Azimuth, Channels)
        feat_window = np.zeros(
            (self.window_size, feat_slice.shape[1], feat_slice.shape[2]),
            dtype=np.float32,
        )
        feat_window[:valid_len] = feat_slice

        # Normalize
        feat_window = self._normalize_features(feat_window)

        # Permute for PyTorch: (T, K, C) -> (C, T, K)
        # Channels first is standard for Conv2d
        feat_tensor = torch.from_numpy(feat_window.transpose(2, 0, 1))

        # 2. Create Mask
        # 1.0 for valid data, 0.0 for padded data
        mask = np.zeros((self.window_size,), dtype=np.float32)
        mask[:valid_len] = 1.0
        mask_tensor = torch.from_numpy(mask)

        item = {
            "features": feat_tensor,
            "mask": mask_tensor,
            "drive_idx": drive_idx,
            "t_start": start_idx,
        }

        # 3. Prepare Targets (if available)
        if drive_data["targets"] is not None:
            targets_all = drive_data["targets"]
            target_slice = targets_all[start_idx : start_idx + valid_len]

            # Initialize buffer
            target_window = np.zeros((self.window_size, 2), dtype=np.float32)
            target_window[:valid_len] = target_slice

            item["targets"] = torch.from_numpy(target_window)

        # 4. Prepare WLS Positions (for reconstruction)
        # Keep as float64 for precision
        wls_all = drive_data["wls_pos"]
        wls_slice = wls_all[start_idx : start_idx + valid_len]

        wls_window = np.zeros((self.window_size, 3), dtype=np.float64)
        wls_window[:valid_len] = wls_slice
        item["wls_pos"] = torch.from_numpy(wls_window)

        # 5. Prepare Timestamps
        ts_all = drive_data["timestamps"]
        ts_slice = ts_all[start_idx : start_idx + valid_len]

        ts_window = np.zeros((self.window_size,), dtype=np.int64)
        ts_window[:valid_len] = ts_slice
        item["timestamps"] = torch.from_numpy(ts_window)

        return item
