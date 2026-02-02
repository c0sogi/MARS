import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config, load_dicom_volume, get_indices


class BraTSDataset(Dataset):
    def __init__(
        self, metadata_path, cache_name=None, load_cached_data=True, is_train=True
    ):
        """
        Dataset class for BraTS21 Glioblastoma classification.

        Args:
            metadata_path (str): Path to the parquet metadata file.
            cache_name (str, optional): Prefix for cache files.
            load_cached_data (bool): Whether to try loading from cache.
            is_train (bool): Whether to load targets (y).
        """
        self.df = pd.read_parquet(metadata_path)
        self.is_train = is_train

        # Ensure working directory exists
        os.makedirs(Config.CACHE_DIR, exist_ok=True)

        self.cache_path_X = (
            os.path.join(Config.CACHE_DIR, f"{cache_name}_X.npy")
            if cache_name
            else None
        )
        self.cache_path_y = (
            os.path.join(Config.CACHE_DIR, f"{cache_name}_y.npy")
            if cache_name
            else None
        )
        self.cache_path_ids = (
            os.path.join(Config.CACHE_DIR, f"{cache_name}_ids.npy")
            if cache_name
            else None
        )

        self.X = None
        self.y = None
        self.ids = None

        if load_cached_data and self._check_cache():
            print(f"Loading cached data from {Config.CACHE_DIR}...")
            self.X = np.load(self.cache_path_X)
            self.ids = np.load(self.cache_path_ids, allow_pickle=True)
            if self.is_train:
                self.y = np.load(self.cache_path_y)
        else:
            print(f"Processing data from {metadata_path}...")
            self._process_and_cache()

    def _check_cache(self):
        """Checks if all required cache files exist."""
        if not self.cache_path_X:
            return False
        if not os.path.exists(self.cache_path_X):
            return False
        if not os.path.exists(self.cache_path_ids):
            return False
        if self.is_train and not os.path.exists(self.cache_path_y):
            return False
        return True

    def _process_and_cache(self):
        """Loads DICOMs, processes them into tensors, and saves to cache."""
        X_list = []
        y_list = []
        ids_list = []

        for idx, row in self.df.iterrows():
            bra_id = row["BraTS21ID"]

            # Load volumes for each modality using helper from config
            # Paths in metadata are relative to input dir
            paths_flair = row.get("flair_paths", [])
            paths_t1 = row.get("t1w_paths", [])
            paths_t1ce = row.get("t1wce_paths", [])
            paths_t2 = row.get("t2w_paths", [])

            # load_dicom_volume handles resizing and global normalization
            vol_flair = load_dicom_volume(paths_flair, Config.IMG_SIZE)
            vol_t1 = load_dicom_volume(paths_t1, Config.IMG_SIZE)
            vol_t1ce = load_dicom_volume(paths_t1ce, Config.IMG_SIZE)
            vol_t2 = load_dicom_volume(paths_t2, Config.IMG_SIZE)

            # Construct 128-channel input
            # We want interleaved slices: [F_0, T1_0, T1c_0, T2_0, F_1, ...]
            channels = []

            # Process each modality
            # Note: All volumes for a patient are assumed to be registered or
            # at least we sample based on the specific volume's depth.
            for vol in [vol_flair, vol_t1, vol_t1ce, vol_t2]:
                # get_indices selects uniformly from 10-90% depth
                indices = get_indices(vol.shape[0], Config.NUM_SLICES)
                selected_slices = vol[indices]  # Shape: (32, 256, 256)
                channels.append(selected_slices)

            # Stack to (4, 32, 256, 256) -> Modality, Depth, H, W
            stacked = np.stack(channels, axis=0)

            # Transpose to (32, 4, 256, 256) -> Depth, Modality, H, W
            # This ensures that when we flatten, we get the interleaved order
            stacked = stacked.transpose(1, 0, 2, 3)

            # Reshape to (128, 256, 256) -> (Depth*Modality, H, W)
            combined = stacked.reshape(-1, Config.IMG_SIZE, Config.IMG_SIZE)

            X_list.append(combined)
            ids_list.append(bra_id)
            if self.is_train:
                y_list.append(row["MGMT_value"])

        # Convert to numpy arrays
        self.X = np.array(X_list, dtype=np.float32)
        self.ids = np.array(ids_list)
        if self.is_train:
            self.y = np.array(y_list, dtype=np.float32)

        # Save to cache
        if self.cache_path_X:
            np.save(self.cache_path_X, self.X)
            np.save(self.cache_path_ids, self.ids)
            if self.is_train:
                np.save(self.cache_path_y, self.y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.is_train:
            y = self.y[idx]
            return torch.tensor(x), torch.tensor(y).float()
        else:
            return torch.tensor(x), self.ids[idx]
