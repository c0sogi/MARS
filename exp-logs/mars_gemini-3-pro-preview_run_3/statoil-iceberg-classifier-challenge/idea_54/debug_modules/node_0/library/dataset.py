import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from library.utils import set_seed


class IcebergDataset(Dataset):
    """
    PyTorch Dataset for the Iceberg Classifier task.
    Handles loading from metadata/JSON, caching processed arrays,
    imputing missing angles, and applying augmentations.
    """

    def __init__(
        self, metadata_csv, mode="train", angle_fill_value=None, load_cached_data=True
    ):
        """
        Args:
            metadata_csv (str): Path to the metadata CSV file (e.g., './metadata/train.csv').
            mode (str): Operation mode - 'train', 'val', or 'test'.
            angle_fill_value (float, optional): Value to replace NaN incidence angles.
                                                If None, the median of the current dataset is computed and used.
            load_cached_data (bool): If True, attempts to load processed data from ./working/idea_54/.
        """
        self.mode = mode
        self.metadata = pd.read_csv(metadata_csv)
        self.cache_dir = "./working/idea_54"
        os.makedirs(self.cache_dir, exist_ok=True)

        # Use the filename (without extension) as a unique prefix for cache files
        self.cache_prefix = os.path.splitext(os.path.basename(metadata_csv))[0]

        # Load Data (Images, Angles, Labels, IDs)
        self.images, self.angles, self.labels, self.ids = self._load_data(
            load_cached_data
        )

        # Handle Incidence Angle Imputation
        if angle_fill_value is not None:
            self.angle_fill_value = angle_fill_value
        else:
            # Compute median of valid angles in the current dataset
            valid_angles = self.angles[~np.isnan(self.angles)]
            if len(valid_angles) > 0:
                self.angle_fill_value = float(np.median(valid_angles))
            else:
                self.angle_fill_value = 0.0  # Fallback if all are NaN

        # Fill missing values
        self.angles = np.nan_to_num(self.angles, nan=self.angle_fill_value)

        # Define Augmentations (Only for training)
        self.transform = None
        if self.mode == "train":
            self.transform = torch.nn.Sequential(
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
            )

    def _load_data(self, load_cached_data):
        """
        Internal method to load data from cache or process from raw JSONs.
        """
        p_X = os.path.join(self.cache_dir, f"{self.cache_prefix}_X.npy")
        p_a = os.path.join(self.cache_dir, f"{self.cache_prefix}_angles.npy")
        p_y = os.path.join(self.cache_dir, f"{self.cache_prefix}_labels.npy")
        p_i = os.path.join(self.cache_dir, f"{self.cache_prefix}_ids.npy")

        # Attempt to load from cache
        if (
            load_cached_data
            and os.path.exists(p_X)
            and os.path.exists(p_a)
            and os.path.exists(p_i)
        ):
            # For test set, labels file might not exist
            if self.mode == "test" or os.path.exists(p_y):
                # print(f"Loading cached data for {self.cache_prefix}...")
                X = np.load(p_X)
                angles = np.load(p_a)
                ids = np.load(p_i, allow_pickle=True)
                labels = np.load(p_y) if os.path.exists(p_y) else None
                return X, angles, labels, ids

        # print(f"Processing raw data for {self.cache_prefix}...")

        # Identify unique source files to load (e.g., train.json, test.json)
        unique_files = self.metadata["source_file"].unique()
        raw_cache = {}
        for f_name in unique_files:
            file_path = os.path.join("./input", f_name)
            with open(file_path, "r") as f:
                raw_cache[f_name] = json.load(f)

        n_samples = len(self.metadata)

        # Pre-allocate arrays
        # Images: N x 3 x 75 x 75 (Band 1, Band 2, Avg)
        X = np.zeros((n_samples, 3, 75, 75), dtype=np.float32)
        angles = np.full(n_samples, np.nan, dtype=np.float32)
        ids = np.empty(n_samples, dtype=object)

        has_labels = "is_iceberg" in self.metadata.columns
        labels = np.zeros(n_samples, dtype=np.float32) if has_labels else None

        # Iterate over metadata and extract data
        # Using itertuples for performance
        for i, row in enumerate(self.metadata.itertuples()):
            # Retrieve raw item using source file and original index
            item = raw_cache[row.source_file][row.original_index]

            ids[i] = item["id"]

            # Use metadata's inc_angle (already numeric/NaN)
            angles[i] = row.inc_angle

            if has_labels:
                labels[i] = row.is_iceberg

            # Process Bands
            # Raw data is flattened 5625 floats
            b1 = np.array(item["band_1"], dtype=np.float32).reshape(75, 75)
            b2 = np.array(item["band_2"], dtype=np.float32).reshape(75, 75)

            # Band 3: Average of Band 1 and Band 2
            b3 = (b1 + b2) / 2.0

            X[i, 0] = b1
            X[i, 1] = b2
            X[i, 2] = b3

        # Save to cache
        np.save(p_X, X)
        np.save(p_a, angles)
        np.save(p_i, ids)
        if labels is not None:
            np.save(p_y, labels)

        return X, angles, labels, ids

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """
        Returns:
            If mode != 'test': ((image, angle), label)
            If mode == 'test': ((image, angle), id)
        """
        # Convert to tensor
        x = torch.from_numpy(self.images[idx])  # Shape: (3, 75, 75)
        angle = torch.tensor(self.angles[idx], dtype=torch.float32)

        # Apply augmentations if available (train mode)
        if self.transform:
            x = self.transform(x)

        if self.mode == "test":
            return (x, angle), self.ids[idx]
        else:
            y = torch.tensor(self.labels[idx], dtype=torch.float32)
            return (x, angle), y
