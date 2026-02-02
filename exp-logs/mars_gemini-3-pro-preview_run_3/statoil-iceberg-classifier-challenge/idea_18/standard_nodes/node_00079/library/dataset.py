import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from library.utils import seed_everything


class IcebergDataset(Dataset):
    """
    Custom Dataset for Iceberg vs Ship classification.
    """

    def __init__(self, X, angles, y_or_ids, transform=None, mode="train"):
        """
        Args:
            X (np.ndarray): Image data of shape (N, 75, 75, 3).
            angles (np.ndarray): Incidence angles of shape (N,).
            y_or_ids (np.ndarray or list): Labels (0/1) for train/val, or IDs for test.
            transform (callable, optional): Optional transform to be applied on a sample.
            mode (str): 'train', 'val', or 'test'.
        """
        self.X = X
        self.angles = angles
        self.y_or_ids = y_or_ids
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Retrieve data
        img = self.X[idx]  # Shape: (75, 75, 3)
        angle = self.angles[idx]
        target = self.y_or_ids[idx]

        # Convert to Tensor and permute to (C, H, W)
        # Input is float32 (dB values), so we keep it as float
        img_tensor = torch.from_numpy(img).permute(2, 0, 1)  # Shape: (3, 75, 75)

        # Apply transforms if any
        if self.transform:
            img_tensor = self.transform(img_tensor)

        # Convert angle to tensor
        angle_tensor = torch.tensor(angle, dtype=torch.float32)

        if self.mode == "test":
            # For test, target is the ID string
            return img_tensor, angle_tensor, target
        else:
            # For train/val, target is the label (0 or 1)
            target_tensor = torch.tensor(target, dtype=torch.float32)
            return img_tensor, angle_tensor, target_tensor


def get_transforms(phase: str):
    """
    Returns transformations for the given phase.

    Args:
        phase (str): 'train', 'val', or 'test'.
    """
    if phase == "train":
        return transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
            ]
        )
    else:
        # No augmentation for validation or test
        return transforms.Compose([])


def load_data(mode, load_cached_data=True, sample_size=None):
    """
    Loads data for the specified mode. Checks cache first, processes from scratch if needed.

    Args:
        mode (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.
        sample_size (int, optional): If provided, returns a subset of the data.

    Returns:
        tuple: (X, angles, y_or_ids)
    """
    seed_everything(42)

    cache_dir = "./working/optimized"
    os.makedirs(cache_dir, exist_ok=True)

    # Define cache file paths
    x_path = os.path.join(cache_dir, f"X_{mode}.npy")
    a_path = os.path.join(cache_dir, f"angles_{mode}.npy")
    y_path = os.path.join(cache_dir, f"y_{mode}.npy")
    ids_path = os.path.join(cache_dir, f"ids_{mode}.npy")

    target_path = ids_path if mode == "test" else y_path

    # Attempt to load from cache
    if (
        load_cached_data
        and os.path.exists(x_path)
        and os.path.exists(a_path)
        and os.path.exists(target_path)
    ):
        # print(f"Loading {mode} data from cache...")
        X = np.load(x_path)
        angles = np.load(a_path)
        y_or_ids = np.load(target_path)

        if mode == "test":
            # IDs are saved as numpy array of strings
            pass
    else:
        # print(f"Processing {mode} data from raw files...")

        # 1. Load Metadata
        meta_path = f"./metadata/{mode}.csv"
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        df_meta = pd.read_csv(meta_path)

        # 2. Determine Angle Median from Training Data (for imputation)
        # We always use the training set median to avoid leakage
        train_meta_path = "./metadata/train.csv"
        df_train_meta = pd.read_csv(train_meta_path)
        angle_median = df_train_meta["inc_angle"].median()

        # 3. Load Raw JSON Data
        # Identify which source files we need to read
        source_files = df_meta["source_file"].unique()
        raw_data_map = {}

        for sf in source_files:
            file_path = os.path.join("./input", sf)
            with open(file_path, "r") as f:
                raw_data_map[sf] = json.load(f)

        # 4. Process Data
        num_samples = len(df_meta)
        X = np.zeros((num_samples, 75, 75, 3), dtype=np.float32)
        angles = np.zeros(num_samples, dtype=np.float32)

        if mode == "test":
            y_or_ids = []
        else:
            y_or_ids = np.zeros(num_samples, dtype=np.float32)

        for i, row in df_meta.iterrows():
            src_file = row["source_file"]
            original_idx = row["original_index"]

            # Retrieve raw sample
            # raw_data_map[src_file] is a list of dicts
            sample = raw_data_map[src_file][original_idx]

            # Process Bands
            # Raw bands are flattened lists of 5625 floats
            band_1 = np.array(sample["band_1"], dtype=np.float32).reshape(75, 75)
            band_2 = np.array(sample["band_2"], dtype=np.float32).reshape(75, 75)

            # Create 3rd Channel: Average of Band 1 and Band 2
            band_3 = (band_1 + band_2) / 2.0

            # Stack channels (H, W, C)
            X[i, :, :, 0] = band_1
            X[i, :, :, 1] = band_2
            X[i, :, :, 2] = band_3

            # Process Angle
            inc_angle = row["inc_angle"]
            if pd.isna(inc_angle):
                angles[i] = angle_median
            else:
                angles[i] = inc_angle

            # Process Label/ID
            if mode == "test":
                y_or_ids.append(row["id"])
            else:
                y_or_ids[i] = row["is_iceberg"]

        if mode == "test":
            y_or_ids = np.array(y_or_ids)

        # 5. Save to Cache
        np.save(x_path, X)
        np.save(a_path, angles)
        np.save(target_path, y_or_ids)

    # Handle sample size for debugging
    if sample_size is not None and sample_size < len(X):
        X = X[:sample_size]
        angles = angles[:sample_size]
        y_or_ids = y_or_ids[:sample_size]

    return X, angles, y_or_ids
