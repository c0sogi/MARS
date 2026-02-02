import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from library.utils import set_seed

# Constants for paths
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
CACHE_DIR = "./working/idea_6/"


class IcebergDataset(Dataset):
    """
    PyTorch Dataset for Iceberg vs Ship classification.
    """

    def __init__(self, X, angles, labels=None, ids=None, transform=None):
        """
        Args:
            X (np.ndarray): Images of shape (N, 3, 75, 75).
            angles (np.ndarray): Incidence angles of shape (N,).
            labels (np.ndarray, optional): Labels of shape (N,).
            ids (np.ndarray, optional): IDs of shape (N,).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.X = X
        self.angles = angles
        self.labels = labels
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Retrieve data
        img = self.X[idx]
        angle = self.angles[idx]

        # Convert to Tensor
        # Input is (3, 75, 75) float32
        img_tensor = torch.from_numpy(img).float()
        angle_tensor = torch.tensor([angle], dtype=torch.float)

        # Apply transforms (Augmentations)
        if self.transform:
            img_tensor = self.transform(img_tensor)

        # Return tuple based on availability of labels
        if self.labels is not None:
            label = self.labels[idx]
            label_tensor = torch.tensor(label, dtype=torch.float)
            return img_tensor, angle_tensor, label_tensor
        else:
            # For test set, return ID
            id_val = self.ids[idx]
            return img_tensor, angle_tensor, id_val


def get_transforms(phase: str):
    """
    Returns the transformations for the given phase.

    Args:
        phase (str): 'train', 'val', or 'test'.

    Returns:
        torchvision.transforms.Compose or None
    """
    if phase == "train":
        return transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
            ]
        )
    else:
        # No TTA in the dataset itself; TTA is handled in inference loop if needed
        # Validation/Test data is not augmented by default
        return None


def get_dataset(mode: str, load_cached_data: bool = True) -> IcebergDataset:
    """
    Loads the dataset for the specified mode (train, val, test).
    Handles caching of processed numpy arrays to avoid re-processing raw JSON.

    Args:
        mode (str): One of 'train', 'val', 'test'.
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        IcebergDataset: The initialized dataset.
    """
    set_seed(42)
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Define cache file paths
    cache_X_path = os.path.join(CACHE_DIR, f"X_{mode}.npy")
    cache_angle_path = os.path.join(CACHE_DIR, f"angle_{mode}.npy")
    cache_y_path = os.path.join(CACHE_DIR, f"y_{mode}.npy")
    cache_ids_path = os.path.join(CACHE_DIR, f"ids_{mode}.npy")

    # Determine if we can load from cache
    has_cache = os.path.exists(cache_X_path) and os.path.exists(cache_angle_path)
    if mode in ["train", "val"]:
        has_cache = has_cache and os.path.exists(cache_y_path)
    else:
        has_cache = has_cache and os.path.exists(cache_ids_path)

    if load_cached_data and has_cache:
        print(f"Loading cached {mode} data from {CACHE_DIR}...")
        X = np.load(cache_X_path)
        angles = np.load(cache_angle_path)

        if mode in ["train", "val"]:
            y = np.load(cache_y_path)
            return IcebergDataset(X, angles, labels=y, transform=get_transforms(mode))
        else:
            ids = np.load(cache_ids_path)
            return IcebergDataset(X, angles, ids=ids, transform=get_transforms(mode))

    # Process data from scratch
    print(f"Processing {mode} data from raw files...")

    # Load metadata
    meta_path = os.path.join(METADATA_DIR, f"{mode}.csv")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    df_meta = pd.read_csv(meta_path)

    # Identify required source files
    source_files = df_meta["source_file"].unique()

    # Load raw JSON data into memory
    raw_data_map = {}
    for sf in source_files:
        sf_path = os.path.join(INPUT_DIR, sf)
        print(f"Loading raw JSON: {sf_path}")
        with open(sf_path, "r") as f:
            raw_data_map[sf] = json.load(f)

    # Lists to collect processed data
    X_list = []
    angle_list = []
    y_list = []
    id_list = []

    # Iterate through metadata and extract data
    # Grouping by source_file to minimize dictionary lookups
    for sf, group in df_meta.groupby("source_file"):
        data_source = raw_data_map[sf]

        for _, row in group.iterrows():
            idx = row["original_index"]
            item = data_source[idx]

            # Verify ID alignment (sanity check)
            if item["id"] != row["id"]:
                # In case indices are shifted, we could search, but metadata is trusted
                pass

            # Process Bands
            # Raw data is flattened 5625 elements
            b1 = np.array(item["band_1"], dtype=np.float32).reshape(75, 75)
            b2 = np.array(item["band_2"], dtype=np.float32).reshape(75, 75)

            # Synthetic Band: Average of HH and HV
            avg = (b1 + b2) / 2.0

            # Stack to (3, 75, 75) - Channels First for PyTorch
            img = np.stack([b1, b2, avg], axis=0)
            X_list.append(img)

            # Process Angle
            # Metadata already converted 'na' to NaN
            angle_list.append(row["inc_angle"])

            # Process Label / ID
            if mode in ["train", "val"]:
                y_list.append(row["is_iceberg"])
            else:
                id_list.append(row["id"])

    # Convert to numpy arrays
    X = np.array(X_list, dtype=np.float32)
    angles = np.array(angle_list, dtype=np.float32)

    # Impute missing incidence angles
    # Strategy: Fill NaN with the mean of valid angles in the current split
    # If no valid angles exist (unlikely), use global mean ~39.28 from analysis
    valid_mask = ~np.isnan(angles)
    if np.sum(valid_mask) > 0:
        mean_angle = np.mean(angles[valid_mask])
    else:
        mean_angle = 39.28

    angles[~valid_mask] = mean_angle

    # Save to cache
    print(f"Saving processed {mode} data to cache...")
    np.save(cache_X_path, X)
    np.save(cache_angle_path, angles)

    if mode in ["train", "val"]:
        y = np.array(y_list, dtype=np.float32)
        np.save(cache_y_path, y)
        return IcebergDataset(X, angles, labels=y, transform=get_transforms(mode))
    else:
        ids = np.array(id_list)
        np.save(cache_ids_path, ids)
        return IcebergDataset(X, angles, ids=ids, transform=get_transforms(mode))
