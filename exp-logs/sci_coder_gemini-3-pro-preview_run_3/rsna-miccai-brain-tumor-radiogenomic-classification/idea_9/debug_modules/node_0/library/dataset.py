import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config
from library.utils import load_dicom_volume, global_normalize, uniform_sample_indices


class MGMTDataset(Dataset):
    """
    PyTorch Dataset for MGMT Promoter Methylation prediction.
    Accesses pre-processed in-memory numpy arrays.
    """

    def __init__(self, X, y=None, ids=None):
        self.X = X
        self.y = y
        self.ids = ids

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Data is already float32 and shape (C, H, W)
        img = self.X[idx]
        img_tensor = torch.from_numpy(img)

        # Return label if available
        if self.y is not None:
            label = torch.tensor(self.y[idx], dtype=torch.float32)
            return img_tensor, label

        # For test set, return a placeholder label (-1.0) to maintain (x, y) signature
        return img_tensor, torch.tensor(-1.0, dtype=torch.float32)


def process_patient_data(row):
    """
    Process a single patient's data: load 4 modalities, normalize, sample, and stack.
    Returns a (128, 256, 256) numpy array.
    """
    modalities = ["flair", "t1w", "t1wce", "t2w"]
    patient_slices = []

    for mod in modalities:
        path_col = f"{mod}_paths"
        paths = row[path_col]

        # 1. Load Volume
        # Paths are relative, utils.load_dicom_volume handles joining with INPUT_DIR
        if paths is None:
            paths = []

        volume = load_dicom_volume(paths, image_size=Config.IMAGE_SIZE)

        # 2. Global Volumetric Normalization
        # Done before sampling to preserve context of the full scan
        volume = global_normalize(volume)

        # 3. High-Density Uniform Sampling
        current_depth = volume.shape[0]
        target_depth = Config.NUM_SLICES  # 32

        if current_depth > 0:
            indices = uniform_sample_indices(current_depth, num_samples=target_depth)
            volume_sampled = volume[indices]
        else:
            # Handle missing modality or empty volume by creating zero-filled slices
            volume_sampled = np.zeros(
                (target_depth, Config.IMAGE_SIZE, Config.IMAGE_SIZE), dtype=np.float32
            )

        patient_slices.append(volume_sampled)

    # 4. Stack Depth-wise
    # Each element in patient_slices is (32, 256, 256)
    # Concatenate along channel dimension (axis 0) -> (128, 256, 256)
    stacked_volume = np.concatenate(patient_slices, axis=0)

    return stacked_volume


def load_dataset(split_name, load_cached_data=True):
    """
    Loads the dataset for a specific split (train, val, test).
    Implements caching mechanism using .npy files.

    Args:
        split_name (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        MGMTDataset: The instantiated dataset.
    """
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    X_path = os.path.join(cache_dir, f"cached_{split_name}_X.npy")
    y_path = os.path.join(cache_dir, f"cached_{split_name}_y.npy")

    # Load metadata to get IDs and Labels (source of truth)
    meta_path = os.path.join(Config.METADATA_DIR, f"{split_name}.parquet")
    df = pd.read_parquet(meta_path)

    # Extract IDs and Labels from DataFrame
    ids = df["BraTS21ID"].values.astype(str)
    y = None
    if "MGMT_value" in df.columns:
        y = df["MGMT_value"].values.astype(np.float32)

    # Try loading X from cache
    if load_cached_data and os.path.exists(X_path):
        print(f"Loading {split_name} data from cache...")
        try:
            X = np.load(X_path)
            # Verify length matches
            if len(X) == len(df):
                return MGMTDataset(X, y, ids)
            else:
                print(f"Cache mismatch (X: {len(X)}, DF: {len(df)}). Re-processing...")
        except Exception as e:
            print(f"Error loading cache: {e}. Re-processing...")

    # Process from scratch
    print(f"Processing {split_name} data from scratch...")

    X_list = []

    for _, row in df.iterrows():
        vol = process_patient_data(row)
        X_list.append(vol)

    X = np.array(X_list, dtype=np.float32)

    # Save X to cache
    print(f"Saving {split_name} data to cache...")
    np.save(X_path, X)

    # We also save y for completeness, though we read it from parquet usually
    if y is not None:
        np.save(y_path, y)

    return MGMTDataset(X, y, ids)
