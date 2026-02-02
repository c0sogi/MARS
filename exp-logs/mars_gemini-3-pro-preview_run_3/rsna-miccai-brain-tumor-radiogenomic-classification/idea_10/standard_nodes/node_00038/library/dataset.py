import os
import re
import numpy as np
import pandas as pd
import pydicom
import cv2
import torch
from torch.utils.data import Dataset
from library.utils import set_seed

# Constants
INPUT_DIR = "./input"
CACHE_DIR = "./working/idea_11"
IMG_SIZE = 256
NUM_SLICES = 16


class BraTSDataset(Dataset):
    """
    PyTorch Dataset for BraTS21 Glioblastoma classification.
    """

    def __init__(self, X, y=None, ids=None):
        """
        Args:
            X (np.ndarray): Input images of shape (N, 128, 256, 256).
            y (np.ndarray, optional): Target labels of shape (N,).
            ids (np.ndarray, optional): BraTS21 IDs corresponding to samples.
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None
        self.ids = ids

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample = self.X[idx]
        if self.y is not None:
            return sample, self.y[idx]
        else:
            # Return ID for test set inference
            return sample, self.ids[idx]


def extract_number(filepath):
    """Extracts the integer number from a DICOM filename (e.g. Image-12.dcm -> 12)."""
    match = re.search(r"Image-(\d+)\.dcm", filepath)
    if match:
        return int(match.group(1))
    return 0


def load_dicom_volume(paths, num_slices=NUM_SLICES, img_size=IMG_SIZE):
    """
    Loads, normalizes, and samples slices from a list of DICOM paths.

    Strategy:
    1. Load full volume to compute global min/max.
    2. Normalize.
    3. Uniformly sample 32 slices from 10%-90% depth.
    4. Resize to 256x256.
    """
    # Sort paths numerically to ensure correct 3D reconstruction
    paths = sorted(paths, key=extract_number)

    slices = []
    # Load all slices
    for p in paths:
        full_path = os.path.join(INPUT_DIR, p)
        try:
            dcm = pydicom.dcmread(full_path)
            # Convert to float32 immediately to save memory compared to keeping dcm objects
            img = dcm.pixel_array.astype(np.float32)
            slices.append(img)
        except Exception:
            continue

    if not slices:
        return np.zeros((num_slices, img_size, img_size), dtype=np.float32)

    volume = np.array(slices)  # Shape: (Depth, H, W)

    # Global Volumetric Normalization
    v_min = volume.min()
    v_max = volume.max()

    # Define sampling range (10% to 90%)
    total_depth = len(slices)
    start = int(total_depth * 0.1)
    end = int(total_depth * 0.9)

    if end <= start:
        start = 0
        end = total_depth

    # High-Density Uniform Sampling
    indices = np.linspace(start, end - 1, num_slices).astype(int)

    processed_slices = []
    for idx in indices:
        # Clamp index just in case
        idx = min(max(idx, 0), total_depth - 1)

        img = volume[idx]

        # Normalize
        if v_max - v_min > 0:
            img = (img - v_min) / (v_max - v_min)
        else:
            img = np.zeros_like(img)

        # Resize
        if img.shape != (img_size, img_size):
            img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)

        processed_slices.append(img)

    return np.array(processed_slices)  # Shape: (32, 256, 256)


def process_data(df, cache_key, load_cached_data=True):
    """
    Orchestrates data loading, processing, and caching.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    x_path = os.path.join(CACHE_DIR, f"X_{cache_key}.npy")
    y_path = os.path.join(CACHE_DIR, f"y_{cache_key}.npy")
    ids_path = os.path.join(CACHE_DIR, f"ids_{cache_key}.npy")

    # 1. Try Loading Cache
    if load_cached_data and os.path.exists(x_path) and os.path.exists(ids_path):
        print(f"Loading cached dataset: {cache_key}")
        X = np.load(x_path)
        ids = np.load(ids_path, allow_pickle=True)
        if os.path.exists(y_path):
            y = np.load(y_path)
        else:
            y = None
        return X, y, ids

    # 2. Process from Scratch
    print(f"Processing dataset from scratch: {cache_key} (This may take a while)...")

    X_list = []
    y_list = []
    ids_list = []

    modalities = ["flair", "t1w", "t1wce", "t2w"]

    for i, row in df.iterrows():
        patient_id = row["BraTS21ID"]

        # Process each modality
        mod_volumes = []
        for mod in modalities:
            # Get paths from metadata (handle potential None/NaN if any, though metadata is clean)
            paths = row[f"{mod}_paths"]
            if paths is None:
                paths = []

            vol = load_dicom_volume(paths, num_slices=NUM_SLICES, img_size=IMG_SIZE)
            mod_volumes.append(vol)

        # Deterministic Stacking: FLAIR, T1w, T1wCE, T2w
        # Each vol is (32, 256, 256). Concatenate along depth/channel dim.
        # Result: (128, 256, 256)
        stacked_volume = np.concatenate(mod_volumes, axis=0)

        X_list.append(stacked_volume)
        ids_list.append(patient_id)

        if "MGMT_value" in row:
            y_list.append(row["MGMT_value"])

    # Convert to arrays
    X = np.array(X_list, dtype=np.float32)
    ids = np.array(ids_list)

    # Save Cache
    print(f"Saving cache for {cache_key}...")
    np.save(x_path, X)
    np.save(ids_path, ids)

    if y_list:
        y = np.array(y_list, dtype=np.float32)
        np.save(y_path, y)
    else:
        y = None

    return X, y, ids


def load_dataset(subset="train", load_cached_data=True):
    """
    Main entry point to get the Dataset object.

    Args:
        subset (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        BraTSDataset: The constructed dataset.
    """
    set_seed(42)

    # Load Metadata
    meta_path = os.path.join("./metadata", f"{subset}.parquet")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    df = pd.read_parquet(meta_path)

    # Process Data (with caching)
    X, y, ids = process_data(df, subset, load_cached_data=load_cached_data)

    return BraTSDataset(X, y, ids)
