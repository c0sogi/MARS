import os
import re
import numpy as np
import pandas as pd
import pydicom
import cv2
import torch
from torch.utils.data import Dataset
from library.utils import seed_everything

# Constants
INPUT_DIR = "./input"
CACHE_DIR = "./working/idea_33"
IMG_SIZE = 256
NUM_SLICES = 16
MODALITIES = ["FLAIR", "T1w", "T1wCE", "T2w"]
TOTAL_CHANNELS = NUM_SLICES * len(MODALITIES)  # 64


def get_image_plane(path):
    """
    Parses the integer slice number from the filename (External Integer Sorting).
    e.g., 'Image-10.dcm' -> 10
    """
    basename = os.path.basename(path)
    # Find all digit sequences, take the last one as the instance number
    nums = re.findall(r"\d+", basename)
    if nums:
        return int(nums[-1])
    return -1


def get_sorted_image_files(file_paths):
    """
    Sorts a list of file paths based on the integer found in the filename.
    """
    if not file_paths:
        return []

    # Create (index, path) tuples
    path_tuples = []
    for p in file_paths:
        idx = get_image_plane(p)
        path_tuples.append((idx, p))

    # Sort by index
    path_tuples.sort(key=lambda x: x[0])

    # Return sorted paths
    return [x[1] for x in path_tuples]


def load_dicom_volume(paths, num_slices=16, img_size=256):
    """
    Loads, sorts, subsamples, resizes, and normalizes a volume for a single modality.

    Implements:
    1. External Integer Sorting
    2. Uniform Sampling (10%-90%)
    3. View-Adaptive Per-Modality Normalization

    Returns: numpy array of shape (num_slices, img_size, img_size)
    """
    if not paths or len(paths) == 0:
        return np.zeros((num_slices, img_size, img_size), dtype=np.float32)

    # 1. External Integer Sorting
    sorted_paths = get_sorted_image_files(paths)

    # Resolve relative paths to full paths
    full_paths = [os.path.join(INPUT_DIR, p) for p in sorted_paths]
    total_files = len(full_paths)

    # 2. Uniform Sampling (10%-90%)
    if total_files < num_slices:
        # If fewer slices than desired, sample with replacement or linspace over available
        if total_files > 0:
            indices = np.linspace(0, total_files - 1, num_slices).astype(int)
        else:
            return np.zeros((num_slices, img_size, img_size), dtype=np.float32)
    else:
        # Exclude top and bottom 10% to avoid artifacts/skull
        start_idx = int(total_files * 0.1)
        end_idx = int(total_files * 0.9)

        # Ensure valid range
        if end_idx <= start_idx:
            start_idx = 0
            end_idx = total_files

        indices = np.linspace(start_idx, end_idx - 1, num_slices).astype(int)

    selected_paths = [full_paths[i] for i in indices]

    volume = []
    for p in selected_paths:
        try:
            dcm = pydicom.dcmread(p)
            img = dcm.pixel_array.astype(np.float32)

            # Resize
            if img.shape != (img_size, img_size):
                img = cv2.resize(
                    img, (img_size, img_size), interpolation=cv2.INTER_AREA
                )

            volume.append(img)
        except Exception:
            # Robust fallback for corrupt files
            volume.append(np.zeros((img_size, img_size), dtype=np.float32))

    volume = np.array(volume)  # Shape: (16, 256, 256)

    # 3. View-Adaptive Per-Modality Normalization
    # Calculate min/max ONLY on the selected 16 slices
    min_val = np.min(volume)
    max_val = np.max(volume)

    if max_val - min_val > 0:
        volume = (volume - min_val) / (max_val - min_val)
    else:
        volume = np.zeros_like(volume)

    return volume


def process_dataset(df, split_name, load_cached_data=True, debug_limit=None):
    """
    Processes the dataframe into X (images) and y (labels).
    Implements caching mechanism to ./working/idea_33/.

    Returns:
        X: numpy array (N, 64, 256, 256)
        y: numpy array (N,) or None
        ids: numpy array (N,)
    """
    seed_everything(42)

    # Ensure cache directory exists
    os.makedirs(CACHE_DIR, exist_ok=True)

    cache_X = os.path.join(CACHE_DIR, f"cached_{split_name}_X.npy")
    cache_y = os.path.join(CACHE_DIR, f"cached_{split_name}_y.npy")
    cache_ids = os.path.join(CACHE_DIR, f"cached_{split_name}_ids.npy")

    # 1. Try Loading Cache
    if load_cached_data and os.path.exists(cache_X) and os.path.exists(cache_ids):
        print(f"Loading cached {split_name} data...")
        try:
            X = np.load(cache_X)
            ids = np.load(cache_ids, allow_pickle=True)
            y = np.load(cache_y) if os.path.exists(cache_y) else None

            # Apply debug limit after loading if requested
            if debug_limit and len(X) > debug_limit:
                return (
                    X[:debug_limit],
                    y[:debug_limit] if y is not None else None,
                    ids[:debug_limit],
                )
            return X, y, ids
        except Exception as e:
            print(f"Error loading cache: {e}. Reprocessing...")

    # 2. Process from Scratch
    print(f"Processing {split_name} data from scratch...")
    X_list = []
    y_list = []
    ids_list = []

    # Map modality names to dataframe columns
    mod_cols = {
        "FLAIR": "flair_paths",
        "T1w": "t1w_paths",
        "T1wCE": "t1wce_paths",
        "T2w": "t2w_paths",
    }

    count = 0
    for idx, row in df.iterrows():
        if debug_limit and count >= debug_limit:
            break

        patient_id = row["BraTS21ID"]

        # 4. Modality-Grouped Stacking
        # Order: FLAIR, T1w, T1wCE, T2w
        patient_volumes = []
        for mod in MODALITIES:
            paths = row.get(mod_cols[mod], [])
            if paths is None:
                paths = []
            paths = list(paths)

            # Load volume (includes sorting, sampling, normalization)
            vol = load_dicom_volume(paths, num_slices=NUM_SLICES, img_size=IMG_SIZE)
            patient_volumes.append(vol)

        # Concatenate along channel dimension (axis 0)
        # Each vol is (16, 256, 256) -> Result: (64, 256, 256)
        full_volume = np.concatenate(patient_volumes, axis=0)

        X_list.append(full_volume)
        ids_list.append(patient_id)

        if "MGMT_value" in row:
            y_list.append(row["MGMT_value"])

        count += 1

    # Convert to numpy arrays
    X = np.array(X_list, dtype=np.float32)
    ids = np.array(ids_list)
    y = np.array(y_list, dtype=np.float32) if y_list else None

    # 3. Save to Cache
    print(f"Saving {split_name} data to cache...")
    np.save(cache_X, X)
    np.save(cache_ids, ids)
    if y is not None:
        np.save(cache_y, y)

    return X, y, ids


class VAMSDataset(Dataset):
    """
    Dataset wrapper for the VAMS Network.

    This class wraps the pre-processed numpy arrays generated by `process_dataset`.
    The heavy lifting (loading, sorting, sampling, normalization) is done during
    the processing stage to allow for efficient caching and training.
    """

    def __init__(self, X, y=None):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = self.X[idx]
        # Ensure float32 tensor
        tensor_img = torch.tensor(img, dtype=torch.float32)

        if self.y is not None:
            label = self.y[idx]
            return tensor_img, torch.tensor(label, dtype=torch.float32)
        return tensor_img
