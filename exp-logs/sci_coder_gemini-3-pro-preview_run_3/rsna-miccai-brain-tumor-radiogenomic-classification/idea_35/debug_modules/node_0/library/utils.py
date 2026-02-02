import os
import random
import numpy as np
import torch
import pandas as pd
import pydicom
import cv2
from typing import Tuple, List, Optional, Union


def seed_everything(seed: int = 42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Returns the appropriate PyTorch device (CUDA if available, else CPU).
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_dicom_image(path: str, target_size: int = 256) -> np.ndarray:
    """
    Reads a DICOM file and returns a normalized 2D numpy array.
    Handles resizing and basic error checking.
    """
    try:
        if not os.path.exists(path):
            return np.zeros((target_size, target_size), dtype=np.float32)

        dcm = pydicom.dcmread(path)
        img = dcm.pixel_array.astype(np.float32)

        # Resize if necessary
        if img.shape[0] != target_size or img.shape[1] != target_size:
            img = cv2.resize(
                img, (target_size, target_size), interpolation=cv2.INTER_LINEAR
            )

        return img
    except Exception:
        # Return zero array for corrupted/missing files to maintain batch shape
        return np.zeros((target_size, target_size), dtype=np.float32)


def get_sorted_image_paths(paths: List[str]) -> List[str]:
    """
    Sorts file paths based on the integer index in the filename (e.g., Image-10.dcm).
    """

    def extract_id(path):
        try:
            filename = os.path.basename(path)
            # Format is typically Image-X.dcm
            num = filename.split("-")[1].split(".")[0]
            return int(num)
        except:
            return -1

    valid_paths = [p for p in paths if p.endswith(".dcm")]
    return sorted(valid_paths, key=extract_id)


def process_patient_scan(
    row: pd.Series, input_dir: str, target_size: int = 256, num_slices: int = 32
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Processes a single patient's MRI scans according to the DSSV-Net strategy.

    Steps:
    1. For each modality (FLAIR, T1w, T1wCE, T2w):
       - Load paths and sort by slice index.
       - Sample 32 slices uniformly from the 10%-90% depth range.
       - Split into Even (0, 2, ..) and Odd (1, 3, ..) sets.
       - Apply View-Adaptive Per-Modality Normalization.
    2. Stack modalities to form two streams.

    Returns:
        even_stream: (64, target_size, target_size)
        odd_stream:  (64, target_size, target_size)
    """
    modalities = ["flair", "t1w", "t1wce", "t2w"]

    even_channels = []
    odd_channels = []

    for mod in modalities:
        # Retrieve relative paths from metadata
        col_name = f"{mod}_paths"
        rel_paths = row.get(col_name, [])
        if not isinstance(rel_paths, list):
            rel_paths = []

        # Construct full paths
        full_paths = [os.path.join(input_dir, p) for p in rel_paths]
        sorted_paths = get_sorted_image_paths(full_paths)

        total_files = len(sorted_paths)

        # High-Density Uniform Sampling
        if total_files < num_slices:
            # If fewer slices than requested, sample from what is available with repetition/interpolation
            if total_files == 0:
                selected_paths = []
            else:
                indices = np.linspace(0, total_files - 1, num_slices).astype(int)
                selected_paths = [sorted_paths[i] for i in indices]
        else:
            # Filter to 10%-90% range to avoid edge artifacts
            start_idx = int(total_files * 0.1)
            end_idx = int(total_files * 0.9)
            if end_idx <= start_idx:
                start_idx = 0
                end_idx = total_files

            # Uniformly sample num_slices
            indices = np.linspace(start_idx, end_idx - 1, num_slices).astype(int)
            selected_paths = [sorted_paths[i] for i in indices]

        # Load images
        volume = []
        for p in selected_paths:
            if p:
                img = load_dicom_image(p, target_size)
            else:
                img = np.zeros((target_size, target_size), dtype=np.float32)
            volume.append(img)

        # Handle case where volume is empty or short (pad with zeros)
        if len(volume) < num_slices:
            for _ in range(num_slices - len(volume)):
                volume.append(np.zeros((target_size, target_size), dtype=np.float32))

        volume = np.array(volume)  # Shape: (32, H, W)

        # Deterministic Strided Splitting
        # Even indices: 0, 2, ..., 30 (16 slices)
        # Odd indices: 1, 3, ..., 31 (16 slices)
        even_vol = volume[0::2]
        odd_vol = volume[1::2]

        # View-Adaptive Per-Modality Normalization
        # Normalize Even View
        if even_vol.max() > even_vol.min():
            even_vol = (even_vol - even_vol.min()) / (even_vol.max() - even_vol.min())
        else:
            even_vol = np.zeros_like(even_vol)

        # Normalize Odd View
        if odd_vol.max() > odd_vol.min():
            odd_vol = (odd_vol - odd_vol.min()) / (odd_vol.max() - odd_vol.min())
        else:
            odd_vol = np.zeros_like(odd_vol)

        even_channels.append(even_vol)
        odd_channels.append(odd_vol)

    # Stack modalities along the channel dimension
    # Each list contains 4 arrays of shape (16, H, W)
    # Resulting shape: (64, H, W)
    even_stream = np.concatenate(even_channels, axis=0)
    odd_stream = np.concatenate(odd_channels, axis=0)

    return even_stream, odd_stream


def load_data(
    split: str = "train",
    load_cached_data: bool = True,
    limit_size: Optional[int] = None,
    cache_dir: str = "./working/idea_35/",
    input_dir: str = "./input",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads and processes the dataset. Implements caching to .npy files.

    Args:
        split: 'train', 'val', or 'test'.
        load_cached_data: If True, attempts to load from cache_dir.
        limit_size: If provided, limits the dataset size (for debugging).
        cache_dir: Directory to store/load cached numpy arrays.
        input_dir: Root directory of the input data.

    Returns:
        X: Numpy array of shape (N, 2, 64, 256, 256).
        y: Numpy array of shape (N,). Contains -1 for test set.
        ids: Numpy array of shape (N,). BraTS21IDs.
    """
    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)

    # Define cache file paths
    cache_X = os.path.join(cache_dir, f"cached_{split}_X.npy")
    cache_y = os.path.join(cache_dir, f"cached_{split}_y.npy")
    cache_ids = os.path.join(cache_dir, f"cached_{split}_ids.npy")

    # Attempt to load from cache
    if (
        load_cached_data
        and os.path.exists(cache_X)
        and os.path.exists(cache_y)
        and os.path.exists(cache_ids)
    ):
        print(f"Loading cached {split} data from {cache_dir}...")
        X = np.load(cache_X)
        y = np.load(cache_y)
        ids = np.load(cache_ids)

        if limit_size is not None:
            return X[:limit_size], y[:limit_size], ids[:limit_size]
        return X, y, ids

    print(f"Processing {split} data from scratch (this may take a while)...")

    # Load metadata
    meta_path = os.path.join("./metadata", f"{split}.parquet")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"Metadata file not found: {meta_path}. Please ensure metadata generation was successful."
        )

    df = pd.read_parquet(meta_path)

    # Apply limit if requested (before processing to save time)
    if limit_size is not None:
        df = df.iloc[:limit_size]

    X_list = []
    y_list = []
    ids_list = []

    for idx, row in df.iterrows():
        pid = row["BraTS21ID"]

        # Process the scan into dual streams
        even_stream, odd_stream = process_patient_scan(row, input_dir)

        # Stack streams: Shape becomes (2, 64, 256, 256)
        streams = np.stack([even_stream, odd_stream], axis=0)

        X_list.append(streams)
        ids_list.append(pid)

        # Handle target
        if "MGMT_value" in row:
            y_list.append(row["MGMT_value"])
        else:
            y_list.append(-1.0)  # Placeholder for test set

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    ids = np.array(ids_list)

    # Cache the results ONLY if we processed the full dataset (limit_size is None)
    # This prevents overwriting a full cache with a partial debug cache.
    if limit_size is None:
        print(f"Saving {split} data to cache at {cache_dir}...")
        np.save(cache_X, X)
        np.save(cache_y, y)
        np.save(cache_ids, ids)

    return X, y, ids
