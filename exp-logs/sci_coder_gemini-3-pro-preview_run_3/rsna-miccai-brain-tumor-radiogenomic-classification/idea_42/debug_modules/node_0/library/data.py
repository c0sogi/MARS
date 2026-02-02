import os
import re
import numpy as np
import pandas as pd
import pydicom
import cv2
import torch
from torch.utils.data import Dataset
from library.utils import seed_everything

# Ensure reproducibility
seed_everything(42)


def extract_slice_index(path):
    """
    Extracts the integer slice index from a DICOM filename (e.g., 'Image-123.dcm').
    Returns -1 if the pattern is not found.
    """
    match = re.search(r"Image-(\d+)\.dcm", path)
    return int(match.group(1)) if match else -1


def load_dicom_slice(path, img_size=224):
    """
    Reads a DICOM file, converts to float, and resizes to (img_size, img_size).
    Returns a zero array if reading fails.
    """
    try:
        dcm = pydicom.dcmread(path)
        img = dcm.pixel_array.astype(float)

        # Resize if dimensions differ
        if img.shape != (img_size, img_size):
            img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)

        return img
    except Exception:
        return np.zeros((img_size, img_size), dtype=float)


def process_patient_volume(row, input_dir, img_size=224, num_slices=32):
    """
    Processes a single patient's data:
    1. Loads 32 slices uniformly from middle 80% for each modality.
    2. Normalizes based on the subset statistics.
    3. Splits into Even and Odd streams.
    4. Stacks by modality.

    Returns:
        x_even (np.ndarray): Shape (64, img_size, img_size)
        x_odd (np.ndarray): Shape (64, img_size, img_size)
    """
    modalities = ["flair", "t1w", "t1wce", "t2w"]

    # Lists to hold the stacked slices for each stream
    # Structure: [FLAIR_even_block, T1w_even_block, ...]
    even_modality_blocks = []
    odd_modality_blocks = []

    for mod in modalities:
        # Retrieve paths from metadata (column names are e.g., 'flair_paths')
        col_name = f"{mod}_paths"
        rel_paths = row.get(col_name, [])

        # Handle empty/NaN paths safely
        if not isinstance(rel_paths, list) and not isinstance(rel_paths, np.ndarray):
            rel_paths = []

        full_paths = [os.path.join(input_dir, p) for p in rel_paths]

        # Sort paths numerically
        valid_paths = [p for p in full_paths if extract_slice_index(p) != -1]
        sorted_paths = sorted(valid_paths, key=extract_slice_index)

        num_files = len(sorted_paths)

        # --- High-Density Uniform Sampling ---
        if num_files == 0:
            # Missing modality: create zero volume
            stack = np.zeros((num_slices, img_size, img_size), dtype=float)
        else:
            # Define middle 80% range
            start_idx = int(num_files * 0.1)
            end_idx = int(num_files * 0.9)

            if end_idx <= start_idx:
                start_idx, end_idx = 0, num_files

            # Generate indices
            if (end_idx - start_idx) < num_slices:
                # Not enough slices in range, sample from all available with replacement/interpolation logic
                # Here using linspace on full range if cropped range is too small
                indices = np.linspace(0, num_files - 1, num_slices).astype(int)
                sample_paths = [sorted_paths[i] for i in indices]
            else:
                indices = np.linspace(start_idx, end_idx - 1, num_slices).astype(int)
                sample_paths = [sorted_paths[i] for i in indices]

            # Load images
            imgs = [load_dicom_slice(p, img_size) for p in sample_paths]
            stack = np.array(imgs)  # (32, H, W)

            # --- Subset-Adaptive Normalization ---
            min_val = stack.min()
            max_val = stack.max()
            if max_val - min_val > 0:
                stack = (stack - min_val) / (max_val - min_val)
            else:
                # If constant value (e.g. all black), keep as zeros
                stack = np.zeros_like(stack)

        # --- Deterministic Strided Splitting ---
        # Even indices: 0, 2, ..., 30 (16 slices)
        even_subset = stack[0::2]
        # Odd indices: 1, 3, ..., 31 (16 slices)
        odd_subset = stack[1::2]

        even_modality_blocks.append(even_subset)
        odd_modality_blocks.append(odd_subset)

    # Stack along channel dimension (axis 0)
    # Result shape: (4 * 16, H, W) = (64, H, W)
    x_even = np.concatenate(even_modality_blocks, axis=0).astype(np.float32)
    x_odd = np.concatenate(odd_modality_blocks, axis=0).astype(np.float32)

    return x_even, x_odd


def get_dataset_arrays(
    metadata_path, cache_name, load_cached_data=True, input_dir="./input"
):
    """
    Loads metadata, processes images into arrays, and handles caching.
    """
    cache_dir = "./working/idea_42"
    os.makedirs(cache_dir, exist_ok=True)

    path_x_even = os.path.join(cache_dir, f"X_{cache_name}_even.npy")
    path_x_odd = os.path.join(cache_dir, f"X_{cache_name}_odd.npy")
    path_y = os.path.join(cache_dir, f"y_{cache_name}.npy")
    path_ids = os.path.join(cache_dir, f"ids_{cache_name}.npy")

    # 1. Try Loading from Cache
    if load_cached_data:
        if (
            os.path.exists(path_x_even)
            and os.path.exists(path_x_odd)
            and os.path.exists(path_ids)
        ):
            # Check y existence (test set might not have it)
            has_y = os.path.exists(path_y)

            try:
                X_even = np.load(path_x_even)
                X_odd = np.load(path_x_odd)
                ids = np.load(path_ids, allow_pickle=True)
                y = np.load(path_y) if has_y else None
                return X_even, X_odd, y, ids
            except Exception:
                # If load fails, fall through to processing
                pass

    # 2. Process from Scratch
    df = pd.read_parquet(metadata_path)

    X_even_list = []
    X_odd_list = []
    ids_list = []
    y_list = []

    # Configuration
    IMG_SIZE = 224
    NUM_SLICES = 32

    for idx, row in df.iterrows():
        xe, xo = process_patient_volume(row, input_dir, IMG_SIZE, NUM_SLICES)

        X_even_list.append(xe)
        X_odd_list.append(xo)
        ids_list.append(str(row["BraTS21ID"]))

        if "MGMT_value" in row:
            y_list.append(row["MGMT_value"])

    X_even = np.array(X_even_list, dtype=np.float32)
    X_odd = np.array(X_odd_list, dtype=np.float32)
    ids = np.array(ids_list)

    # Save to Cache
    np.save(path_x_even, X_even)
    np.save(path_x_odd, X_odd)
    np.save(path_ids, ids)

    if len(y_list) > 0:
        y = np.array(y_list, dtype=np.float32)
        np.save(path_y, y)
    else:
        y = None

    return X_even, X_odd, y, ids


class BraTSDataset(Dataset):
    def __init__(self, X_even, X_odd, y=None):
        self.X_even = X_even
        self.X_odd = X_odd
        self.y = y

    def __len__(self):
        return len(self.X_even)

    def __getitem__(self, idx):
        xe = torch.tensor(self.X_even[idx], dtype=torch.float32)
        xo = torch.tensor(self.X_odd[idx], dtype=torch.float32)

        if self.y is not None:
            label = torch.tensor(self.y[idx], dtype=torch.float32).unsqueeze(0)
            return xe, xo, label
        else:
            return xe, xo
