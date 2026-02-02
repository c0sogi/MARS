import os
import sys
import numpy as np
import pandas as pd
import pydicom
import cv2
import torch
from torch.utils.data import Dataset

# Import configuration and utilities from the provided library
from library import config
from library import utils


class BraTSDataset(Dataset):
    """
    PyTorch Dataset for BraTS21 data.
    """

    def __init__(self, X, y=None):
        """
        Args:
            X (np.ndarray): Input data of shape (N, C, H, W).
            y (np.ndarray, optional): Target labels of shape (N,).
        """
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Convert numpy array to torch tensor
        img = torch.from_numpy(self.X[idx]).float()

        if self.y is not None:
            label = torch.tensor(self.y[idx], dtype=torch.float)
            return img, label
        else:
            return img


def read_dicom(rel_path):
    """
    Reads a DICOM file from the input directory.

    Args:
        rel_path (str): Relative path to the DICOM file from input root.

    Returns:
        np.ndarray: Pixel array (float32). Returns zeros if read fails.
    """
    full_path = os.path.join(config.INPUT_DIR, rel_path)
    try:
        dcm = pydicom.dcmread(full_path)
        img = dcm.pixel_array.astype(np.float32)
        return img
    except Exception as e:
        # Fallback for corrupt files or read errors
        return np.zeros((config.IMG_SIZE, config.IMG_SIZE), dtype=np.float32)


def process_volume(row, img_size=config.IMG_SIZE, num_slices=config.NUM_SLICES):
    """
    Loads, normalizes, and stacks MRI slices for a single patient.

    Args:
        row (pd.Series): Row from the metadata DataFrame containing file paths.
        img_size (int): Target spatial resolution (H=W).
        num_slices (int): Number of slices to sample per modality.

    Returns:
        np.ndarray: Processed volume of shape (num_slices * 4, img_size, img_size).
    """
    # Define modalities in the specific order for interleaving
    # Metadata columns are named: flair_paths, t1w_paths, t1wce_paths, t2w_paths
    modalities = ["flair", "t1w", "t1wce", "t2w"]

    # Storage: (Slices, Modalities, H, W)
    # This shape facilitates the interleaved flattening later
    volume_stack = np.zeros(
        (num_slices, len(modalities), img_size, img_size), dtype=np.float32
    )

    for m_idx, mod in enumerate(modalities):
        col_name = f"{mod}_paths"
        paths = row[col_name] if col_name in row else []

        # Handle cases where paths might be None or empty
        if paths is None:
            paths = []

        num_files = len(paths)

        if num_files == 0:
            # If no files, the volume remains zeros for this modality
            continue

        # High-Density Uniform Sampling (10% - 90% depth)
        start_idx = int(num_files * 0.1)
        end_idx = int(num_files * 0.9)

        # Handle edge cases with very few slices
        if end_idx <= start_idx:
            start_idx = 0
            end_idx = num_files

        # Generate evenly spaced indices
        if num_files > 0:
            indices = np.linspace(start_idx, end_idx - 1, num_slices).astype(int)
            # Clip indices just in case
            indices = np.clip(indices, 0, num_files - 1)
        else:
            indices = []

        # Load slices
        modality_slices = []
        for i in indices:
            img = read_dicom(paths[i])

            # Resize if necessary
            if img.shape != (img_size, img_size):
                try:
                    img = cv2.resize(
                        img, (img_size, img_size), interpolation=cv2.INTER_AREA
                    )
                except Exception:
                    img = np.zeros((img_size, img_size), dtype=np.float32)

            modality_slices.append(img)

        modality_volume = np.array(
            modality_slices
        )  # Shape: (num_slices, img_size, img_size)

        # Global Volumetric Normalization (per modality)
        # We normalize based on the min/max of the sampled volume for this modality
        v_min = np.min(modality_volume)
        v_max = np.max(modality_volume)

        if v_max - v_min > 0:
            modality_volume = (modality_volume - v_min) / (v_max - v_min)
        else:
            # If constant value (e.g., all zeros), keep as is
            modality_volume = np.zeros_like(modality_volume)

        # Store into the main stack
        volume_stack[:, m_idx, :, :] = modality_volume

    # Reshape to Interleaved format: (Slices, Modalities, H, W) -> (Slices*Modalities, H, W)
    # Order becomes: S0_M0, S0_M1, S0_M2, S0_M3, S1_M0...
    final_volume = volume_stack.reshape(-1, img_size, img_size)

    return final_volume


def get_dataset(metadata_path, cache_prefix, load_cached_data=True):
    """
    Loads dataset from metadata, using caching to save processing time.

    Args:
        metadata_path (str): Path to the parquet metadata file.
        cache_prefix (str): Prefix for the cache files (e.g., 'train', 'val', 'test').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (X, y) numpy arrays. y is None if target column is missing.
    """
    # Ensure working directory exists
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    X_path = os.path.join(config.WORKING_DIR, f"{cache_prefix}_X.npy")
    y_path = os.path.join(config.WORKING_DIR, f"{cache_prefix}_y.npy")

    # 1. Try loading from cache
    if load_cached_data:
        if os.path.exists(X_path):
            # If X exists, we assume y exists if it was meant to be there
            # (or we check y_path existence for labeled sets)
            try:
                X = np.load(X_path)
                y = None
                if os.path.exists(y_path):
                    y = np.load(y_path)
                return X, y
            except Exception as e:
                # If load fails, proceed to process from scratch
                pass

    # 2. Process from scratch
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_parquet(metadata_path)

    X_list = []
    y_list = []

    # Iterate over patients
    for idx, row in df.iterrows():
        vol = process_volume(
            row, img_size=config.IMG_SIZE, num_slices=config.NUM_SLICES
        )
        X_list.append(vol)

        if "MGMT_value" in row:
            y_list.append(row["MGMT_value"])

    X = np.array(X_list, dtype=np.float32)

    # Save X to cache
    np.save(X_path, X)

    if len(y_list) > 0:
        y = np.array(y_list, dtype=np.float32)
        np.save(y_path, y)
    else:
        y = None
        # Clean up old y cache if it exists but shouldn't (e.g. switching from train to test with same prefix)
        if os.path.exists(y_path):
            os.remove(y_path)

    return X, y
