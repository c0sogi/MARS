import os
import numpy as np
import pandas as pd
import pydicom
import cv2
import torch
import torch.nn.functional as F
from library.config import Config


def read_dicom(path):
    """
    Reads a DICOM file using pydicom and returns a spatially resized numpy array.

    Args:
        path (str): Relative path to the DICOM file from INPUT_DIR.

    Returns:
        np.ndarray: 2D float32 array of shape (Config.IMG_SIZE, Config.IMG_SIZE).
    """
    full_path = os.path.join(Config.INPUT_DIR, path)
    try:
        dcm = pydicom.dcmread(full_path)
        img = dcm.pixel_array.astype(np.float32)

        # Resize spatial dimensions (H, W) if they don't match target
        if img.shape[0] != Config.IMG_SIZE or img.shape[1] != Config.IMG_SIZE:
            img = cv2.resize(
                img, (Config.IMG_SIZE, Config.IMG_SIZE), interpolation=cv2.INTER_LINEAR
            )

        return img
    except Exception as e:
        # Return zero array if file is corrupt or unreadable
        return np.zeros((Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)


def process_patient(row):
    """
    Loads, processes, and normalizes the full MRI volume for a single patient.

    Steps:
    1. Load all slices for each of the 4 modalities.
    2. Crop depth to 10%-90% range.
    3. Interpolate depth to Config.NUM_SLICES (32).
    4. Stack to (4, 32, 256, 256).
    5. Apply Global Volumetric Normalization.

    Args:
        row (pd.Series): Row from the metadata dataframe containing file paths.

    Returns:
        torch.Tensor: 4D tensor of shape (4, 32, 256, 256).
    """
    modalities = ["flair", "t1w", "t1wce", "t2w"]
    patient_volume = []

    for mod in modalities:
        paths = row[f"{mod}_paths"]

        # Handle missing modality by creating a zero block
        if paths is None or len(paths) == 0:
            mod_tensor = torch.zeros(
                (Config.NUM_SLICES, Config.IMG_SIZE, Config.IMG_SIZE),
                dtype=torch.float32,
            )
            patient_volume.append(mod_tensor)
            continue

        # Load all slices -> List of (256, 256) arrays
        slices = [read_dicom(p) for p in paths]

        # Stack to 3D numpy array: (D_original, 256, 256)
        volume_3d = np.stack(slices, axis=0)

        # -- High-Density Uniform Sampling (10%-90% Depth Crop) --
        d_orig = volume_3d.shape[0]
        if d_orig > 1:
            d_start = int(d_orig * 0.10)
            d_end = int(d_orig * 0.90)
            # Ensure we have at least one slice
            if d_end <= d_start:
                d_end = d_start + 1
            volume_3d = volume_3d[d_start:d_end]

        # -- Deterministic Linear Interpolation for Depth --
        # Prepare for torch interpolation: (N, C, D, H, W)
        # Current shape: (D, H, W) -> (1, 1, D, H, W)
        tensor_5d = torch.tensor(volume_3d).unsqueeze(0).unsqueeze(0)

        target_size = (Config.NUM_SLICES, Config.IMG_SIZE, Config.IMG_SIZE)

        # Trilinear interpolation handles the 3D resizing (Depth, Height, Width)
        # Since H/W are already 256, this effectively resamples Depth linearly.
        tensor_resized = F.interpolate(
            tensor_5d, size=target_size, mode="trilinear", align_corners=False
        )

        # Squeeze back to (D, H, W) -> (32, 256, 256)
        mod_tensor = tensor_resized.squeeze(0).squeeze(0)
        patient_volume.append(mod_tensor)

    # Stack modalities -> (4, 32, 256, 256)
    volume_4d = torch.stack(patient_volume, dim=0)

    # -- Global Volumetric Normalization --
    v_min = volume_4d.min()
    v_max = volume_4d.max()

    if v_max - v_min > 1e-8:
        volume_4d = (volume_4d - v_min) / (v_max - v_min)
    else:
        # If volume is flat (e.g. all zeros), keep it as is
        volume_4d = volume_4d - v_min

    return volume_4d


def load_dataset(subset, load_cached_data=True):
    """
    Loads the processed dataset (X, y, ids).
    Implements caching to disk to avoid re-processing DICOMs.

    Args:
        subset (str): One of 'train', 'val', 'test'.
        load_cached_data (bool): If True, attempts to load from .npy files first.

    Returns:
        X (np.ndarray): Input data (N, 4, 32, 256, 256).
        y (np.ndarray or None): Targets (N,). None for test set.
        ids (np.ndarray): BraTS21IDs (N,).
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    cache_x_path = os.path.join(Config.WORKING_DIR, f"cached_{subset}_X.npy")
    cache_y_path = os.path.join(Config.WORKING_DIR, f"cached_{subset}_y.npy")
    cache_ids_path = os.path.join(Config.WORKING_DIR, f"cached_{subset}_ids.npy")

    # 1. Attempt to load from cache
    if load_cached_data:
        if os.path.exists(cache_x_path) and os.path.exists(cache_ids_path):
            print(f"Loading {subset} data from cache...")
            X = np.load(cache_x_path)
            ids = np.load(cache_ids_path)

            y = None
            if os.path.exists(cache_y_path):
                y = np.load(cache_y_path)

            return X, y, ids
        else:
            print(f"Cache miss for {subset}. Processing from scratch...")

    # 2. Process from scratch
    if subset == "train":
        meta_path = Config.TRAIN_META_PATH
    elif subset == "val":
        meta_path = Config.VAL_META_PATH
    elif subset == "test":
        meta_path = Config.TEST_META_PATH
    else:
        raise ValueError("Invalid subset.")

    df = pd.read_parquet(meta_path)

    # Handle Debug Mode
    if Config.DEBUG:
        print(f"DEBUG MODE: Loading only {Config.DEBUG_SIZE} samples.")
        df = df.head(Config.DEBUG_SIZE)

    print(f"Processing {len(df)} samples for {subset}...")

    X_list = []
    y_list = []
    ids_list = []

    for idx, row in df.iterrows():
        # Process volume
        tensor = process_patient(row)  # Returns torch tensor
        X_list.append(tensor.numpy())
        ids_list.append(row["BraTS21ID"])

        if "MGMT_value" in row:
            y_list.append(row["MGMT_value"])

    # Aggregate
    X = np.stack(X_list, axis=0).astype(np.float32)
    ids = np.array(ids_list)

    if len(y_list) > 0:
        y = np.array(y_list, dtype=np.float32)
    else:
        y = None

    # 3. Save to cache
    print(f"Saving processed {subset} data to {Config.WORKING_DIR}...")
    np.save(cache_x_path, X)
    np.save(cache_ids_path, ids)
    if y is not None:
        np.save(cache_y_path, y)

    return X, y, ids
