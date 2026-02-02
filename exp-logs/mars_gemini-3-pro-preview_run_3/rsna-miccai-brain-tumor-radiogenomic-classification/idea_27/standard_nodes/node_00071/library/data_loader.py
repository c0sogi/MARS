import os
import numpy as np
import pandas as pd
import pydicom
import cv2
import torch
from torch.utils.data import Dataset
from library.config import (
    INPUT_DIR,
    TRAIN_META_PATH,
    VAL_META_PATH,
    TEST_META_PATH,
    WORKING_DIR,
    IMG_SIZE,
    NUM_SLICES_PER_MODALITY,
    DEBUG_SAMPLE_SIZE,
)
from library.utils import get_logger

logger = get_logger("data_loader")


class BraTSDataset(Dataset):
    def __init__(self, X, y=None, ids=None, transform=None):
        """
        Args:
            X (np.ndarray): Input data of shape (N, Channels, H, W).
            y (np.ndarray, optional): Target labels of shape (N,).
            ids (np.ndarray, optional): BraTS21IDs.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.X = X
        self.y = y
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Data is already (C, H, W) float32
        img = self.X[idx]

        # Convert to tensor
        img_tensor = torch.from_numpy(img).float()

        if self.transform:
            # Note: Transforms usually expect (C, H, W) or (H, W, C)
            # Adjust based on specific transform library requirements if added later
            pass

        if self.y is not None:
            label = torch.tensor(self.y[idx], dtype=torch.float32)
            return img_tensor, label

        return img_tensor


def load_dicom_volume(file_paths):
    """
    Loads a list of DICOM files, sorts them by Instance Number,
    and returns the volume as a numpy array.
    """
    if not file_paths:
        return None

    slices = []
    for rel_path in file_paths:
        full_path = os.path.join(INPUT_DIR, rel_path)
        try:
            dcm = pydicom.dcmread(full_path)
            # Extract pixel array and instance number
            # We use a tuple (InstanceNumber, pixel_array)
            # Default to 0 if InstanceNumber is missing (unlikely in BraTS)
            inst_num = int(dcm.get(0x00200013, "0").value) if 0x00200013 in dcm else 0
            slices.append((inst_num, dcm.pixel_array))
        except Exception as e:
            # Silent failure for individual bad files, but log if needed
            continue

    if not slices:
        return None

    # Sort by Instance Number to ensure spatial consistency
    slices.sort(key=lambda x: x[0])

    # Stack to create 3D volume (Depth, H, W)
    volume = np.stack([s[1] for s in slices])
    return volume


def process_modality(file_paths):
    """
    Processes a single modality:
    1. Load Volume
    2. Compute Global Min/Max
    3. Normalize Volume
    4. Uniformly Sample 32 slices (10%-90% depth)
    5. Resize to 224x224
    """
    volume = load_dicom_volume(file_paths)

    # Create empty block if volume loading failed
    if volume is None or volume.shape[0] == 0:
        return np.zeros((NUM_SLICES_PER_MODALITY, IMG_SIZE, IMG_SIZE), dtype=np.float32)

    # 1. Per-Modality Volumetric Normalization
    v_min = np.min(volume)
    v_max = np.max(volume)

    # Avoid division by zero
    if v_max - v_min > 0:
        volume = (volume - v_min) / (v_max - v_min)
    else:
        volume = np.zeros_like(volume, dtype=np.float32)

    # 2. Uniform Sampling (10% - 90%)
    depth = volume.shape[0]
    if depth < NUM_SLICES_PER_MODALITY:
        # If fewer slices than required, take all and pad, or just resample
        # Here we use linear interpolation indices to stretch/repeat
        indices = np.linspace(0, depth - 1, NUM_SLICES_PER_MODALITY).astype(int)
    else:
        start = int(depth * 0.1)
        end = int(depth * 0.9)
        # Ensure end > start
        if end <= start:
            start = 0
            end = depth - 1
        indices = np.linspace(start, end, NUM_SLICES_PER_MODALITY).astype(int)

    sampled_volume = volume[indices]

    # 3. Resize
    processed_slices = []
    for i in range(sampled_volume.shape[0]):
        slc = sampled_volume[i]
        # cv2.resize expects (W, H)
        resized = cv2.resize(slc, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        processed_slices.append(resized)

    return np.array(processed_slices, dtype=np.float32)


def process_patient_data(row):
    """
    Process all 4 modalities for a single patient and stack them.
    Returns tensor of shape (128, 224, 224).
    """
    modalities = ["flair", "t1w", "t1wce", "t2w"]
    modality_blocks = []

    for mod in modalities:
        col_name = f"{mod}_paths"
        paths = row[col_name]
        # Handle NaN or None
        if not isinstance(paths, list):
            paths = []

        block = process_modality(paths)
        modality_blocks.append(block)

    # Stack along the depth/channel dimension
    # Each block is (32, 224, 224) -> Result (128, 224, 224)
    full_volume = np.concatenate(modality_blocks, axis=0)
    return full_volume


def generate_data(meta_path, cache_prefix, load_cached_data=True):
    """
    Generates or loads data for a specific split (train/val/test).
    """
    # Cache file paths
    cache_X = os.path.join(WORKING_DIR, f"{cache_prefix}_X.npy")
    cache_y = os.path.join(WORKING_DIR, f"{cache_prefix}_y.npy")
    cache_ids = os.path.join(WORKING_DIR, f"{cache_prefix}_ids.npy")

    # Try loading from cache
    if load_cached_data and os.path.exists(cache_X) and os.path.exists(cache_ids):
        logger.info(f"Loading cached {cache_prefix} data from {WORKING_DIR}...")
        X = np.load(cache_X)
        ids = np.load(cache_ids)
        y = np.load(cache_y) if os.path.exists(cache_y) else None
        return X, y, ids

    # Generate from scratch
    logger.info(f"Generating {cache_prefix} data from scratch...")
    df = pd.read_parquet(meta_path)

    if DEBUG_SAMPLE_SIZE is not None:
        df = df.head(DEBUG_SAMPLE_SIZE)
        logger.info(f"Debug Mode: Sampled {len(df)} rows.")

    X_list = []
    y_list = []
    ids_list = []

    total = len(df)
    for idx, row in df.iterrows():
        if idx % 10 == 0:
            logger.info(f"Processing {idx}/{total}...")

        vol = process_patient_data(row)
        X_list.append(vol)
        ids_list.append(row["BraTS21ID"])

        if "MGMT_value" in row:
            y_list.append(row["MGMT_value"])

    X = np.array(X_list, dtype=np.float32)
    ids = np.array(ids_list)
    y = np.array(y_list, dtype=np.float32) if y_list else None

    # Save to cache
    logger.info(f"Saving {cache_prefix} data to cache...")
    np.save(cache_X, X)
    np.save(cache_ids, ids)
    if y is not None:
        np.save(cache_y, y)

    return X, y, ids


def get_datasets(load_cached_data=True):
    """
    Main entry point to get Train, Val, and Test datasets.
    Handles caching and generation logic.
    """
    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # 1. Train Data
    X_train, y_train, ids_train = generate_data(
        TRAIN_META_PATH, "cached_train", load_cached_data
    )
    train_dataset = BraTSDataset(X_train, y_train, ids_train)

    # 2. Validation Data
    X_val, y_val, ids_val = generate_data(VAL_META_PATH, "cached_val", load_cached_data)
    val_dataset = BraTSDataset(X_val, y_val, ids_val)

    # 3. Test Data
    X_test, _, ids_test = generate_data(TEST_META_PATH, "cached_test", load_cached_data)
    test_dataset = BraTSDataset(X_test, None, ids_test)

    logger.info(
        f"Data Loaded: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}"
    )
    return train_dataset, val_dataset, test_dataset
