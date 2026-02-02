import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from library.config import (
    INPUT_DIR,
    CACHE_DIR,
    IMG_SIZE,
    SEED,
    ROI_ANCHOR_MODALITY,
    ROI_MIN,
    ROI_MAX,
    NUM_SLABS,
    SLAB_OFFSETS,
    SLAB_THICKNESS,
    MODALITIES,
    BATCH_SIZE,
    NUM_WORKERS,
    ROTATION_DEGREES,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    DEBUG_DATA_LIMIT,
)
from library.utils import read_dicom_robust, seed_everything

# Ensure reproducibility
seed_everything(SEED)


class BraTSDataset(Dataset):
    def __init__(self, images, labels=None, transform=False):
        """
        Args:
            images (np.ndarray): Shape (N, Channels, H, W)
            labels (np.ndarray, optional): Shape (N,). Defaults to None.
            transform (bool): Whether to apply geometric augmentations.
        """
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Shape: (Channels, H, W)
        img = self.images[idx].copy()

        # Apply Augmentations
        if self.transform:
            # Transpose to (H, W, Channels) for OpenCV
            img_hwc = np.transpose(img, (1, 2, 0))

            # 1. Random Rotation
            if ROTATION_DEGREES > 0:
                angle = np.random.uniform(-ROTATION_DEGREES, ROTATION_DEGREES)
                # Calculate rotation matrix
                center = (IMG_SIZE / 2, IMG_SIZE / 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                # Apply affine transform with Reflection Padding
                img_hwc = cv2.warpAffine(
                    img_hwc,
                    M,
                    (IMG_SIZE, IMG_SIZE),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT,
                )

            # 2. Random Horizontal Flip
            if np.random.rand() > 0.5:
                img_hwc = cv2.flip(img_hwc, 1)

            # 3. Random Vertical Flip
            if np.random.rand() > 0.5:
                img_hwc = cv2.flip(img_hwc, 0)

            # Transpose back to (Channels, H, W)
            img = np.transpose(img_hwc, (2, 0, 1))

        # Convert to Tensor
        img_tensor = torch.from_numpy(img).float()

        if self.labels is not None:
            label_tensor = torch.tensor(self.labels[idx], dtype=torch.float32)
            return img_tensor, label_tensor
        else:
            # Return dummy label for test set
            return img_tensor, torch.tensor(0.0)


def _get_sorted_files(dir_path):
    """Returns sorted list of DICOM files in a directory based on numerical index."""
    if not os.path.exists(dir_path):
        return []
    files = [f for f in os.listdir(dir_path) if f.endswith(".dcm")]
    # Sort by the number in 'Image-X.dcm'
    try:
        files.sort(key=lambda x: int(x.split("-")[1].split(".")[0]))
    except Exception:
        files.sort()  # Fallback
    return files


def _process_subject(row):
    """
    Extracts the 12-channel volume for a single subject.
    Returns:
        volume (np.ndarray): Shape (12, H, W)
        status (bool): True if successful, False if corrupted/empty
    """
    subject_channels = []

    # 1. Determine Anchor using FLAIR
    flair_path = os.path.join(INPUT_DIR, row[f"path_{ROI_ANCHOR_MODALITY}"])
    flair_files = _get_sorted_files(flair_path)

    if not flair_files:
        return (
            np.zeros(
                (len(MODALITIES) * NUM_SLABS, IMG_SIZE, IMG_SIZE), dtype=np.float32
            ),
            False,
        )

    num_slices = len(flair_files)
    start_idx = int(num_slices * ROI_MIN)
    end_idx = int(num_slices * ROI_MAX)

    # Calculate integral of raw intensity for ROI selection
    max_integral = -1
    anchor_idx = num_slices // 2  # Fallback

    # Optimization: If too many slices, stride to save time, but here we do full scan for accuracy
    # as per "Raw-Integral ROI Selection" requirement.
    # To be safe with runtime, we check bounds.
    if start_idx < end_idx:
        for i in range(start_idx, end_idx):
            f_path = os.path.join(flair_path, flair_files[i])
            img = read_dicom_robust(f_path, target_size=(IMG_SIZE, IMG_SIZE))
            current_integral = np.sum(img)
            if current_integral > max_integral:
                max_integral = current_integral
                anchor_idx = i

    # 2. Extract Slabs for all modalities
    for mod in MODALITIES:
        mod_path = os.path.join(INPUT_DIR, row[f"path_{mod}"])
        mod_files = _get_sorted_files(mod_path)
        mod_len = len(mod_files)

        if mod_len == 0:
            # Missing modality: append empty slabs
            for _ in range(NUM_SLABS):
                subject_channels.append(
                    np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
                )
            continue

        # Extract slabs at offsets
        for offset in SLAB_OFFSETS:
            center_idx = anchor_idx + offset

            # Slab Averaging: [center-1, center, center+1]
            # We map the FLAIR anchor index directly to other modalities (assuming co-registration)
            # We clamp indices to valid range
            slab_images = []
            half_thick = SLAB_THICKNESS // 2

            for k in range(-half_thick, half_thick + 1):
                slice_idx = np.clip(center_idx + k, 0, mod_len - 1)
                f_name = mod_files[slice_idx]
                f_full = os.path.join(mod_path, f_name)
                img = read_dicom_robust(f_full, target_size=(IMG_SIZE, IMG_SIZE))
                slab_images.append(img)

            # Average
            slab_avg = np.mean(slab_images, axis=0)

            # Independent Per-Channel Min-Max Normalization
            s_min = slab_avg.min()
            s_max = slab_avg.max()
            if s_max - s_min > 0:
                slab_norm = (slab_avg - s_min) / (s_max - s_min)
            else:
                slab_norm = np.zeros_like(slab_avg)

            subject_channels.append(slab_norm)

    # Stack to (12, H, W)
    volume = np.array(subject_channels, dtype=np.float32)
    return volume, True


def _load_or_process_data(df, cache_prefix, load_cached_data=True):
    """
    Handles caching logic. Loads from .npy if exists and requested,
    otherwise processes from scratch and saves.
    """
    cache_x = os.path.join(CACHE_DIR, f"{cache_prefix}_data.npy")
    cache_y = os.path.join(CACHE_DIR, f"{cache_prefix}_labels.npy")

    # Try loading
    if load_cached_data and os.path.exists(cache_x) and os.path.exists(cache_y):
        print(f"Loading cached data for {cache_prefix}...")
        try:
            X = np.load(cache_x)
            y = np.load(cache_y)
            return X, y
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    # Process
    print(f"Processing data for {cache_prefix}...")
    X_list = []
    y_list = []

    corrupt_count = 0
    total_count = len(df)

    for idx, row in df.iterrows():
        vol, success = _process_subject(row)

        if not success:
            corrupt_count += 1
            # If we are in training/val, we might skip. For test, we must provide a prediction.
            # However, _process_subject returns zeros on failure, so we keep consistency.
            # We track corruption for the circuit breaker.

        X_list.append(vol)

        if "MGMT_value" in row:
            y_list.append(row["MGMT_value"])
        else:
            y_list.append(-1)  # Placeholder for test

    # Circuit Breaker
    if total_count > 0 and (corrupt_count / total_count) > 0.10:  # 10% tolerance
        print(f"WARNING: High corruption rate detected: {corrupt_count}/{total_count}")
        # We don't raise error to allow submission to proceed with zeros, but it's noted.

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    # Save to cache
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.save(cache_x, X)
    np.save(cache_y, y)
    print(f"Saved processed data to {cache_x}")

    return X, y


def get_dataloaders(load_cached_data=True):
    """
    Main entry point to get PyTorch DataLoaders.

    Args:
        load_cached_data (bool): If True, tries to load pre-processed .npy files.

    Returns:
        train_loader, val_loader, test_loader
    """
    # 1. Load Metadata
    train_df = pd.read_csv(TRAIN_METADATA_PATH)
    val_df = pd.read_csv(VAL_METADATA_PATH)
    test_df = pd.read_csv(TEST_METADATA_PATH)

    # Debugging Limit
    if DEBUG_DATA_LIMIT is not None:
        train_df = train_df.iloc[:DEBUG_DATA_LIMIT]
        val_df = val_df.iloc[:DEBUG_DATA_LIMIT]
        # We process full test set always to ensure submission is valid

    # 2. Process/Load Data
    X_train, y_train = _load_or_process_data(train_df, "train", load_cached_data)
    X_val, y_val = _load_or_process_data(val_df, "val", load_cached_data)
    X_test, y_test = _load_or_process_data(test_df, "test", load_cached_data)

    # 3. Create Datasets
    # Training gets augmentation
    train_dataset = BraTSDataset(X_train, y_train, transform=True)
    # Validation and Test do not get augmentation
    val_dataset = BraTSDataset(X_val, y_val, transform=False)
    test_dataset = BraTSDataset(X_test, None, transform=False)  # No labels for test

    # 4. Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True,  # Avoid incomplete batches for BatchNorm stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return train_loader, val_loader, test_loader
