import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from concurrent.futures import ThreadPoolExecutor

from library.config import Config
from library.utils import read_dicom_robust, resize_image, normalize_image, set_seed

# ------------------------------------------------------------------------------
# Helper Functions for Data Processing
# ------------------------------------------------------------------------------


def get_sorted_file_ids(folder_path):
    """
    Returns a sorted list of integer IDs from DICOM files in a folder.
    Files are expected to be named like 'Image-123.dcm'.
    """
    if not os.path.exists(folder_path):
        return []

    files = [f for f in os.listdir(folder_path) if f.endswith(".dcm")]
    ids = []
    for f in files:
        try:
            # Extract number from "Image-X.dcm"
            # Some files might have different naming, handle robustly
            name_part = os.path.splitext(f)[0]
            if "-" in name_part:
                num = int(name_part.split("-")[-1])
                ids.append(num)
            elif name_part.isdigit():
                ids.append(int(name_part))
        except ValueError:
            continue

    return sorted(ids)


def load_slice_robust(folder_path, file_id):
    """
    Constructs filename from ID and loads it.
    """
    # Try standard naming convention
    fname = f"Image-{file_id}.dcm"
    path = os.path.join(folder_path, fname)
    if os.path.exists(path):
        return read_dicom_robust(path)

    # Fallback: iterate to find file if naming is different (unlikely based on dataset desc)
    return np.zeros((Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)


def process_subject(row, input_dir):
    """
    Process a single subject row from metadata.
    Returns volume (H, W, 12) or None if corrupt.
    """
    braTS21ID = row["BraTS21ID"]

    # Paths
    paths = {
        "FLAIR": os.path.join(input_dir, row["path_FLAIR"]),
        "T1w": os.path.join(input_dir, row["path_T1w"]),
        "T1wCE": os.path.join(input_dir, row["path_T1wCE"]),
        "T2w": os.path.join(input_dir, row["path_T2w"]),
    }

    # 1. Anchor Selection using FLAIR
    flair_ids = get_sorted_file_ids(paths["FLAIR"])
    if not flair_ids:
        return None  # Corrupt

    num_slices = len(flair_ids)
    start_idx = int(num_slices * Config.ROI_DEPTH_MIN)
    end_idx = int(num_slices * Config.ROI_DEPTH_MAX)

    # Ensure valid range
    if start_idx >= end_idx:
        start_idx = 0
        end_idx = num_slices

    search_ids = flair_ids[start_idx:end_idx]
    if not search_ids:
        search_ids = flair_ids  # Fallback to all

    max_intensity = -1.0
    anchor_id = flair_ids[num_slices // 2]  # Default to middle

    # Find max intensity slice
    for fid in search_ids:
        img = load_slice_robust(paths["FLAIR"], fid)
        intensity = np.sum(img)
        if intensity > max_intensity:
            max_intensity = intensity
            anchor_id = fid

    # 2. Stacking (Cite Lesson 00118: Early Fusion)
    # Modalities order
    modalities = ["FLAIR", "T1w", "T1wCE", "T2w"]

    # Offsets (Cite Lesson 00110: Fixed Stride)
    offsets = [-Config.STRIDE, 0, Config.STRIDE]

    channels = []

    # Cache available IDs for all modalities to avoid repeated OS calls
    mod_ids = {m: get_sorted_file_ids(paths[m]) for m in modalities}

    # Helper to get slice
    def get_slice_for_modality(mod, target_id):
        available_ids = mod_ids[mod]
        if not available_ids:
            return np.zeros((Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)

        # Edge Clamping
        if target_id < available_ids[0]:
            clamped_id = available_ids[0]
        elif target_id > available_ids[-1]:
            clamped_id = available_ids[-1]
        else:
            # Check if exact ID exists, else find nearest
            if target_id in available_ids:
                clamped_id = target_id
            else:
                # Find nearest
                arr = np.array(available_ids)
                idx = (np.abs(arr - target_id)).argmin()
                clamped_id = arr[idx]

        img = load_slice_robust(paths[mod], clamped_id)
        img = resize_image(img, size=(Config.IMG_SIZE, Config.IMG_SIZE))
        img = normalize_image(img)
        return img

    # Build Stack
    # Structure: [Mod1_S1, Mod1_S2, Mod1_S3, Mod2_S1, ...]
    for mod in modalities:
        for offset in offsets:
            img = get_slice_for_modality(mod, anchor_id + offset)
            channels.append(img)

    # Stack to numpy (H, W, C)
    vol = np.stack(channels, axis=-1)  # (224, 224, 12)

    return vol


def process_and_cache_data(
    metadata_path,
    cache_data_path,
    cache_labels_path=None,
    cache_ids_path=None,
    load_cached=True,
):
    """
    Processes dataset defined in metadata CSV and caches to NPY.
    """
    # 1. Try Load
    if load_cached:
        if os.path.exists(cache_data_path):
            print(f"Loading cached data from {cache_data_path}...")
            data = np.load(cache_data_path)

            labels = None
            if cache_labels_path and os.path.exists(cache_labels_path):
                labels = np.load(cache_labels_path)

            ids = None
            if cache_ids_path and os.path.exists(cache_ids_path):
                ids = np.load(cache_ids_path)

            return data, labels, ids

    # 2. Process
    print(f"Processing data from {metadata_path}...")
    df = pd.read_csv(metadata_path)

    # Limit for debugging
    if Config.DEBUG_SAMPLE_SIZE is not None:
        df = df.head(Config.DEBUG_SAMPLE_SIZE)

    processed_data = []
    processed_labels = []
    processed_ids = []

    corruption_count = 0

    # Sequential processing
    for idx, row in df.iterrows():
        result = process_subject(row, Config.INPUT_DIR)

        if result is None:
            corruption_count += 1
            continue

        vol = result
        processed_data.append(vol)
        processed_ids.append(row["BraTS21ID"])

        if "MGMT_value" in row:
            processed_labels.append(row["MGMT_value"])

    # Circuit Breaker
    corruption_rate = corruption_count / len(df) if len(df) > 0 else 0
    if corruption_rate > Config.CORRUPTION_THRESHOLD:
        raise RuntimeError(
            f"Data Corruption Circuit Breaker Triggered! {corruption_rate:.2%} of data is corrupt."
        )

    # Convert to numpy arrays
    # Data shape: (N, 224, 224, 12)
    final_data = np.array(processed_data, dtype=np.float32)

    if processed_labels:
        final_labels = np.array(processed_labels, dtype=np.float32).reshape(-1, 1)
    else:
        final_labels = None

    final_ids = np.array(processed_ids, dtype=np.int64)

    # 3. Save
    os.makedirs(os.path.dirname(cache_data_path), exist_ok=True)
    np.save(cache_data_path, final_data)
    if cache_ids_path:
        np.save(cache_ids_path, final_ids)
    if final_labels is not None and cache_labels_path:
        np.save(cache_labels_path, final_labels)

    print(f"Cached data saved to {cache_data_path}. Shape: {final_data.shape}")

    return final_data, final_labels, final_ids


# ------------------------------------------------------------------------------
# Dataset Class
# ------------------------------------------------------------------------------


class MGMTDataset(Dataset):
    def __init__(self, data, labels=None, mode="train", transform=None):
        """
        data: numpy array of shape (N, H, W, 12)
        labels: numpy array of shape (N, 1) or None
        """
        self.data = data
        self.labels = labels
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Shape: (H, W, 12)
        img = self.data[idx]

        # Apply Augmentations
        if self.transform:
            augmented = self.transform(image=img)["image"]
            # Albumentations with ToTensorV2 returns (C, H, W)
            img_tensor = augmented
        else:
            # Convert to tensor (C, H, W)
            img_tensor = torch.from_numpy(img.transpose(2, 0, 1))

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return img_tensor, label
        else:
            return img_tensor


# ------------------------------------------------------------------------------
# DataLoader Factory
# ------------------------------------------------------------------------------


def get_dataloaders(load_cached=True):
    """
    Prepares data and returns dataloaders for train, val, and test.
    """
    set_seed(Config.SEED)

    # 1. Define Transforms
    # Train: Flip, Rotate with Reflection
    train_transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(
                limit=Config.ROTATION_DEGREES, border_mode=cv2.BORDER_REFLECT, p=0.5
            ),
            ToTensorV2(),
        ]
    )

    # Val/Test: Just ToTensor (Normalization is done in preprocessing)
    val_transform = A.Compose([ToTensorV2()])

    # 2. Process/Load Data
    # Train
    train_data, train_labels, _ = process_and_cache_data(
        Config.METADATA_TRAIN,
        Config.CACHE_TRAIN_DATA,
        Config.CACHE_TRAIN_LABELS,
        load_cached=load_cached,
    )

    # Val
    val_data, val_labels, _ = process_and_cache_data(
        Config.METADATA_VAL,
        Config.CACHE_VAL_DATA,
        Config.CACHE_VAL_LABELS,
        load_cached=load_cached,
    )

    # Test
    test_data, _, test_ids = process_and_cache_data(
        Config.METADATA_TEST,
        Config.CACHE_TEST_DATA,
        cache_ids_path=Config.CACHE_TEST_IDS,
        load_cached=load_cached,
    )

    # 3. Create Datasets
    train_dataset = MGMTDataset(
        train_data, train_labels, mode="train", transform=train_transform
    )
    val_dataset = MGMTDataset(val_data, val_labels, mode="val", transform=val_transform)
    test_dataset = MGMTDataset(
        test_data, labels=None, mode="test", transform=val_transform
    )

    # 4. Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return train_loader, val_loader, test_loader, test_ids
