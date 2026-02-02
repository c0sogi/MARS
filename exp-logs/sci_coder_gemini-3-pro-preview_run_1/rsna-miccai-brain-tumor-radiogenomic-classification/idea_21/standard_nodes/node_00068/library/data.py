import os
import cv2
import numpy as np
import pandas as pd
import torch
import re
import logging
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Import configuration and utilities
from library.config import (
    INPUT_DIR,
    WORKING_DIR,
    TRAIN_METADATA,
    VAL_METADATA,
    TEST_METADATA,
    IMAGE_SIZE,
    STRIDE,
    NUM_CHANNELS,
    BATCH_SIZE,
    NUM_WORKERS,
    SEED,
    MODALITIES,
)
from library.utils import get_logger

logger = get_logger(__name__)

# Check for pydicom availability
try:
    import pydicom

    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False


def extract_instance_number(filename):
    """Extracts the integer instance number from a DICOM filename (e.g., 'Image-123.dcm')."""
    match = re.search(r"Image-(\d+)\.dcm", filename)
    if match:
        return int(match.group(1))
    return 0


def load_dicom_slice(path, target_size=IMAGE_SIZE):
    """
    Loads a DICOM slice, resizes it to target_size, and applies min-max normalization.
    Returns a float32 numpy array of shape (target_size, target_size).
    """
    img = None
    # Attempt 1: pydicom
    if HAS_PYDICOM:
        try:
            ds = pydicom.dcmread(path)
            img = ds.pixel_array
        except Exception:
            pass

    # Attempt 2: cv2
    if img is None:
        try:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        except Exception:
            pass

    # Fallback: Empty image
    if img is None:
        return np.zeros((target_size, target_size), dtype=np.float32)

    # Resize
    if img.shape[0] != target_size or img.shape[1] != target_size:
        img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)

    # Normalize to [0, 1]
    img = img.astype(np.float32)
    min_val = np.min(img)
    max_val = np.max(img)
    if max_val - min_val > 0:
        img = (img - min_val) / (max_val - min_val)
    else:
        img = np.zeros_like(img)

    return img


def get_subject_volume(row, input_dir, stride=STRIDE):
    """
    Constructs the 9-channel volumetric input for a single subject.
    Structure:
      Channels 0-2: [FLAIR, T1wCE, T2w] at Depth M - stride
      Channels 3-5: [FLAIR, T1wCE, T2w] at Depth M
      Channels 6-8: [FLAIR, T1wCE, T2w] at Depth M + stride
    """
    channels = []

    # Depths relative to median: -stride, 0, +stride
    offsets = [-stride, 0, stride]

    # For each depth offset
    for offset in offsets:
        # For each modality in the specific order
        for mod in MODALITIES:
            # Construct path to modality folder
            # row keys are like 'flair_path', 't1wce_path' (lowercase in metadata usually? let's check metadata script)
            # Metadata script keys: "flair_path", "t1w_path", "t1wce_path", "t2w_path"
            # MODALITIES in config: ["FLAIR", "T1wCE", "T2w"]
            # We need to map config modality name to metadata column name
            mod_key = f"{mod.lower()}_path"

            rel_path = row[mod_key]
            full_path = os.path.join(input_dir, rel_path)

            # List and sort files
            if os.path.exists(full_path):
                files = [f for f in os.listdir(full_path) if f.endswith(".dcm")]
                # Sort numerically
                files.sort(key=extract_instance_number)
            else:
                files = []

            if not files:
                # Handle missing data with black slice
                channels.append(np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32))
                continue

            # Calculate Median Index
            num_files = len(files)
            mid_idx = num_files // 2

            # Calculate target index
            target_idx = mid_idx + offset

            # Clamp index to valid range
            target_idx = max(0, min(target_idx, num_files - 1))

            # Load image
            img_path = os.path.join(full_path, files[target_idx])
            img = load_dicom_slice(img_path, IMAGE_SIZE)
            channels.append(img)

    # Stack channels: (IMAGE_SIZE, IMAGE_SIZE, 9)
    volume = np.stack(channels, axis=-1)
    return volume


def process_dataset(metadata_path, dataset_name, load_cached_data=True):
    """
    Loads metadata, processes images (or loads from cache), and returns arrays.
    dataset_name: 'train', 'val', or 'test' used for cache filenames.
    """
    cache_dir = WORKING_DIR
    cache_imgs_path = os.path.join(cache_dir, f"cached_{dataset_name}_images.npy")
    cache_ids_path = os.path.join(cache_dir, f"cached_{dataset_name}_ids.npy")
    cache_lbls_path = os.path.join(cache_dir, f"cached_{dataset_name}_labels.npy")

    # Check cache
    if load_cached_data:
        if os.path.exists(cache_imgs_path) and os.path.exists(cache_ids_path):
            logger.info(f"Loading {dataset_name} data from cache...")
            images = np.load(cache_imgs_path)
            ids = np.load(cache_ids_path)

            # Labels might not exist for test set
            if os.path.exists(cache_lbls_path):
                labels = np.load(cache_lbls_path)
            else:
                labels = None

            return images, labels, ids
        else:
            logger.info(
                f"Cache not found for {dataset_name}. Processing from scratch..."
            )

    # Load Metadata
    df = pd.read_csv(metadata_path)

    # Prepare lists
    ids = df["BraTS21ID"].values

    # Check if labels exist
    if "MGMT_value" in df.columns:
        labels = df["MGMT_value"].values.astype(np.float32)
    else:
        labels = None

    # Process Images using ThreadPool
    logger.info(f"Processing {len(df)} subjects for {dataset_name}...")

    # Helper for threading
    def _process_row(row):
        return get_subject_volume(row, INPUT_DIR, STRIDE)

    rows = [row for _, row in df.iterrows()]

    with ThreadPoolExecutor(max_workers=NUM_WORKERS * 2) as executor:
        # Use tqdm for progress tracking
        results = list(
            tqdm(executor.map(_process_row, rows), total=len(rows), disable=None)
        )

    images = np.stack(results, axis=0)  # (N, 224, 224, 9)

    # Save to cache
    logger.info(f"Saving {dataset_name} to cache...")
    np.save(cache_imgs_path, images)
    np.save(cache_ids_path, ids)
    if labels is not None:
        np.save(cache_lbls_path, labels)

    return images, labels, ids


class SFWIVDataset(Dataset):
    def __init__(self, images, labels=None, transform=None):
        """
        images: (N, H, W, C) numpy array
        labels: (N,) numpy array or None
        transform: albumentations transform
        """
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Image is (H, W, C)
        image = self.images[idx]

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Fallback to tensor conversion if no transform provided
            image = ToTensorV2()(image=image)["image"]

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return image, label
        else:
            return image


def get_transforms(phase="train"):
    """
    Returns Albumentations transforms.
    Strictly excludes Shift and Scale to preserve spatial priors.
    """
    if phase == "train":
        return A.Compose(
            [
                # Rotate: Limit 15 degrees, p=0.5
                A.Rotate(limit=15, p=0.5),
                # Elastic: alpha=1, sigma=50, alpha_affine=50
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.5),
                # Grid Distortion
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
                # Ensure tensor conversion
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose([ToTensorV2()])


def get_dataloaders(load_cached_data=True):
    """
    Main entry point to get train and validation dataloaders.
    Handles caching, loading, and dataset creation.
    """
    # 1. Process/Load Train Data
    train_imgs, train_lbls, _ = process_dataset(
        TRAIN_METADATA, "train", load_cached_data=load_cached_data
    )

    # 2. Process/Load Val Data
    val_imgs, val_lbls, _ = process_dataset(
        VAL_METADATA, "val", load_cached_data=load_cached_data
    )

    # 3. Create Datasets
    train_dataset = SFWIVDataset(
        train_imgs, train_lbls, transform=get_transforms("train")
    )
    val_dataset = SFWIVDataset(val_imgs, val_lbls, transform=get_transforms("val"))

    # 4. Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader


def get_test_dataloader(load_cached_data=True):
    """
    Returns test dataloader and the list of BraTS21IDs.
    """
    test_imgs, _, test_ids = process_dataset(
        TEST_METADATA, "test", load_cached_data=load_cached_data
    )

    test_dataset = SFWIVDataset(
        test_imgs, labels=None, transform=get_transforms("test")
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return test_loader, test_ids
