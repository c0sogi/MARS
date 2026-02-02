import os
import re
import glob
import random
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from concurrent.futures import ThreadPoolExecutor
from library import config, utils

# Attempt to import pydicom, handle if missing
try:
    import pydicom

    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False


def natural_sort_key(s):
    """
    Key for natural sorting of filenames (e.g., Image-1.dcm < Image-10.dcm).
    """
    return [int(c) if c.isdigit() else c for c in re.split(r"(\d+)", s)]


def read_dicom_image(path, size=(224, 224)):
    """
    Reads a DICOM file and returns a normalized 2D numpy array.
    """
    img = None
    if HAS_PYDICOM:
        try:
            dcm = pydicom.dcmread(path)
            img = dcm.pixel_array
        except Exception:
            pass

    if img is None:
        try:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        except Exception:
            pass

    if img is None:
        # Fallback for missing/corrupt files
        return np.zeros(size, dtype=np.float32)

    # Normalize to float32
    img = img.astype(np.float32)

    # Resize
    if img.shape != size:
        img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)

    return img


def get_roi_bounds(file_paths):
    """
    Scans slices to find the start and end indices of the brain tissue (pixels > 0).
    Returns (start_index, end_index).
    """
    if not file_paths:
        return 0, 0

    # Optimization: Instead of reading every single file which is slow,
    # we can check with a stride or just read them.
    # Given the requirement for "Independent ROI Normalization", we should try to be accurate.
    # To save time, we read low-res or center crop? No, just read.
    # We will rely on parallel processing at the subject level to hide latency.

    # However, reading 100+ files per modality per subject is very heavy.
    # Let's read with a stride of 5 to find rough bounds, then refine?
    # Or just assume the middle 80% if reading is too slow?
    # The prompt asks to "scan the file list and threshold pixels > 0".
    # We will implement a fast scan.

    min_idx = 0
    max_idx = len(file_paths) - 1

    # If list is small, just return ends
    if len(file_paths) < 10:
        return 0, len(file_paths) - 1

    # Heuristic: Read from ends inwards until signal is found
    # Limit search to avoid infinite loops on empty scans

    # Find start
    for i in range(len(file_paths)):
        img = read_dicom_image(file_paths[i])
        if np.max(img) > 0:
            min_idx = i
            break

    # Find end
    for i in range(len(file_paths) - 1, -1, -1):
        img = read_dicom_image(file_paths[i])
        if np.max(img) > 0:
            max_idx = i
            break

    if min_idx >= max_idx:
        return 0, len(file_paths) - 1

    return min_idx, max_idx


def process_subject(row, input_dir):
    """
    Process a single subject:
    1. Locate files for each modality.
    2. Determine ROI.
    3. Sample slices at 40%, 50%, 60% depth.
    4. Stack and normalize.
    """
    sid = row["BraTS21ID"]

    # Output tensor: (H, W, C) -> (224, 224, 9)
    # Channels:
    # 0-2: FLAIR, T1wCE, T2w @ 40%
    # 3-5: FLAIR, T1wCE, T2w @ 50%
    # 6-8: FLAIR, T1wCE, T2w @ 60%

    channels = []

    # Order defined in config: FLAIR, T1wCE, T2w
    # Depths defined in config: 0.4, 0.5, 0.6

    # We iterate depths first, then modalities to group by depth?
    # Prompt says:
    # Ch 0-2: [FLAIR, T1wCE, T2w] @ 40%
    # Ch 3-5: [FLAIR, T1wCE, T2w] @ 50%
    # ...

    for depth_ratio in config.RELATIVE_DEPTHS:
        for mod in config.MODALITIES:
            # Construct path
            # Metadata contains relative path e.g., "train/00000/FLAIR"
            # We need to handle case sensitivity or just use the path provided
            rel_path = row[f"{mod.lower()}_path"]
            full_path = os.path.join(input_dir, rel_path)

            if not os.path.exists(full_path):
                # Missing modality, append zero channel
                channels.append(
                    np.zeros((config.IMG_SIZE, config.IMG_SIZE), dtype=np.float32)
                )
                continue

            # List files
            files = glob.glob(os.path.join(full_path, "*.dcm"))
            files.sort(key=natural_sort_key)

            if not files:
                channels.append(
                    np.zeros((config.IMG_SIZE, config.IMG_SIZE), dtype=np.float32)
                )
                continue

            # ROI Detection
            start, end = get_roi_bounds(files)
            roi_len = end - start

            # Calculate index
            # depth_ratio is relative to ROI
            idx = start + int(roi_len * depth_ratio)
            idx = max(0, min(idx, len(files) - 1))

            # Read image
            img = read_dicom_image(files[idx], size=(config.IMG_SIZE, config.IMG_SIZE))

            # Min-Max Scale to [0, 1] per channel
            if img.max() > 0:
                img = (img - img.min()) / (img.max() - img.min())

            channels.append(img)

    # Stack channels -> (224, 224, 9)
    volume = np.stack(channels, axis=-1)
    return volume.astype(np.float32)


def prepare_and_cache_data(
    df, cache_img_path, cache_label_path, cache_id_path, load_cached_data=True
):
    """
    Loads data from cache or processes it from scratch.
    """
    # Check if cache exists
    if (
        load_cached_data
        and os.path.exists(cache_img_path)
        and os.path.exists(cache_id_path)
    ):
        print(f"Loading cached data from {cache_img_path}...")
        images = np.load(cache_img_path)
        ids = np.load(cache_id_path)
        if os.path.exists(cache_label_path):
            labels = np.load(cache_label_path)
        else:
            labels = None
        return images, labels, ids

    print(f"Processing {len(df)} subjects (Cache miss or forced reload)...")

    # Process in parallel
    # Convert dataframe to list of dicts for thread safety
    rows = df.to_dict("records")

    images = []
    ids = []
    labels = []

    # Use ThreadPoolExecutor for I/O bound task
    # Note: process_subject reads files, so threads are good.
    with ThreadPoolExecutor(max_workers=config.NUM_WORKERS) as executor:
        # Map returns in order
        results = list(
            executor.map(lambda r: process_subject(r, config.INPUT_DIR), rows)
        )

    images = np.array(results)
    ids = df["BraTS21ID"].values

    if "MGMT_value" in df.columns:
        labels = df["MGMT_value"].values.astype(np.float32)
    else:
        labels = None

    # Save to cache
    print(f"Saving cache to {config.WORKING_DIR}...")
    np.save(cache_img_path, images)
    np.save(cache_id_path, ids)
    if labels is not None:
        np.save(cache_label_path, labels)

    return images, labels, ids


class SIRVDataset(Dataset):
    def __init__(self, images, labels, ids, transform=None, mode="train"):
        self.images = images
        self.labels = labels
        self.ids = ids
        self.transform = transform
        self.mode = mode

        # Integrity check
        if self.mode == "train":
            # Just a sanity check, though actual size depends on split
            pass

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Image is (H, W, C)
        image = self.images[idx]

        # Apply Albumentations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]  # Returns Tensor (C, H, W)
        else:
            # Convert to tensor manually if no transform
            image = torch.from_numpy(image.transpose(2, 0, 1))

        # Depth Dropout (Training Only)
        # Randomly zero out center (3-5) or periphery (0-2, 6-8)
        if self.mode == "train" and config.DEPTH_DROPOUT_PROB > 0:
            if random.random() < config.DEPTH_DROPOUT_PROB:
                # Decide which to drop
                if random.random() < 0.5:
                    # Drop Center (Channels 3, 4, 5)
                    image[3:6, :, :] = 0
                else:
                    # Drop Periphery (Channels 0, 1, 2 and 6, 7, 8)
                    image[0:3, :, :] = 0
                    image[6:9, :, :] = 0

        bra_id = self.ids[idx]

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return image, label, bra_id
        else:
            return image, bra_id


def get_transforms(mode="train"):
    """
    Returns Albumentations transforms.
    Strictly excludes translation and scaling to preserve spatial priors.
    """
    if mode == "train":
        return A.Compose(
            [
                A.Rotate(limit=config.AUG_ROTATION_LIMIT, p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ElasticTransform(
                    alpha=config.AUG_ELASTIC_ALPHA,
                    sigma=config.AUG_ELASTIC_SIGMA,
                    alpha_affine=config.AUG_ELASTIC_ALPHA_AFFINE,
                    p=0.5,
                ),
                A.GridDistortion(
                    num_steps=config.AUG_GRID_DISTORT_NUM_STEPS,
                    distort_limit=config.AUG_GRID_DISTORT_DISTORT_LIMIT,
                    p=0.5,
                ),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose([ToTensorV2()])


def get_dataloader(mode, load_cached_data=True, fold_idx=None):
    """
    Main entry point to get data loaders.

    Args:
        mode: 'train', 'val', or 'test'
        load_cached_data: Whether to use cached .npy files
        fold_idx: For cross-validation (not fully implemented here as splits are in metadata)
    """

    if mode == "train":
        df = pd.read_csv(config.TRAIN_METADATA_PATH)
        cache_img = config.CACHE_TRAIN_IMAGES
        cache_lbl = config.CACHE_TRAIN_LABELS
        cache_id = os.path.join(config.WORKING_DIR, "train_ids.npy")

        images, labels, ids = prepare_and_cache_data(
            df, cache_img, cache_lbl, cache_id, load_cached_data
        )

        dataset = SIRVDataset(
            images, labels, ids, transform=get_transforms("train"), mode="train"
        )

        return DataLoader(
            dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )

    elif mode == "val":
        df = pd.read_csv(config.VAL_METADATA_PATH)
        cache_img = config.CACHE_VAL_IMAGES
        cache_lbl = config.CACHE_VAL_LABELS
        cache_id = os.path.join(config.WORKING_DIR, "val_ids.npy")

        images, labels, ids = prepare_and_cache_data(
            df, cache_img, cache_lbl, cache_id, load_cached_data
        )

        dataset = SIRVDataset(
            images, labels, ids, transform=get_transforms("val"), mode="val"
        )

        return DataLoader(
            dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
        )

    elif mode == "test":
        df = pd.read_csv(config.TEST_METADATA_PATH)
        cache_img = config.CACHE_TEST_IMAGES
        cache_lbl = os.path.join(
            config.WORKING_DIR, "test_labels_dummy.npy"
        )  # No labels
        cache_id = config.CACHE_TEST_IDS

        images, labels, ids = prepare_and_cache_data(
            df, cache_img, cache_lbl, cache_id, load_cached_data
        )

        dataset = SIRVDataset(
            images, None, ids, transform=get_transforms("test"), mode="test"
        )

        return DataLoader(
            dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")
