import os
import glob
import re
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import configuration and utilities
from library.config import (
    WORKING_DIR,
    IMG_SIZE,
    MODALITIES,
    ROI_DEPTHS,
    INPUT_DROPOUT_PROB,
    NUM_WORKERS,
    SEED,
    INPUT_DIR,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
)
from library.utils import get_logger

# Initialize Logger
logger = get_logger("data_processing")


def read_dicom(path):
    """
    Reads a DICOM file and returns a numpy float32 array.
    Tries pydicom first, then cv2.
    """
    try:
        import pydicom

        dcm = pydicom.dcmread(path)
        img = dcm.pixel_array.astype(np.float32)
        return img
    except (ImportError, Exception):
        pass

    try:
        # cv2.IMREAD_UNCHANGED is crucial for 16-bit DICOMs
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            return img.astype(np.float32)
    except Exception:
        pass

    # Return zeros if read fails (should be caught by integrity checks elsewhere if critical)
    return np.zeros(IMG_SIZE, dtype=np.float32)


def get_roi_bounds(file_paths):
    """
    Determines the start and end indices of the brain ROI within a sorted list of file paths.
    Scans from the start and end until non-zero pixels are found.
    """
    if not file_paths:
        return 0, 0

    n_files = len(file_paths)
    start_idx = 0
    end_idx = n_files - 1

    # Find start: scan forward
    # Limit scan to avoid infinite loops on empty volumes, though unlikely
    found_start = False
    for i in range(n_files):
        img = read_dicom(file_paths[i])
        if np.max(img) > 0:
            start_idx = i
            found_start = True
            break

    # Find end: scan backward
    found_end = False
    for i in range(n_files - 1, -1, -1):
        img = read_dicom(file_paths[i])
        if np.max(img) > 0:
            end_idx = i
            found_end = True
            break

    if not found_start or not found_end or end_idx < start_idx:
        # Fallback to full range if volume is empty
        return 0, n_files - 1

    return start_idx, end_idx


def prepare_slice_cache(df, cache_name, load_cached_data=True):
    """
    Generates or loads a cache containing the specific file paths for the
    required relative depths (40%, 50%, 60%) for each subject and modality.
    """
    cache_path = os.path.join(WORKING_DIR, f"{cache_name}.parquet")

    if load_cached_data and os.path.exists(cache_path):
        logger.info(f"Loading cached slice paths from {cache_path}")
        return pd.read_parquet(cache_path)

    logger.info(f"Generating slice cache for {cache_name}...")

    cache_data = []

    # Iterate over subjects
    for idx, row in df.iterrows():
        subject_id = row["BraTS21ID"]
        record = {"BraTS21ID": subject_id}

        # Preserve target if it exists
        if "MGMT_value" in row:
            record["MGMT_value"] = row["MGMT_value"]

        # Process each modality
        for mod in MODALITIES:  # FLAIR, T1wCE, T2w
            # Construct full directory path
            # metadata contains relative path e.g. 'train/00000/FLAIR'
            rel_path = row[f"{mod.lower()}_path"]
            mod_dir = os.path.join(INPUT_DIR, rel_path)

            # List and sort files
            if os.path.exists(mod_dir):
                files = [f for f in os.listdir(mod_dir) if f.endswith(".dcm")]
                # Sort by image number: Image-10.dcm -> 10
                files.sort(key=lambda x: int(re.search(r"Image-(\d+)", x).group(1)))
                full_paths = [os.path.join(mod_dir, f) for f in files]
            else:
                full_paths = []

            # Determine ROI
            if not full_paths:
                # Missing directory handling: fill with None, will be handled in Dataset
                for depth in ROI_DEPTHS:
                    record[f"{mod}_{depth}_path"] = None
                continue

            start, end = get_roi_bounds(full_paths)
            roi_len = end - start

            # Calculate indices for 0.4, 0.5, 0.6
            for depth in ROI_DEPTHS:
                # Relative index from start
                rel_idx = int(start + roi_len * depth)
                # Clamp to bounds
                rel_idx = max(0, min(len(full_paths) - 1, rel_idx))

                record[f"{mod}_{depth}_path"] = full_paths[rel_idx]

        cache_data.append(record)

        if (idx + 1) % 50 == 0:
            logger.info(f"Processed {idx + 1}/{len(df)} subjects")

    df_cache = pd.DataFrame(cache_data)
    df_cache.to_parquet(cache_path, index=False)
    logger.info(f"Saved slice cache to {cache_path}")

    return df_cache


class RARVDataset(Dataset):
    def __init__(self, df_cache, transforms=None, is_train=False):
        self.df = df_cache
        self.transforms = transforms
        self.is_train = is_train
        self.input_dropout_prob = INPUT_DROPOUT_PROB

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Channels:
        # 0,1,2: FLAIR(0.4), T1wCE(0.4), T2w(0.4)
        # 3,4,5: FLAIR(0.5), T1wCE(0.5), T2w(0.5)
        # 6,7,8: FLAIR(0.6), T1wCE(0.6), T2w(0.6)

        channels = []

        # Order of stacking: Depth 0.4 (All Mods), Depth 0.5 (All Mods), Depth 0.6 (All Mods)
        # This groups them by depth triplet for the dropout logic
        for depth in ROI_DEPTHS:
            for mod in MODALITIES:
                col_name = f"{mod}_{depth}_path"
                path = row.get(col_name)

                if path and isinstance(path, str) and os.path.exists(path):
                    img = read_dicom(path)

                    # Resize
                    img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)

                    # Min-Max Normalize to [0, 1]
                    min_val = img.min()
                    max_val = img.max()
                    if max_val > min_val:
                        img = (img - min_val) / (max_val - min_val)
                    else:
                        img = np.zeros_like(img)
                else:
                    # Fallback for missing files
                    img = np.zeros(IMG_SIZE, dtype=np.float32)

                channels.append(img)

        # Stack to (H, W, C) for Albumentations
        # Resulting order:
        # Ch 0: FLAIR 0.4, Ch 1: T1wCE 0.4, Ch 2: T2w 0.4
        # Ch 3: FLAIR 0.5, Ch 4: T1wCE 0.5, Ch 5: T2w 0.5
        # Ch 6: FLAIR 0.6, Ch 7: T1wCE 0.6, Ch 8: T2w 0.6
        image = np.stack(channels, axis=-1)

        # Apply Augmentations
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]  # Returns Tensor (C, H, W)
        else:
            # Convert to tensor manually if no transforms
            image = torch.from_numpy(image.transpose(2, 0, 1))  # (C, H, W)

        # Structured Input Dropout (Training Only)
        if self.is_train and np.random.rand() < self.input_dropout_prob:
            # 50% chance: Drop Center (Ch 3-5)
            # 50% chance: Drop Peripherals (Ch 0-2, 6-8)
            if np.random.rand() < 0.5:
                # Drop Center
                image[3:6, :, :] = 0
            else:
                # Drop Peripherals
                image[0:3, :, :] = 0
                image[6:9, :, :] = 0

        # Get Label
        label = (
            torch.tensor(row["MGMT_value"], dtype=torch.float32)
            if "MGMT_value" in row
            else torch.tensor(-1.0)
        )

        return image, label, str(int(row["BraTS21ID"]))


def get_dataloaders(batch_size=32, load_cached_data=True):
    """
    Prepares DataLoaders for Train, Validation, and Test sets.
    Handles caching of ROI metadata.
    """

    # 1. Load Metadata CSVs
    df_train_meta = pd.read_csv(TRAIN_METADATA_PATH)
    df_val_meta = pd.read_csv(VAL_METADATA_PATH)
    df_test_meta = pd.read_csv(TEST_METADATA_PATH)

    # 2. Prepare Slice Caches (ROI Detection)
    # This step is computationally intensive on first run, so it's cached.
    df_train_cache = prepare_slice_cache(
        df_train_meta, "roi_cache_train", load_cached_data
    )
    df_val_cache = prepare_slice_cache(df_val_meta, "roi_cache_val", load_cached_data)
    df_test_cache = prepare_slice_cache(
        df_test_meta, "roi_cache_test", load_cached_data
    )

    # 3. Define Augmentations
    # Strictly exclude Translation (Shift) and Scaling to preserve ROI anchoring
    train_transforms = A.Compose(
        [
            A.Rotate(limit=15, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
            ToTensorV2(),
        ]
    )

    val_test_transforms = A.Compose([ToTensorV2()])

    # 4. Create Datasets
    train_dataset = RARVDataset(
        df_train_cache, transforms=train_transforms, is_train=True
    )
    val_dataset = RARVDataset(
        df_val_cache, transforms=val_test_transforms, is_train=False
    )
    test_dataset = RARVDataset(
        df_test_cache, transforms=val_test_transforms, is_train=False
    )

    # 5. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
