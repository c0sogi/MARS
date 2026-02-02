import os
import re
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library import config
from library import utils

# Attempt to import pydicom for robust DICOM reading
try:
    import pydicom

    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False


def natural_sort_key(s):
    """
    Key for natural sorting of filenames (e.g., Image-1, Image-2, Image-10).
    Extracts integers from strings to sort numerically rather than lexicographically.
    """
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split("([0-9]+)", s)
    ]


def read_dicom(path, target_size=(224, 224)):
    """
    Reads a DICOM file and resizes it to the target dimensions.
    Returns a float32 numpy array.
    """
    img = None

    # Method 1: pydicom (Preferred for DICOM)
    if HAS_PYDICOM:
        try:
            dcm = pydicom.dcmread(path)
            img = dcm.pixel_array.astype(np.float32)
        except Exception:
            pass

    # Method 2: cv2 (Fallback)
    if img is None:
        try:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                img = img.astype(np.float32)
        except Exception:
            pass

    # Fallback: Return black image if reading fails
    if img is None:
        img = np.zeros(target_size, dtype=np.float32)

    # Resize to target dimensions
    try:
        if img.shape[:2] != target_size:
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
    except Exception:
        img = np.zeros(target_size, dtype=np.float32)

    return img


def get_sorted_files(folder_path):
    """
    Returns a numerically sorted list of .dcm files in a folder.
    """
    if not os.path.exists(folder_path):
        return []
    files = glob.glob(os.path.join(folder_path, "*.dcm"))
    files.sort(key=natural_sort_key)
    return files


def construct_montage(row, input_dir, slice_size, grid_size, slice_depths, modalities):
    """
    Constructs the 2x2 montage for a single subject.

    Args:
        row: Pandas Series containing subject paths.
        input_dir: Root directory of the dataset.
        slice_size: Height/Width of a single slice (e.g., 224).
        grid_size: Number of slices per row/col (e.g., 2).
        slice_depths: List of percentiles for slice selection.
        modalities: List of modality names (FLAIR, T1wCE, T2w).

    Returns:
        (H_total, W_total, C) numpy array, float32, normalized to [0,1].
    """
    montage_h = slice_size * grid_size
    montage_w = slice_size * grid_size
    n_channels = len(modalities)

    # Initialize montage canvas
    montage = np.zeros((montage_h, montage_w, n_channels), dtype=np.float32)

    # Pre-fetch sorted file lists for each modality
    modality_files = {}
    for mod in modalities:
        col_name = f"{mod.lower()}_path"
        full_path = os.path.join(input_dir, row[col_name])
        modality_files[mod] = get_sorted_files(full_path)

    # Define grid positions: (0,0), (0,1), (1,0), (1,1)
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for idx, depth_pct in enumerate(slice_depths):
        if idx >= len(positions):
            break

        r, c = positions[idx]
        y_start = r * slice_size
        x_start = c * slice_size

        for c_idx, mod in enumerate(modalities):
            files = modality_files[mod]
            n_files = len(files)

            if n_files > 0:
                # Deterministic strided sampling
                file_idx = int(n_files * depth_pct)
                file_idx = min(file_idx, n_files - 1)
                file_path = files[file_idx]

                # Read and resize slice
                img_slice = read_dicom(file_path, (slice_size, slice_size))

                # Place in the appropriate channel and grid position
                montage[
                    y_start : y_start + slice_size,
                    x_start : x_start + slice_size,
                    c_idx,
                ] = img_slice

    # Instance-level Min-Max Normalization
    min_val = montage.min()
    max_val = montage.max()

    if max_val - min_val > 1e-8:
        montage = (montage - min_val) / (max_val - min_val)
    else:
        # Avoid division by zero for constant images
        pass

    return montage


def process_dataset(df, cache_name, load_cached=True):
    """
    Processes the dataset: loads images, builds montages, and caches results to disk.

    Args:
        df: Metadata DataFrame.
        cache_name: Unique identifier for the cache file (e.g., 'train_montage').
        load_cached: Boolean to enable loading from existing cache.

    Returns:
        images (N, H, W, C), targets (N,), ids (N,)
    """
    cache_dir = config.WORKING_DIR
    images_path = os.path.join(cache_dir, f"{cache_name}_images.npy")
    targets_path = os.path.join(cache_dir, f"{cache_name}_targets.npy")
    ids_path = os.path.join(cache_dir, f"{cache_name}_ids.npy")

    # 1. Try loading from cache
    if load_cached and os.path.exists(images_path) and os.path.exists(ids_path):
        has_targets_file = os.path.exists(targets_path)
        needs_targets = "MGMT_value" in df.columns

        # Load if targets match requirements (present if needed, or not needed)
        if (needs_targets and has_targets_file) or (not needs_targets):
            print(f"Loading cached data: {cache_name}")
            images = np.load(images_path)
            ids = np.load(ids_path)
            targets = np.load(targets_path) if has_targets_file else None
            return images, targets, ids

    # 2. Process from scratch
    print(f"Generating dataset from scratch: {cache_name}")

    n_samples = len(df)
    h = config.SLICE_SIZE * config.GRID_SIZE
    w = config.SLICE_SIZE * config.GRID_SIZE
    c = len(config.SELECTED_MODALITIES)

    # Pre-allocate arrays
    images = np.zeros((n_samples, h, w, c), dtype=np.float32)
    ids = np.zeros(n_samples, dtype=np.int64)

    has_targets = "MGMT_value" in df.columns
    targets = np.zeros(n_samples, dtype=np.float32) if has_targets else None

    for i, row in df.iterrows():
        img = construct_montage(
            row,
            config.INPUT_DIR,
            config.SLICE_SIZE,
            config.GRID_SIZE,
            config.SLICE_DEPTHS,
            config.SELECTED_MODALITIES,
        )

        images[i] = img
        ids[i] = row["BraTS21ID"]

        if has_targets:
            targets[i] = row["MGMT_value"]

    # 3. Save to cache
    np.save(images_path, images)
    np.save(ids_path, ids)
    if targets is not None:
        np.save(targets_path, targets)

    return images, targets, ids


class BraTSMontageDataset(Dataset):
    def __init__(self, images, targets=None, ids=None, transform=None):
        self.images = images
        self.targets = targets
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Image is (H, W, C) float32 [0,1]
        image = self.images[idx]

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            image = ToTensorV2()(image=image)["image"]

        subject_id = self.ids[idx]

        if self.targets is not None:
            # Return target as (1,) tensor for BCEWithLogitsLoss
            target = torch.tensor(self.targets[idx], dtype=torch.float32).unsqueeze(0)
            return image, target, subject_id
        else:
            return image, subject_id


def get_dataloaders(load_cached=True):
    """
    Main function to prepare data and return dataloaders for Train, Val, and Test.
    """
    # 1. Load Metadata
    df_train = pd.read_csv(config.TRAIN_METADATA_PATH)
    df_val = pd.read_csv(config.VAL_METADATA_PATH)
    df_test = pd.read_csv(config.TEST_METADATA_PATH)

    # 2. Process Data (Load from cache or compute)
    train_imgs, train_targets, train_ids = process_dataset(
        df_train, "train_slice", load_cached=load_cached
    )
    val_imgs, val_targets, val_ids = process_dataset(
        df_val, "val_slice", load_cached=load_cached
    )
    test_imgs, _, test_ids = process_dataset(
        df_test, "test_slice", load_cached=load_cached
    )

    # 3. Define Transforms (Albumentations)
    train_transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.OneOf(
                [
                    A.ElasticTransform(p=1.0),
                    A.GridDistortion(p=1.0),
                ],
                p=0.5,
            ),
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
            ToTensorV2(),
        ]
    )

    val_transform = A.Compose([ToTensorV2()])

    # 4. Create Datasets
    train_dataset = BraTSMontageDataset(
        train_imgs, train_targets, train_ids, transform=train_transform
    )
    val_dataset = BraTSMontageDataset(
        val_imgs, val_targets, val_ids, transform=val_transform
    )
    test_dataset = BraTSMontageDataset(
        test_imgs, None, test_ids, transform=val_transform
    )

    # 5. Create Loaders
    # Use generator and worker_init_fn for reproducibility (Cite Lesson 00011)
    g = torch.Generator()
    g.manual_seed(config.SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=utils.seed_worker,
        generator=g,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
