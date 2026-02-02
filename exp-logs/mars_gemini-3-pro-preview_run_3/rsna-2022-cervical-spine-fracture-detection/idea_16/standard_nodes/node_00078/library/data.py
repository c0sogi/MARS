import os
import glob
import numpy as np
import pandas as pd
import pydicom
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config

# ====================================================
# DICOM Processing & Caching
# ====================================================


def load_dicom_volume(path):
    """
    Reads a directory of DICOM files, sorts them by Z-position,
    applies HU conversion and Bone Windowing, and resizes.

    Returns:
        np.ndarray: Volume of shape (Depth, H, W) in uint8 [0, 255].
    """
    # 1. List files
    files = glob.glob(os.path.join(path, "*.dcm"))
    if not files:
        # Fallback for empty or malformed directories
        return np.zeros(
            (Config.NUM_SLICES, Config.IMAGE_SIZE, Config.IMAGE_SIZE), dtype=np.uint8
        )

    # 2. Read headers for sorting
    slices = []
    for f in files:
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True)
            # We need ImagePositionPatient for Z-ordering
            # Some files might miss this, handle gracefully by defaulting to 0 or filename
            pos = (
                ds.ImagePositionPatient[2] if hasattr(ds, "ImagePositionPatient") else 0
            )
            slices.append((pos, f))
        except Exception:
            continue

    # Sort by Z position to ensure physical continuity
    slices.sort(key=lambda x: x[0])
    sorted_files = [x[1] for x in slices]

    # 3. Read pixels and process
    processed_slices = []

    # Pre-calculate windowing limits
    center = Config.BONE_WINDOW_LEVEL
    width = Config.BONE_WINDOW_WIDTH
    lower = center - (width / 2)
    upper = center + (width / 2)

    for f in sorted_files:
        try:
            ds = pydicom.dcmread(f)
            img = ds.pixel_array.astype(np.float32)

            # Apply Rescale Slope/Intercept (HU Conversion)
            slope = getattr(ds, "RescaleSlope", 1.0)
            intercept = getattr(ds, "RescaleIntercept", 0.0)
            img = img * slope + intercept

            # Apply Bone Window
            img = np.clip(img, lower, upper)

            # Normalize to 0-1 then 0-255
            img = (img - lower) / (upper - lower)
            img = (img * 255.0).astype(np.uint8)

            # Resize
            if img.shape[0] != Config.IMAGE_SIZE or img.shape[1] != Config.IMAGE_SIZE:
                img = cv2.resize(
                    img,
                    (Config.IMAGE_SIZE, Config.IMAGE_SIZE),
                    interpolation=cv2.INTER_LINEAR,
                )

            processed_slices.append(img)

        except Exception:
            # Skip corrupt slices
            continue

    if not processed_slices:
        return np.zeros(
            (Config.NUM_SLICES, Config.IMAGE_SIZE, Config.IMAGE_SIZE), dtype=np.uint8
        )

    volume = np.stack(processed_slices)  # (D, H, W)
    return volume


def process_and_cache_volume(study_id, relative_path, load_cached_data=True):
    """
    Handles caching logic. Checks for existing .npy file, otherwise processes DICOMs.
    """
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(Config.CACHE_DIR, f"{study_id}.npy")

    # 1. Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            return np.load(cache_path)
        except Exception:
            pass  # Fallback to processing if load fails

    # 2. Process from scratch
    full_path = os.path.join(Config.INPUT_DIR, relative_path)
    volume = load_dicom_volume(full_path)

    # 3. Save to cache
    try:
        np.save(cache_path, volume)
    except Exception:
        pass  # If save fails (e.g. disk full), just return volume

    return volume


# ====================================================
# Transforms
# ====================================================


def get_transforms(data_type="train"):
    """
    Returns Albumentations ReplayCompose transforms.
    ReplayCompose is essential for applying the exact same geometric
    augmentation to every slice in the 2.5D stack sequence.
    """
    if data_type == "train":
        return A.ReplayCompose(
            [
                # Volumetric-consistent ShiftScaleRotate
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.2,
                    rotate_limit=15,
                    p=0.5,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                ),
                # Standard ImageNet normalization
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ]
        )
    else:
        return A.ReplayCompose(
            [
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ]
        )


# ====================================================
# Dataset
# ====================================================


class CervicalDataset(Dataset):
    def __init__(self, df, transforms=None, load_cached_data=True):
        self.df = df
        self.transforms = transforms
        self.load_cached_data = load_cached_data

        # Pre-extract paths and labels to avoid pandas overhead in loop
        self.study_ids = self.df["StudyInstanceUID"].values
        self.image_paths = self.df["image_path"].values

        # Labels are only present for train/val
        self.has_labels = "patient_overall" in self.df.columns
        if self.has_labels:
            # Order: C1-C7, patient_overall
            label_cols = [f"C{i}" for i in range(1, 8)] + ["patient_overall"]
            self.labels = self.df[label_cols].values.astype(np.float32)
        else:
            self.labels = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        study_id = self.study_ids[idx]
        rel_path = self.image_paths[idx]

        # 1. Load Volume (D, H, W)
        volume = process_and_cache_volume(study_id, rel_path, self.load_cached_data)
        depth = volume.shape[0]

        # 2. Uniform Sampling
        # We need exactly Config.NUM_SLICES
        if depth == 0:
            indices = np.zeros(Config.NUM_SLICES, dtype=int)
        else:
            # Uniformly sample indices across the volume depth
            indices = np.linspace(0, depth - 1, Config.NUM_SLICES).astype(int)

        # 3. Construct 2.5D Stacks
        # Input to model: (Batch=NUM_SLICES, Channels=3, H, W)
        # We construct a list of (H, W, 3) images

        stacked_images = []
        for i in indices:
            # Get neighbor indices with boundary handling (clamp)
            idx_prev = max(0, i - 1)
            idx_curr = i
            idx_next = min(depth - 1, i + 1)

            # Extract slices
            s_prev = volume[idx_prev]
            s_curr = volume[idx_curr]
            s_next = volume[idx_next]

            # Stack: (H, W, 3)
            stack = np.stack([s_prev, s_curr, s_next], axis=-1)
            stacked_images.append(stack)

        # 4. Augmentation (Volumetric-Consistent)
        # We need to apply the SAME transform to all slices in the bag

        if self.transforms:
            # Use ReplayCompose logic
            # Apply to first slice to generate parameters
            first_img = stacked_images[0]
            res = self.transforms(image=first_img)
            replay_data = res.get("replay")

            augmented_images = [res["image"]]

            # Apply replay to the rest of the slices
            for img in stacked_images[1:]:
                res_i = self.transforms.replay(replay_data, image=img)
                augmented_images.append(res_i["image"])

            # Stack into tensor: (NUM_SLICES, 3, H, W)
            # Albumentations ToTensorV2 converts to (3, H, W)
            batch_tensor = torch.stack(augmented_images)

        else:
            # Fallback manual conversion if no transform provided
            processed = []
            for img in stacked_images:
                # H, W, 3 -> 3, H, W and div 255
                t = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
                processed.append(t)
            batch_tensor = torch.stack(processed)

        # 5. Return
        if self.has_labels:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return batch_tensor, label
        else:
            # Return dummy label for test set
            return batch_tensor, torch.zeros(Config.NUM_CLASSES)


# ====================================================
# Data Loaders
# ====================================================


def get_dataloaders(train_df, val_df, test_df):
    """
    Creates DataLoaders for train, validation, and test sets.
    """

    # Train Loader
    train_ds = CervicalDataset(
        train_df, transforms=get_transforms("train"), load_cached_data=True
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    # Val Loader
    val_ds = CervicalDataset(
        val_df, transforms=get_transforms("val"), load_cached_data=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    # Test Loader
    test_ds = CervicalDataset(
        test_df, transforms=get_transforms("test"), load_cached_data=True
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
