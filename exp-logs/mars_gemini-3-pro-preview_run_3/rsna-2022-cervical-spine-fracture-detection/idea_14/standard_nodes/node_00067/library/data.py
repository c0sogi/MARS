import os
import glob
import numpy as np
import pandas as pd
import pydicom
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from library.config import Config


def process_dicom_study(study_dir, target_size=(256, 256)):
    """
    Reads all DICOMs in a directory, sorts by Z, applies windowing, resizes.
    Returns: numpy array of shape (Depth, H, W) in uint8.
    """
    # List all files
    files = glob.glob(os.path.join(study_dir, "*.dcm"))
    if not files:
        files = glob.glob(os.path.join(study_dir, "*"))

    if not files:
        # Return empty volume if directory is empty (should not happen based on metadata check)
        return np.zeros((Config.NUM_SLICES, *target_size), dtype=np.uint8)

    # Read headers for sorting
    slices = []
    for f in files:
        try:
            # Read only header for speed
            dcm = pydicom.dcmread(f, stop_before_pixels=True)
            # Try Z-position, fallback to InstanceNumber
            if hasattr(dcm, "ImagePositionPatient"):
                z = float(dcm.ImagePositionPatient[2])
            else:
                z = float(dcm.InstanceNumber)
            slices.append((z, f))
        except Exception:
            continue

    # Sort by Z position
    slices.sort(key=lambda x: x[0])

    # Bone Window Settings
    wl = Config.WINDOW_LEVEL
    ww = Config.WINDOW_WIDTH
    lower_bound = wl - ww / 2
    upper_bound = wl + ww / 2

    processed_volume = []

    for _, f_path in slices:
        try:
            dcm = pydicom.dcmread(f_path)

            # HU Calibration
            slope = float(getattr(dcm, "RescaleSlope", 1.0))
            intercept = float(getattr(dcm, "RescaleIntercept", 0.0))

            img = dcm.pixel_array.astype(np.float32)
            img = img * slope + intercept

            # Windowing
            img = np.clip(img, lower_bound, upper_bound)

            # Normalize to 0-255 uint8
            img = (img - lower_bound) / (upper_bound - lower_bound)
            img = (img * 255.0).astype(np.uint8)

            # Resize
            if img.shape[:2] != target_size:
                img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)

            processed_volume.append(img)
        except Exception:
            continue

    if not processed_volume:
        return np.zeros((Config.NUM_SLICES, *target_size), dtype=np.uint8)

    return np.stack(processed_volume, axis=0)  # (D, H, W)


def load_volume(study_uid, image_rel_path, load_cached_data=True):
    """
    Loads volume from cache or processes from DICOMs.
    """
    cache_path = os.path.join(Config.CACHE_DIR, f"{study_uid}.npy")

    # 1. Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            volume = np.load(cache_path)
            # Verify consistency with current runtime config (Cite debug_lesson_2)
            if volume.shape[1:] == Config.IMAGE_SIZE:
                return volume
        except Exception:
            pass  # Fallback to processing if load fails

    # 2. Process from scratch
    full_path = os.path.join(Config.INPUT_DIR, image_rel_path)
    if not os.path.exists(full_path):
        # Fallback for test sets where path might differ or if path is incorrect
        # Try finding the folder by UID in train/test images
        if os.path.exists(os.path.join(Config.TRAIN_IMAGES_DIR, study_uid)):
            full_path = os.path.join(Config.TRAIN_IMAGES_DIR, study_uid)
        elif os.path.exists(os.path.join(Config.TEST_IMAGES_DIR, study_uid)):
            full_path = os.path.join(Config.TEST_IMAGES_DIR, study_uid)

    volume = process_dicom_study(full_path, Config.IMAGE_SIZE)

    # 3. Save to cache
    try:
        np.save(cache_path, volume)
    except Exception:
        pass  # Ignore save errors (disk full, etc)

    return volume


class RSNADataset(Dataset):
    def __init__(self, df, phase="train", transform=None):
        self.df = df.reset_index(drop=True)
        self.phase = phase
        self.transform = transform
        self.has_labels = "patient_overall" in df.columns

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        study_uid = row["StudyInstanceUID"]
        image_path = row["image_path"]

        # Load Volume (D, H, W) uint8
        volume = load_volume(study_uid, image_path, load_cached_data=True)

        current_depth = volume.shape[0]
        target_depth = Config.NUM_SLICES

        # Uniform Sampling
        if current_depth > target_depth:
            indices = np.linspace(0, current_depth - 1, target_depth).astype(int)
        else:
            # If smaller, we still use linspace which repeats indices
            indices = np.linspace(0, current_depth - 1, target_depth).astype(int)

        # Construct 2.5D Stacks (z-1, z, z+1)
        # We want output shape (N, 3, H, W)

        # Gather slices
        selected_slices = []
        for i in indices:
            s_prev = max(0, i - 1)
            s_curr = i
            s_next = min(current_depth - 1, i + 1)

            # Stack channel-wise: (H, W, 3)
            stack = np.stack([volume[s_prev], volume[s_curr], volume[s_next]], axis=-1)
            selected_slices.append(stack)

        # Combine into (N, H, W, 3)
        volume_stacked = np.stack(selected_slices, axis=0)
        N, H, W, C = volume_stacked.shape

        # Reshape to (H, W, N*3) for volumetric-consistent augmentation
        # Albumentations applies spatial transforms to H,W and preserves channels
        volume_flat = volume_stacked.reshape(H, W, N * C)

        # Apply Augmentation
        if self.transform:
            augmented = self.transform(image=volume_flat)["image"]
            volume_flat = augmented

        # Convert to Tensor and Normalize
        # (H, W, N*3) -> (N*3, H, W)
        volume_flat = volume_flat.astype(np.float32) / 255.0
        volume_tensor = torch.from_numpy(volume_flat).permute(2, 0, 1)

        # Reshape back to (N, 3, H, W)
        # The channel order is preserved: [s0_c0, s0_c1, s0_c2, s1_c0...]
        volume_reshaped = volume_tensor.view(N, C, H, W)

        # Prepare Targets
        if self.has_labels:
            # Order: C1-C7, then Patient Overall
            labels = [
                row["C1"],
                row["C2"],
                row["C3"],
                row["C4"],
                row["C5"],
                row["C6"],
                row["C7"],
                row["patient_overall"],
            ]
            targets = torch.tensor(labels, dtype=torch.float32)
            return volume_reshaped, targets
        else:
            # Return dummy targets for test set
            return volume_reshaped, torch.zeros(Config.NUM_CLASSES)


def get_transforms(phase):
    if phase == "train":
        return A.Compose(
            [
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.1,
                    rotate_limit=15,
                    p=0.5,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                ),
            ]
        )
    else:
        return None


def get_dataloaders(train_df, val_df, test_df=None):
    """
    Creates DataLoaders for train, validation, and optional test sets.
    Handles debug sampling if Config.DEBUG is True.
    """

    # Debug Subsampling
    if Config.DEBUG:
        train_df = train_df.head(Config.DEBUG_SAMPLE_SIZE)
        val_df = val_df.head(Config.DEBUG_SAMPLE_SIZE)
        if test_df is not None:
            test_df = test_df.head(Config.DEBUG_SAMPLE_SIZE)

    # Datasets
    train_ds = RSNADataset(train_df, phase="train", transform=get_transforms("train"))
    val_ds = RSNADataset(val_df, phase="val", transform=get_transforms("val"))

    # Loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = None
    if test_df is not None:
        test_ds = RSNADataset(test_df, phase="test", transform=get_transforms("test"))
        test_loader = DataLoader(
            test_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

    return train_loader, val_loader, test_loader
