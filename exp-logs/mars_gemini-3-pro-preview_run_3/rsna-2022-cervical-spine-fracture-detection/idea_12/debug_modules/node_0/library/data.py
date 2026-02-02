import os
import glob
import numpy as np
import pandas as pd
import torch
import cv2
import pydicom
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import load_dicom


class RSNADataset(Dataset):
    def __init__(self, df, subset="train", transform=None):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe.
            subset (str): 'train', 'val', or 'test'.
            transform (bool): Whether to apply augmentation (usually True for train).
        """
        self.df = df
        self.subset = subset
        self.transform = transform

        # Target columns order
        self.target_cols = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "patient_overall"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        study_id = row["StudyInstanceUID"]

        # Construct full path to image directory
        # Metadata contains relative path, e.g., "train_images/..."
        image_dir = os.path.join(Config.INPUT_DIR, row["image_path"])

        # Load processed volume (D, H, W)
        # We use the cache mechanism implemented in process_scan
        volume = process_scan(study_id, image_dir, load_cached_data=Config.USE_CACHE)

        # --- Sampling ---
        # We need exactly Config.NUM_SLICES (64) slices.
        current_depth = volume.shape[0]
        target_depth = Config.NUM_SLICES

        if current_depth == target_depth:
            indices = np.arange(current_depth)
        elif current_depth < target_depth:
            # Pad by repeating
            indices = (
                np.linspace(0, current_depth - 1, target_depth).round().astype(int)
            )
        else:
            # Downsample uniformly
            indices = (
                np.linspace(0, current_depth - 1, target_depth).round().astype(int)
            )

        # --- 2.5D Stacking ---
        # Output shape: (Num_Slices, 3, H, W)
        # Channels: [z-1, z, z+1]

        stacked_images = np.zeros(
            (target_depth, Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8
        )

        for i, z_idx in enumerate(indices):
            # Center slice
            img_z = volume[z_idx]

            # Previous slice (handle boundary)
            prev_idx = max(0, z_idx - 1)
            img_prev = volume[prev_idx]

            # Next slice (handle boundary)
            next_idx = min(current_depth - 1, z_idx + 1)
            img_next = volume[next_idx]

            # Stack channels (H, W, 3)
            stacked_images[i, ..., 0] = img_prev
            stacked_images[i, ..., 1] = img_z
            stacked_images[i, ..., 2] = img_next

        # --- Augmentation (Volumetric-Consistent) ---
        if self.subset == "train" and self.transform:
            stacked_images = self.apply_augmentation(stacked_images)

        # --- Normalization & Tensor Conversion ---
        # Normalize 0-255 -> 0.0-1.0
        stacked_images = stacked_images.astype(np.float32) / 255.0

        # Rearrange to (Batch/Seq, Channels, H, W) -> (64, 3, 224, 224)
        stacked_images = np.transpose(stacked_images, (0, 3, 1, 2))

        image_tensor = torch.from_numpy(stacked_images)

        # --- Labels ---
        if self.subset != "test":
            labels = row[self.target_cols].values.astype(np.float32)
            return image_tensor, torch.tensor(labels)
        else:
            # For test set, return dummy labels or just the ID for tracking if needed
            # Returning zeros matching label shape
            return image_tensor, torch.zeros(len(self.target_cols))

    def apply_augmentation(self, volume_bag):
        """
        Applies the same affine transformation to all slices in the bag.
        volume_bag: (N, H, W, 3)
        """
        N, H, W, C = volume_bag.shape

        # Generate random parameters once
        angle = np.random.uniform(-Config.AUG_ROTATION, Config.AUG_ROTATION)
        scale = np.random.uniform(Config.AUG_SCALE[0], Config.AUG_SCALE[1])

        # Shift is relative to size
        tx = np.random.uniform(-Config.AUG_SHIFT, Config.AUG_SHIFT) * W
        ty = np.random.uniform(-Config.AUG_SHIFT, Config.AUG_SHIFT) * H

        # Get rotation matrix
        center = (W // 2, H // 2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        M[0, 2] += tx
        M[1, 2] += ty

        # Apply to each slice
        augmented_bag = np.zeros_like(volume_bag)

        for i in range(N):
            # Warp the 3-channel image directly
            augmented_bag[i] = cv2.warpAffine(
                volume_bag[i],
                M,
                (W, H),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )

        return augmented_bag


def process_scan(study_id, image_dir, load_cached_data=True):
    """
    Loads DICOMs, sorts, windows, resizes, and caches the 3D volume.
    Returns: numpy array (D, H, W) uint8
    """
    cache_path = os.path.join(Config.CACHE_DIR, f"{study_id}.npy")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            return np.load(cache_path)
        except Exception as e:
            print(f"Failed to load cache for {study_id}: {e}. Re-processing.")

    # 2. Process from scratch
    dicom_files = glob.glob(os.path.join(image_dir, "*"))
    if not dicom_files:
        # Fallback for empty directories or errors: return empty volume
        # Create a dummy volume to prevent crash
        print(f"Warning: No files found in {image_dir}")
        dummy = np.zeros(
            (Config.NUM_SLICES, Config.IMAGE_SIZE, Config.IMAGE_SIZE), dtype=np.uint8
        )
        return dummy

    # Read headers to sort
    slices = []
    for f in dicom_files:
        # We use the utility function which handles pydicom exceptions
        dcm = load_dicom(f)
        if dcm and hasattr(dcm, "ImagePositionPatient"):
            slices.append((dcm.ImagePositionPatient[2], f))  # Sort by Z
        elif dcm:
            # Fallback if no ImagePositionPatient (rare), use InstanceNumber
            if hasattr(dcm, "InstanceNumber"):
                slices.append((float(dcm.InstanceNumber), f))
            else:
                slices.append((0, f))

    # Sort by Z position
    slices.sort(key=lambda x: x[0])
    sorted_files = [x[1] for x in slices]

    # If no valid slices found
    if not sorted_files:
        dummy = np.zeros(
            (Config.NUM_SLICES, Config.IMAGE_SIZE, Config.IMAGE_SIZE), dtype=np.uint8
        )
        return dummy

    # Load pixel data and preprocess
    processed_slices = []
    for f in sorted_files:
        dcm = load_dicom(f)
        if dcm is None:
            continue

        try:
            # Convert to HU
            img = dcm.pixel_array.astype(np.float32)
            slope = getattr(dcm, "RescaleSlope", 1.0)
            intercept = getattr(dcm, "RescaleIntercept", 0.0)
            img = img * slope + intercept

            # Windowing (Bone Window)
            center = Config.WINDOW_LEVEL
            width = Config.WINDOW_WIDTH
            lower = center - width / 2
            upper = center + width / 2

            img = np.clip(img, lower, upper)
            img = (img - lower) / (upper - lower) * 255.0

            # Resize
            if img.shape[0] != Config.IMAGE_SIZE or img.shape[1] != Config.IMAGE_SIZE:
                img = cv2.resize(
                    img,
                    (Config.IMAGE_SIZE, Config.IMAGE_SIZE),
                    interpolation=cv2.INTER_LINEAR,
                )

            processed_slices.append(img.astype(np.uint8))

        except Exception as e:
            continue

    if not processed_slices:
        dummy = np.zeros(
            (Config.NUM_SLICES, Config.IMAGE_SIZE, Config.IMAGE_SIZE), dtype=np.uint8
        )
        return dummy

    volume = np.array(processed_slices, dtype=np.uint8)

    # 3. Save to cache
    try:
        np.save(cache_path, volume)
    except Exception as e:
        print(f"Failed to save cache for {study_id}: {e}")

    return volume


def get_dataloaders(train_batch_size=None, val_batch_size=None, debug=False):
    """
    Creates DataLoaders for training and validation.
    """
    if train_batch_size is None:
        train_batch_size = Config.BATCH_SIZE
    if val_batch_size is None:
        val_batch_size = Config.BATCH_SIZE * 2  # Validation can handle larger batches

    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)

    # Debug Mode: Subset data
    if debug or Config.DEBUG:
        train_df = train_df.head(Config.DEBUG_SAMPLE_SIZE)
        val_df = val_df.head(Config.DEBUG_SAMPLE_SIZE)
        print(f"DEBUG MODE: Using {len(train_df)} train and {len(val_df)} val samples.")

    # Datasets
    train_dataset = RSNADataset(train_df, subset="train", transform=True)
    val_dataset = RSNADataset(val_df, subset="val", transform=False)

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Drop incomplete batches to maintain stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def get_test_dataloader(batch_size=None):
    """
    Creates DataLoader for the test set.
    """
    if batch_size is None:
        batch_size = Config.BATCH_SIZE * 2

    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    test_dataset = RSNADataset(test_df, subset="test", transform=False)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return test_loader, test_df
