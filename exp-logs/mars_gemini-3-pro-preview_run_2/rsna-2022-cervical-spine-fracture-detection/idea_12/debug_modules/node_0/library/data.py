import os
import re
import glob
import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import get_logger

# Initialize logger
logger = get_logger(name="data_module")


def natural_key(string_):
    """
    Provides a key for natural sorting of filenames (e.g., 1.dcm, 2.dcm, 10.dcm).
    """
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", string_)]


def load_dicom_slice(path, image_size):
    """
    Reads a DICOM file, applies bone windowing, and resizes.

    Args:
        path (str): Path to the DICOM file.
        image_size (tuple): Target size (H, W).

    Returns:
        np.ndarray: Preprocessed image normalized to [0, 1].
    """
    try:
        if not os.path.exists(path):
            # Return black image if file missing (padding case)
            return np.zeros(image_size, dtype=np.float32)

        ds = pydicom.dcmread(path)
        img = ds.pixel_array.astype(np.float32)

        # Apply Rescale Slope/Intercept if present
        slope = getattr(ds, "RescaleSlope", 1.0)
        intercept = getattr(ds, "RescaleIntercept", 0.0)
        img = img * slope + intercept

        # Bone Windowing
        # Window Center (WL) = 400, Window Width (WW) = 1800
        center = 400
        width = 1800
        low = center - width / 2
        high = center + width / 2

        img = np.clip(img, low, high)
        img = (img - low) / (high - low)

    except Exception as e:
        # Fallback for corrupted files
        # logger.warning(f"Error reading DICOM {path}: {e}")
        return np.zeros(image_size, dtype=np.float32)

    # Resize
    if img.shape[:2] != image_size:
        img = cv2.resize(
            img, (image_size[1], image_size[0]), interpolation=cv2.INTER_LINEAR
        )

    return img


def get_study_paths(metadata_df, load_cached_data=True):
    """
    Retrieves sorted file paths for each study. Implements caching to Parquet.

    Args:
        metadata_df (pd.DataFrame): DataFrame containing StudyInstanceUID and image_path.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        dict: Mapping {StudyInstanceUID: [list of filenames]}
    """
    cache_file = os.path.join(Config.OUTPUT_DIR, "paths_cache.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_file):
        logger.info(f"Loading file paths from cache: {cache_file}")
        try:
            cache_df = pd.read_parquet(cache_file)
            # Convert to dictionary
            path_map = cache_df.set_index("StudyInstanceUID")["file_names"].to_dict()
            # Verify coverage
            if set(metadata_df["StudyInstanceUID"].unique()).issubset(
                set(path_map.keys())
            ):
                return path_map
            else:
                logger.info("Cache incomplete. Rebuilding...")
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}. Rebuilding...")

    # 2. Build from scratch
    logger.info("Scanning directories to build file path cache...")
    path_map = {}
    unique_studies = metadata_df[["StudyInstanceUID", "image_path"]].drop_duplicates()

    for _, row in unique_studies.iterrows():
        uid = row["StudyInstanceUID"]
        rel_dir = row["image_path"]
        full_dir = os.path.join(Config.INPUT_DIR, rel_dir)

        if os.path.exists(full_dir):
            files = [f for f in os.listdir(full_dir) if f.endswith(".dcm")]
            files.sort(key=natural_key)
            path_map[uid] = files
        else:
            path_map[uid] = []

    # 3. Save to cache
    try:
        # Convert dict to DF for parquet
        cache_data = [
            {"StudyInstanceUID": k, "file_names": v} for k, v in path_map.items()
        ]
        cache_df = pd.DataFrame(cache_data)
        cache_df.to_parquet(cache_file)
        logger.info(f"Saved file paths to cache: {cache_file}")
    except Exception as e:
        logger.warning(f"Failed to save cache: {e}")

    return path_map


class FractureDataset(Dataset):
    def __init__(self, metadata_df, path_map, phase="train", transform=None):
        """
        Args:
            metadata_df (pd.DataFrame): Metadata with UIDs and targets.
            path_map (dict): Dictionary mapping UIDs to sorted list of filenames.
            phase (str): 'train', 'val', or 'test'.
            transform (A.Compose): Albumentations transform pipeline.
        """
        self.metadata = metadata_df
        self.path_map = path_map
        self.phase = phase
        self.transform = transform

        # Define targets
        self.target_cols = ["patient_overall", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]

        # Debugging subset
        if Config.DEBUG:
            logger.info(
                f"DEBUG MODE: Subsetting {phase} dataset to {Config.DEBUG_SAMPLE_SIZE} samples."
            )
            self.metadata = self.metadata.iloc[: Config.DEBUG_SAMPLE_SIZE].reset_index(
                drop=True
            )

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        uid = row["StudyInstanceUID"]
        image_dir = os.path.join(Config.INPUT_DIR, row["image_path"])

        # Get sorted filenames
        files = self.path_map.get(uid, [])
        num_files = len(files)

        # Sampling Logic: Select 96 equidistant indices
        if num_files == 0:
            # Handle empty directory edge case
            indices = np.zeros(Config.SEQ_LEN, dtype=int)
        else:
            indices = np.linspace(0, num_files - 1, Config.SEQ_LEN).round().astype(int)

        # Prepare container for the sequence
        # Shape: (SEQ_LEN, 3, H, W)
        sequence_tensor = np.zeros(
            (Config.SEQ_LEN, Config.IMAGE_SIZE[0], Config.IMAGE_SIZE[1], 3),
            dtype=np.float32,
        )

        # Volumetric Augmentation Setup
        # We need to apply the SAME geometric transform to all slices in the stack
        replay_data = None

        if self.phase == "train" and self.transform:
            # We use ReplayCompose to generate params once
            # We will apply it to the first slice to get the params
            pass

        # Iterate through sampled indices
        for i, center_idx in enumerate(indices):
            # 2.5D Stacking: z-1, z, z+1
            stack_indices = [center_idx - 1, center_idx, center_idx + 1]

            stack_img = np.zeros(
                (Config.IMAGE_SIZE[0], Config.IMAGE_SIZE[1], 3), dtype=np.float32
            )

            for channel, slice_idx in enumerate(stack_indices):
                # Boundary checks
                slice_idx = max(0, min(slice_idx, num_files - 1))

                if num_files > 0:
                    file_path = os.path.join(image_dir, files[slice_idx])
                    img = load_dicom_slice(file_path, Config.IMAGE_SIZE)
                else:
                    img = np.zeros(Config.IMAGE_SIZE, dtype=np.float32)

                stack_img[..., channel] = img

            # Apply Augmentation
            if self.transform:
                if self.phase == "train":
                    if replay_data is None:
                        # First item: apply transform and save params
                        res = self.transform(image=stack_img)
                        stack_img = res["image"]
                        replay_data = res["replay"]
                    else:
                        # Subsequent items: replay transform
                        res = self.transform.replay(replay_data, image=stack_img)
                        stack_img = res["image"]
                else:
                    # Validation/Test: Just resize/normalize (no random geom)
                    res = self.transform(image=stack_img)
                    stack_img = res["image"]

            sequence_tensor[i] = stack_img

        # Convert to PyTorch Tensor format: (Seq, H, W, C) -> (Seq, C, H, W)
        # Note: Albumentations ToTensorV2 converts to (C, H, W) but we might have returned numpy
        # If transform includes ToTensorV2, it's a tensor.
        # However, for ReplayCompose with sequence, it's safer to handle tensor conversion manually
        # to ensure consistency if ToTensorV2 wasn't the last step or if we want to stack first.

        # Let's assume transform returns numpy (no ToTensorV2 in Compose for simplicity here,
        # we do manual conversion to ensure shape (Seq, 3, H, W))

        sequence_tensor = np.transpose(
            sequence_tensor, (0, 3, 1, 2)
        )  # (96, 3, 384, 384)
        sequence_tensor = torch.from_numpy(sequence_tensor)

        # Get Targets
        if self.phase != "test":
            labels = row[self.target_cols].values.astype(np.float32)
            labels = torch.tensor(labels)
        else:
            labels = torch.zeros(8, dtype=np.float32)  # Dummy for test

        return sequence_tensor, labels, uid


def get_dataloaders(load_cached_data=True):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        load_cached_data (bool): Whether to use cached file paths.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # 1. Load Metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA)
    val_df = pd.read_csv(Config.VAL_METADATA)
    test_df = pd.read_csv(Config.TEST_METADATA)

    # 2. Build/Load Path Cache (Combine all UIDs to scan once)
    all_meta = pd.concat([train_df, val_df, test_df], ignore_index=True)
    path_map = get_study_paths(all_meta, load_cached_data=load_cached_data)

    # 3. Define Transforms
    # Note: We use ReplayCompose for Train to sync augmentation across the sequence
    train_transform = A.ReplayCompose(
        [
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=15,
                p=0.5,
                border_mode=cv2.BORDER_CONSTANT,
            ),
            A.Resize(height=Config.IMAGE_SIZE[0], width=Config.IMAGE_SIZE[1]),
            # No Normalize here, we did 0-1 manually.
            # No ToTensorV2 here, we do it in __getitem__ to handle the sequence stack.
        ]
    )

    val_transform = A.Compose(
        [
            A.Resize(height=Config.IMAGE_SIZE[0], width=Config.IMAGE_SIZE[1]),
        ]
    )

    # 4. Create Datasets
    train_dataset = FractureDataset(
        train_df, path_map, phase="train", transform=train_transform
    )
    val_dataset = FractureDataset(
        val_df, path_map, phase="val", transform=val_transform
    )
    test_dataset = FractureDataset(
        test_df, path_map, phase="test", transform=val_transform
    )

    # 5. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    logger.info(
        f"DataLoaders created. Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )

    return train_loader, val_loader, test_loader
