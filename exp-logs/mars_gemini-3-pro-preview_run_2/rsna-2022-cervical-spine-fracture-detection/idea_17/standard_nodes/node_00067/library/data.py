import os
import glob
import re
import random
import numpy as np
import pandas as pd
import torch
import pydicom
import cv2
import albumentations as A
from albumentations.core.composition import ReplayCompose
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import seed_everything


def load_dicom_array(path, size=None):
    """
    Reads a DICOM file and returns a normalized numpy array (uint8).
    Handles Hounsfield Unit conversion and Bone Windowing if metadata is available.
    """
    try:
        dicom = pydicom.dcmread(path)
        data = dicom.pixel_array

        # Convert to Hounsfield Units if Rescale Slope/Intercept exist
        if hasattr(dicom, "RescaleSlope") and hasattr(dicom, "RescaleIntercept"):
            slope = float(dicom.RescaleSlope)
            intercept = float(dicom.RescaleIntercept)
            data = data * slope + intercept

        # Apply Bone Windowing (Level: 300, Width: 2000) -> [-700, 1300]
        # This highlights bone structures
        window_center = 300
        window_width = 2000
        min_value = window_center - (window_width / 2)
        max_value = window_center + (window_width / 2)

        data = np.clip(data, min_value, max_value)

        # Normalize to 0-255
        if data.max() > data.min():
            data = (data - data.min()) / (data.max() - data.min())
        else:
            data = np.zeros_like(data)

        data = (data * 255).astype(np.uint8)

        # Resize if requested (though usually handled by transforms)
        if size:
            data = cv2.resize(data, (size[1], size[0]))

        return data

    except Exception as e:
        # Return black image on failure
        if size:
            return np.zeros((size[0], size[1]), dtype=np.uint8)
        return np.zeros((512, 512), dtype=np.uint8)


def get_study_paths_map(metadata_df, image_dir, cache_name, load_cached_data=True):
    """
    Scans the image directory to find all DICOM files for each study.
    Caches the result to a parquet file to speed up subsequent runs.
    """
    cache_path = os.path.join(Config.WORKING_DIR, f"{cache_name}.parquet")

    # 1. Try to load cached data
    if load_cached_data and os.path.exists(cache_path):
        try:
            # Load parquet
            cached_df = pd.read_parquet(cache_path)
            # Convert back to dictionary: StudyUID -> List of filenames
            # We assume the dataframe has columns [StudyInstanceUID, filename] and is sorted
            paths_map = (
                cached_df.groupby("StudyInstanceUID")["filename"].apply(list).to_dict()
            )
            return paths_map
        except Exception as e:
            print(f"Failed to load cache {cache_path}: {e}. Recomputing...")

    # 2. Compute from scratch
    study_uids = metadata_df["StudyInstanceUID"].unique()
    paths_list = []

    for uid in study_uids:
        study_dir = os.path.join(image_dir, uid)
        if not os.path.exists(study_dir):
            continue

        # Get all dcm files
        files = os.listdir(study_dir)
        dcm_files = [f for f in files if f.endswith(".dcm")]

        # Sort by slice number (integer value of filename)
        # Filenames are like '10.dcm', '1.dcm'
        def extract_number(f):
            match = re.search(r"(\d+)", f)
            return int(match.group(1)) if match else 0

        dcm_files.sort(key=extract_number)

        for f in dcm_files:
            paths_list.append({"StudyInstanceUID": uid, "filename": f})

    # Create DataFrame
    paths_df = pd.DataFrame(paths_list)

    # Save to cache
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    paths_df.to_parquet(cache_path, index=False)

    # Convert to dictionary
    paths_map = paths_df.groupby("StudyInstanceUID")["filename"].apply(list).to_dict()

    return paths_map


class RSNADataset(Dataset):
    def __init__(self, df, image_dir, paths_map, transform=None, seq_len=96):
        self.df = df
        self.image_dir = image_dir
        self.paths_map = paths_map
        self.transform = transform
        self.seq_len = seq_len

        # Filter df to only include studies we found images for
        valid_uids = set(paths_map.keys())
        self.df = self.df[self.df["StudyInstanceUID"].isin(valid_uids)].reset_index(
            drop=True
        )

        self.uids = self.df["StudyInstanceUID"].values

        # Pre-fetch labels if available
        self.labels = None
        if "patient_overall" in self.df.columns:
            target_cols = Config.TARGET_COLS
            self.labels = self.df[target_cols].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        uid = self.uids[idx]
        files = self.paths_map[uid]
        num_files = len(files)

        # Uniform Sampling
        if num_files >= self.seq_len:
            # Downsample
            indices = np.linspace(0, num_files - 1, self.seq_len).astype(int)
        else:
            # Upsample (repeat indices)
            indices = np.linspace(0, num_files - 1, self.seq_len).astype(int)

        # Prepare storage for the sequence
        # Shape: (SEQ_LEN, 3, H, W)
        # We will stack later. First collect list of (H, W, 3) arrays
        image_sequence = []

        # Dataset-Encapsulated Augmentation
        # We need to apply the same random geometric transform to all slices in the stack
        replay_params = None

        for i, slice_idx in enumerate(indices):
            # Identify neighbors (z-1, z, z+1)
            # Handle boundary conditions by clamping
            prev_idx = max(0, slice_idx - 1)
            curr_idx = slice_idx
            next_idx = min(num_files - 1, slice_idx + 1)

            # Load paths
            path_prev = os.path.join(self.image_dir, uid, files[prev_idx])
            path_curr = os.path.join(self.image_dir, uid, files[curr_idx])
            path_next = os.path.join(self.image_dir, uid, files[next_idx])

            # Load images
            img_prev = load_dicom_array(path_prev)
            img_curr = load_dicom_array(path_curr)
            img_next = load_dicom_array(path_next)

            # Stack to create 2.5D representation (H, W, 3)
            img_stack = np.stack([img_prev, img_curr, img_next], axis=-1)

            # Apply Transforms
            if self.transform:
                if replay_params is None:
                    # First slice: Apply transform and record parameters
                    augmented = self.transform(image=img_stack)
                    img_stack = augmented["image"]
                    # If using ReplayCompose, we get 'replay' data
                    if isinstance(self.transform, ReplayCompose):
                        replay_params = augmented["replay"]
                else:
                    # Subsequent slices: Replay parameters
                    if isinstance(self.transform, ReplayCompose):
                        augmented = ReplayCompose.replay(replay_params, image=img_stack)
                        img_stack = augmented["image"]
                    else:
                        # Fallback if not ReplayCompose (shouldn't happen with correct setup)
                        augmented = self.transform(image=img_stack)
                        img_stack = augmented["image"]

            # Normalize to 0-1 and convert to tensor
            # Albumentations usually returns numpy array.
            # If ToTensorV2 is used, it returns Tensor (C, H, W).
            # If not, we do it manually.
            if isinstance(img_stack, np.ndarray):
                img_stack = img_stack.astype(np.float32) / 255.0
                img_stack = np.transpose(img_stack, (2, 0, 1))  # (H, W, C) -> (C, H, W)
                img_stack = torch.from_numpy(img_stack)

            image_sequence.append(img_stack)

        # Stack sequence -> (SEQ_LEN, C, H, W)
        video_tensor = torch.stack(image_sequence, dim=0)

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return video_tensor, label
        else:
            # Return dummy label or just the image for inference
            return video_tensor, torch.zeros(Config.NUM_CLASSES)


def get_transforms(phase="train"):
    """
    Returns Albumentations transforms.
    Uses ReplayCompose to ensure consistent application across the sequence.
    """
    height, width = Config.IMAGE_SIZE

    if phase == "train":
        return ReplayCompose(
            [
                A.Resize(height=height, width=width),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.2
                ),
                A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.2),
                # Note: Normalization is handled manually in __getitem__ to ensure consistency with 3D stacking logic
            ]
        )
    else:
        return ReplayCompose(
            [
                A.Resize(height=height, width=width),
            ]
        )


def create_dataloaders(
    train_df=None, val_df=None, test_df=None, load_cached_data=True, debug=False
):
    """
    Creates DataLoaders for train, val, and test sets.

    Args:
        train_df (pd.DataFrame): Training metadata.
        val_df (pd.DataFrame): Validation metadata.
        test_df (pd.DataFrame): Test metadata.
        load_cached_data (bool): Whether to use cached file paths.
        debug (bool): If True, subsets data for debugging.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """

    # 1. Prepare Path Maps
    # We combine all UIDs to scan directories efficiently or scan per split
    # Scanning everything ensures we cover all cases

    train_loader = None
    val_loader = None
    test_loader = None

    # --- Train Loader ---
    if train_df is not None:
        if debug:
            train_df = train_df.iloc[:20]

        print(f"Scanning training files for {len(train_df)} studies...")
        train_map = get_study_paths_map(
            train_df, Config.TRAIN_IMAGES_DIR, "train_paths_cache", load_cached_data
        )

        train_dataset = RSNADataset(
            train_df,
            Config.TRAIN_IMAGES_DIR,
            train_map,
            transform=get_transforms("train"),
            seq_len=Config.SEQ_LEN,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )

    # --- Val Loader ---
    if val_df is not None:
        if debug:
            val_df = val_df.iloc[:10]

        print(f"Scanning validation files for {len(val_df)} studies...")
        val_map = get_study_paths_map(
            val_df, Config.TRAIN_IMAGES_DIR, "val_paths_cache", load_cached_data
        )

        val_dataset = RSNADataset(
            val_df,
            Config.TRAIN_IMAGES_DIR,
            val_map,
            transform=get_transforms("val"),
            seq_len=Config.SEQ_LEN,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=False,
        )

    # --- Test Loader ---
    if test_df is not None:
        if debug:
            test_df = test_df.iloc[:10]

        print(f"Scanning test files for {len(test_df)} studies...")
        test_map = get_study_paths_map(
            test_df, Config.TEST_IMAGES_DIR, "test_paths_cache", load_cached_data
        )

        test_dataset = RSNADataset(
            test_df,
            Config.TEST_IMAGES_DIR,
            test_map,
            transform=get_transforms("test"),
            seq_len=Config.SEQ_LEN,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=False,
        )

    return train_loader, val_loader, test_loader
