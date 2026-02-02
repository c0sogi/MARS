import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import get_logger

# Initialize logger
logger = get_logger(name="DataLoader")


def load_dicom_image(path):
    """
    Robustly loads a DICOM image.
    Attempts OpenCV first. Falls back to raw binary tail-read.
    Returns float32 numpy array.
    """
    if not os.path.exists(path):
        # Return a blank image if file is missing to prevent crash,
        # though data integrity checks should catch this.
        return np.zeros((Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)

    # 1. Try OpenCV
    try:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            return img.astype(np.float32)
    except Exception:
        pass

    # 2. Fallback: Raw Binary Tail-Read
    # Assumes uncompressed uint16 data at the end of the file.
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)  # Seek to end
            file_size = f.tell()

            # Heuristic for resolution based on file size
            # 512x512x2 bytes = 524,288
            # 256x256x2 bytes = 131,072
            if file_size >= 524288:
                dim = 512
            else:
                dim = 256

            num_bytes = dim * dim * 2
            f.seek(-num_bytes, 2)
            data = f.read(num_bytes)
            img = np.frombuffer(data, dtype=np.uint16).reshape((dim, dim))
            return img.astype(np.float32)
    except Exception as e:
        # If all fails, return zeros
        return np.zeros((Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)


def resize_image(img):
    """
    Resizes image to Config.IMG_SIZE using Area Interpolation.
    """
    if img.shape[0] != Config.IMG_SIZE or img.shape[1] != Config.IMG_SIZE:
        img = cv2.resize(
            img, (Config.IMG_SIZE, Config.IMG_SIZE), interpolation=cv2.INTER_AREA
        )
    return img


def get_sorted_image_files(directory):
    """
    Returns a sorted list of image files in a directory.
    Assumes filenames are like Image-1.dcm, Image-10.dcm, etc.
    """
    if not os.path.exists(directory):
        return []
    files = [f for f in os.listdir(directory) if "Image-" in f]
    # Sort by the integer number in the filename
    files.sort(key=lambda x: int(x.split("-")[1].split(".")[0]))
    return files


def get_anchor_slices(metadata_df, load_cached_data=True):
    """
    Determines the anchor slice index for each subject based on FLAIR intensity.
    Uses caching to store results.
    """
    cache_file = os.path.join(Config.WORKING_DIR, "roi_cache.parquet")

    # 1. Try to load cache
    if load_cached_data and os.path.exists(cache_file):
        try:
            cache_df = pd.read_parquet(cache_file)
            # Convert to dictionary {BraTS21ID: anchor_index}
            anchor_dict = dict(zip(cache_df["BraTS21ID"], cache_df["anchor_index"]))
            logger.info(f"Loaded ROI anchors from cache: {len(anchor_dict)} entries.")
            return anchor_dict
        except Exception as e:
            logger.warning(f"Failed to load ROI cache: {e}. Recomputing...")

    # 2. Compute anchors
    logger.info("Computing ROI anchors (this may take a while)...")
    anchor_dict = {}

    # Filter for unique subjects to avoid redundant computation
    unique_subjects = metadata_df[["BraTS21ID", "path_FLAIR"]].drop_duplicates()

    count = 0
    for _, row in unique_subjects.iterrows():
        subject_id = row["BraTS21ID"]
        flair_path = os.path.join(Config.INPUT_DIR, row["path_FLAIR"])

        files = get_sorted_image_files(flair_path)
        num_files = len(files)

        if num_files == 0:
            anchor_dict[subject_id] = 0
            continue

        # Define depth bounds
        start_idx = int(num_files * Config.ROI_DEPTH_MIN)
        end_idx = int(num_files * Config.ROI_DEPTH_MAX)

        # Ensure valid range
        if start_idx >= end_idx:
            start_idx = 0
            end_idx = num_files

        max_intensity = -1.0
        best_idx = num_files // 2  # Default to middle

        # Iterate through valid range
        for i in range(start_idx, end_idx):
            f_path = os.path.join(flair_path, files[i])
            img = load_dicom_image(f_path)
            current_intensity = np.sum(img)

            if current_intensity > max_intensity:
                max_intensity = current_intensity
                best_idx = i

        anchor_dict[subject_id] = best_idx
        count += 1
        if count % 50 == 0:
            logger.info(f"Processed {count} subjects...")

    # 3. Save cache
    try:
        cache_df = pd.DataFrame(
            list(anchor_dict.items()), columns=["BraTS21ID", "anchor_index"]
        )
        cache_df.to_parquet(cache_file, index=False)
        logger.info("ROI anchors saved to cache.")
    except Exception as e:
        logger.warning(f"Failed to save ROI cache: {e}")

    return anchor_dict


class MRIDataset(Dataset):
    def __init__(self, df, anchor_dict, transform=None, mode="train"):
        self.df = df
        self.anchor_dict = anchor_dict
        self.transform = transform
        self.mode = mode

        # Define modalities in order
        self.modalities = ["FLAIR", "T1w", "T1wCE", "T2w"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        subject_id = row["BraTS21ID"]

        # Get anchor slice index
        anchor_idx = self.anchor_dict.get(subject_id, 0)

        # Prepare list to hold all 24 channels
        # Order:
        # Grp0: FLAIR Local, Grp1: FLAIR Context
        # Grp2: T1w Local,   Grp3: T1w Context
        # Grp4: T1wCE Local, Grp5: T1wCE Context
        # Grp6: T2w Local,   Grp7: T2w Context

        all_channels = []

        for mod in self.modalities:
            mod_path_rel = row[f"path_{mod}"]
            mod_dir = os.path.join(Config.INPUT_DIR, mod_path_rel)
            files = get_sorted_image_files(mod_dir)
            num_files = len(files)

            if num_files == 0:
                # Handle missing modality: generate zeros
                # 2 scales * 3 slices = 6 channels per modality
                for _ in range(6):
                    all_channels.append(
                        np.zeros((Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)
                    )
                continue

            # Define slice indices for Local (Stride 2) and Context (Stride 5)
            # Local: [A-2, A, A+2]
            # Context: [A-5, A, A+5]

            # We must handle out-of-bounds by clamping
            indices_local = [anchor_idx - 2, anchor_idx, anchor_idx + 2]
            indices_context = [anchor_idx - 5, anchor_idx, anchor_idx + 5]

            # Helper to load and process a specific index
            def get_slice(i):
                # Clamp index
                i_clamped = max(0, min(i, num_files - 1))
                f_path = os.path.join(mod_dir, files[i_clamped])
                img = load_dicom_image(f_path)
                img = resize_image(img)
                return img

            # Load Local Slices
            for i in indices_local:
                all_channels.append(get_slice(i))

            # Load Context Slices
            for i in indices_context:
                all_channels.append(get_slice(i))

        # Stack into (H, W, C)
        # Total channels = 4 mods * 2 scales * 3 slices = 24
        img_stack = np.stack(all_channels, axis=-1)  # (224, 224, 24)

        # Normalize: Independent Per-Channel Min-Max
        # Avoid division by zero
        min_vals = img_stack.min(axis=(0, 1), keepdims=True)
        max_vals = img_stack.max(axis=(0, 1), keepdims=True)
        img_stack = (img_stack - min_vals) / (max_vals - min_vals + 1e-8)

        # Augmentation
        if self.transform:
            augmented = self.transform(image=img_stack)
            img_stack = augmented[
                "image"
            ]  # Albumentations returns (H, W, C) if not ToTensorV2

        # Convert to Tensor (C, H, W)
        # If transform included ToTensorV2, it's already tensor.
        # But we handle manual transform logic usually with albumentations for multi-channel.
        # Standard Albumentations ToTensorV2 transposes to (C, H, W).

        if not torch.is_tensor(img_stack):
            img_stack = torch.from_numpy(img_stack).permute(2, 0, 1).float()

        # Get Label (if available)
        if "MGMT_value" in row:
            label = torch.tensor(row["MGMT_value"], dtype=torch.float32)
            return img_stack, label
        else:
            return img_stack, torch.tensor(-1.0)  # Dummy label for test


def get_transforms(mode="train"):
    """
    Returns Albumentations transforms.
    """
    if mode == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT, p=0.5),
                # Note: We do not use Normalize here because we did custom per-channel min-max
                # We do not use ToTensorV2 here to keep it numpy for manual permute or ensure flow
            ]
        )
    else:
        return None


def get_dataloaders(train_df=None, val_df=None, test_df=None):
    """
    Factory function to create dataloaders.
    """
    # 1. Load/Compute Anchors
    # We need anchors for all subjects involved.
    # Concatenate DFs to get full list for anchor computation
    dfs = []
    if train_df is not None:
        dfs.append(train_df)
    if val_df is not None:
        dfs.append(val_df)
    if test_df is not None:
        dfs.append(test_df)

    full_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    anchor_dict = {}
    if not full_df.empty:
        anchor_dict = get_anchor_slices(full_df, load_cached_data=True)

    loaders = {}

    if train_df is not None:
        train_ds = MRIDataset(
            train_df, anchor_dict, transform=get_transforms(mode="train"), mode="train"
        )
        loaders["train"] = DataLoader(
            train_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

    if val_df is not None:
        val_ds = MRIDataset(
            val_df, anchor_dict, transform=get_transforms(mode="val"), mode="val"
        )
        loaders["val"] = DataLoader(
            val_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

    if test_df is not None:
        test_ds = MRIDataset(
            test_df, anchor_dict, transform=get_transforms(mode="test"), mode="test"
        )
        loaders["test"] = DataLoader(
            test_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

    return loaders
