import os
import glob
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

from library.config import Config
from library.utils import read_dicom_image, normalize_min_max, get_centroids

# ==========================================
# Helper Functions
# ==========================================


def _extract_instance_number(filename):
    """
    Extracts the integer instance number from the filename.
    Re-implemented here to support efficient file sorting without loading volumes.
    """
    match = re.search(r"Image-(\d+)\.dcm", filename)
    if match:
        return int(match.group(1))
    numbers = re.findall(r"\d+", filename)
    if numbers:
        return int(numbers[-1])
    return 0


def get_sorted_file_list(folder_path):
    """
    Returns a sorted list of DICOM file paths in a directory.
    """
    if not os.path.exists(folder_path):
        return []
    files = glob.glob(os.path.join(folder_path, "*.dcm"))
    # Sort by instance number to ensure spatial order
    files.sort(key=lambda x: _extract_instance_number(os.path.basename(x)))
    return files


# ==========================================
# Augmentation Pipeline
# ==========================================


def get_transforms(split):
    """
    Returns the Albumentations transform pipeline for the given split.
    Strictly excludes Translation and Scaling as per VCAE strategy.
    """
    if split == "train":
        return A.Compose(
            [
                A.Resize(height=Config.IMG_SIZE[0], width=Config.IMG_SIZE[1]),
                # Spatially-Coupled Augmentations
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                # Rotation only, no shifting or scaling
                A.Rotate(limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0),
                # Elastic & Grid distortions
                A.ElasticTransform(
                    alpha=1,
                    sigma=50,
                    alpha_affine=50,
                    p=0.5,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                ),
                A.GridDistortion(p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0),
                ToTensorV2(),
            ]
        )
    else:
        # Validation / Test: Resize and convert to tensor
        return A.Compose(
            [
                A.Resize(height=Config.IMG_SIZE[0], width=Config.IMG_SIZE[1]),
                ToTensorV2(),
            ]
        )


# ==========================================
# Dataset Class
# ==========================================


class VCAEDataset(Dataset):
    def __init__(
        self,
        metadata_df,
        expert_offset,
        split="train",
        transform=None,
        cache_file_lists=True,
    ):
        """
        Args:
            metadata_df (pd.DataFrame): DataFrame containing paths and CoM info.
            expert_offset (int): The slice offset relative to CoM (e.g., -5, 0, 5).
            split (str): 'train', 'val', or 'test'.
            transform (A.Compose): Albumentations transforms.
            cache_file_lists (bool): Whether to cache sorted file paths in memory.
        """
        self.df = metadata_df.reset_index(drop=True)
        self.expert_offset = expert_offset
        self.split = split
        self.transform = transform
        self.modalities = Config.MODALITIES  # ["FLAIR", "T1wCE", "T2w"]

        # Cache for file lists to avoid globbing every iteration
        self.file_list_cache = {}
        if cache_file_lists:
            self._preload_file_lists()

    def _preload_file_lists(self):
        """Pre-fetches sorted file lists for all subjects to speed up training."""
        # Only print if not in a worker process to avoid spam
        if torch.utils.data.get_worker_info() is None:
            print(f"Pre-loading file lists for {len(self.df)} subjects...")

        for idx, row in self.df.iterrows():
            sid = row["BraTS21ID"]
            self.file_list_cache[sid] = {}
            for mod in self.modalities:
                rel_path = row[f"{mod.lower()}_path"]
                full_path = os.path.join(Config.INPUT_DIR, rel_path)
                self.file_list_cache[sid][mod] = get_sorted_file_list(full_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sid = row["BraTS21ID"]

        channels = []

        # Determine jitter for this iteration (only for training)
        jitter = 0
        if self.split == "train" and Config.SLICE_JITTER > 0:
            jitter = np.random.randint(-Config.SLICE_JITTER, Config.SLICE_JITTER + 1)

        for mod in self.modalities:
            # 1. Get CoM for this modality
            com_col = f"{mod}_CoM"
            if com_col not in row:
                # Fallback should not happen if get_centroids was called, but safety first
                com = 0
            else:
                com = int(row[com_col])

            # 2. Calculate target slice index
            target_index = com + self.expert_offset + jitter

            # 3. Get sorted file list
            if sid in self.file_list_cache and mod in self.file_list_cache[sid]:
                files = self.file_list_cache[sid][mod]
            else:
                # Fallback if not cached
                rel_path = row[f"{mod.lower()}_path"]
                full_path = os.path.join(Config.INPUT_DIR, rel_path)
                files = get_sorted_file_list(full_path)

            num_files = len(files)

            if num_files == 0:
                # Missing data case: return black slice
                img = np.zeros(Config.IMG_SIZE, dtype=np.float32)
            else:
                # Clamp index to valid range
                idx_to_load = max(0, min(target_index, num_files - 1))
                file_path = files[idx_to_load]

                # Load and Normalize
                img = read_dicom_image(file_path)
                if img is None:
                    img = np.zeros(Config.IMG_SIZE, dtype=np.float32)
                else:
                    img = normalize_min_max(img)
                    # Explicitly resize to ensure all channels have the same shape for np.stack
                    if img.shape[:2] != Config.IMG_SIZE:
                        img = cv2.resize(img, (Config.IMG_SIZE[1], Config.IMG_SIZE[0]))

            # Resize is handled by albumentations, but we need consistent 2D arrays here
            # If image is not 2D (rare), handle it
            if img.ndim != 2:
                if img.ndim == 3:
                    img = img[:, :, 0]  # Take first channel if accidentally 3D
                else:
                    img = np.zeros(Config.IMG_SIZE, dtype=np.float32)

            channels.append(img)

        # Stack channels: (H, W, 3)
        image = np.stack(channels, axis=-1)

        # Apply Augmentations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]  # Returns Tensor (C, H, W)

        # Return Logic
        if self.split == "test":
            return image, str(sid)
        else:
            label = row["MGMT_value"]
            return image, torch.tensor(label, dtype=torch.float32)


# ==========================================
# Data Factory
# ==========================================


def get_dataloaders(fold_idx=0, expert_offset=0, load_cached_data=True):
    """
    Creates DataLoaders for a specific Fold.

    Args:
        fold_idx (int): Fold index for cross-validation (0-4).
        expert_offset (int): The slice offset relative to CoM.
        load_cached_data (bool): Whether to use cached CoM data.

    Returns:
        train_loader, val_loader
    """
    # 1. Setup & Metadata Loading

    # Load raw metadata
    df_train_full = pd.read_csv(Config.TRAIN_METADATA_PATH)
    df_val_full = pd.read_csv(Config.VAL_METADATA_PATH)

    # 2. Compute/Load Centroids (Deterministic Processing Cache)
    # This modifies the DFs to include {Modality}_CoM columns
    df_train_full = get_centroids(
        df_train_full, split_name="train", load_cached_data=load_cached_data
    )
    df_val_full = get_centroids(
        df_val_full, split_name="val", load_cached_data=load_cached_data
    )

    # 3. Combine for CV splitting (if we were doing dynamic CV, but we have fixed val set)
    # The prompt implies using the provided train/val split or doing 5-fold.
    # Config says NUM_FOLDS = 5. We should respect the fold_idx.
    # We will combine train and val, then split based on BraTS21ID % NUM_FOLDS for reproducibility.

    df_all = pd.concat([df_train_full, df_val_full], ignore_index=True)

    # Debugging: Reduce dataset size
    if Config.DEBUG:
        df_all = df_all.head(Config.DEBUG_SAMPLE_SIZE)
        print(f"DEBUG MODE: Reduced dataset to {len(df_all)} samples.")

    # Create Folds
    # Simple modulo split based on ID to ensure subject isolation
    all_ids = df_all["BraTS21ID"].unique()
    val_ids = [sid for sid in all_ids if sid % Config.NUM_FOLDS == fold_idx]

    df_train = df_all[~df_all["BraTS21ID"].isin(val_ids)].reset_index(drop=True)
    df_val = df_all[df_all["BraTS21ID"].isin(val_ids)].reset_index(drop=True)

    print(f"Fold {fold_idx} | Train: {len(df_train)} | Val: {len(df_val)}")

    # 4. Create Datasets
    train_dataset = VCAEDataset(
        df_train,
        expert_offset=expert_offset,
        split="train",
        transform=get_transforms("train"),
    )

    val_dataset = VCAEDataset(
        df_val,
        expert_offset=expert_offset,
        split="val",
        transform=get_transforms("val"),
    )

    # 5. Create Loaders
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

    return train_loader, val_loader


def get_test_dataloader(expert_offset=0, load_cached_data=True):
    """
    Creates DataLoader for the Test set.
    """

    df_test = pd.read_csv(Config.TEST_METADATA_PATH)

    # Compute/Load Centroids for Test
    df_test = get_centroids(
        df_test, split_name="test", load_cached_data=load_cached_data
    )

    test_dataset = VCAEDataset(
        df_test,
        expert_offset=expert_offset,
        split="test",
        transform=get_transforms("test"),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return test_loader
