import os
import cv2
import numpy as np
import pandas as pd
import torch
import concurrent.futures
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from library.config import Config
from library.utils import rle_decode, get_metadata


def compute_p99(path):
    """Helper to compute 99th percentile of an image."""
    full_path = os.path.join(Config.INPUT_DIR, path)
    try:
        img = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return 0.0
        return np.percentile(img, 99)
    except Exception:
        return 0.0


def add_volume_stats(df):
    """
    Computes volume-level normalization statistics (max/p99) and adds to dataframe.
    Cite solution_lesson_node_00002: Computing normalization statistics across the entire 3D volume.
    """
    if df.empty:
        return df

    print("Computing volume statistics for normalization...")
    unique_paths = df["image_path"].unique()

    # Compute p99 for all images in parallel
    path_to_p99 = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(executor.map(compute_p99, unique_paths))

    path_to_p99 = dict(zip(unique_paths, results))

    # Map p99 to dataframe
    df["slice_p99"] = df["image_path"].map(path_to_p99)

    # Compute Volume Max (Max of slice p99s in the volume)
    # This is robust to single-pixel outliers but captures the dynamic range of the volume
    vol_stats = df.groupby(["case", "day"])["slice_p99"].max().reset_index()
    vol_stats = vol_stats.rename(columns={"slice_p99": "vol_max"})

    # Merge back
    df = df.merge(vol_stats, on=["case", "day"], how="left")

    # Fill NaNs or zeros with a safe global default (e.g. 1000) to avoid division by zero
    df["vol_max"] = df["vol_max"].replace(0, 1000.0).fillna(1000.0)

    return df


def process_and_cache_25d_metadata(load_cached_data=True):
    """
    Loads metadata and processes it to include paths for previous and next slices
    to facilitate 2.5D input. Caches the processed dataframes.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (train_df, val_df, test_df) with added 'prev_image_path' and 'next_image_path' columns.
    """
    cache_dir = Config.CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # Updated cache names to force regeneration with volume stats
    train_cache = os.path.join(cache_dir, "train_25d_v2.parquet")
    val_cache = os.path.join(cache_dir, "val_25d_v2.parquet")
    test_cache = os.path.join(cache_dir, "test_25d_v2.parquet")

    # Try loading from cache
    if load_cached_data:
        if (
            os.path.exists(train_cache)
            and os.path.exists(val_cache)
            and os.path.exists(test_cache)
        ):
            try:
                train_df = pd.read_parquet(train_cache)
                val_df = pd.read_parquet(val_cache)
                test_df = pd.read_parquet(test_cache)
                return train_df, val_df, test_df
            except Exception:
                pass  # Fallback to processing

    # Load base metadata
    train_df, val_df, test_df = get_metadata(load_cached_data=True)

    def add_neighbors(df):
        if df.empty:
            return df

        # Ensure slice is integer for correct sorting
        df["slice_int"] = df["slice"].astype(int)

        # Sort by case, day, slice
        df = df.sort_values(["case", "day", "slice_int"]).reset_index(drop=True)

        # Group by case and day
        grouped = df.groupby(["case", "day"])

        # Shift to get prev and next paths
        df["prev_image_path"] = grouped["image_path"].shift(1)
        df["next_image_path"] = grouped["image_path"].shift(-1)

        # Fill NaNs (boundaries) with current image path
        df["prev_image_path"] = df["prev_image_path"].fillna(df["image_path"])
        df["next_image_path"] = df["next_image_path"].fillna(df["image_path"])

        # Drop temp column
        df = df.drop(columns=["slice_int"])

        return df

    # Process each split
    train_df = add_neighbors(train_df)
    val_df = add_neighbors(val_df)
    test_df = add_neighbors(test_df)

    # Add Volume Normalization Stats
    train_df = add_volume_stats(train_df)
    val_df = add_volume_stats(val_df)
    test_df = add_volume_stats(test_df)

    # Cache results
    train_df.to_parquet(train_cache, index=False)
    val_df.to_parquet(val_cache, index=False)
    test_df.to_parquet(test_cache, index=False)

    return train_df, val_df, test_df


def get_transforms(mode="train"):
    """
    Returns Albumentations transforms for the given mode.

    Args:
        mode (str): 'train', 'val', or 'test'.
    """
    if mode == "train":
        return A.Compose(
            [
                A.Resize(height=Config.IMAGE_SIZE, width=Config.IMAGE_SIZE),
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.1, rotate_limit=10, p=0.5
                ),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                    max_pixel_value=1.0,
                ),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(height=Config.IMAGE_SIZE, width=Config.IMAGE_SIZE),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                    max_pixel_value=1.0,
                ),
                ToTensorV2(),
            ]
        )


class GI_MRI_Dataset(Dataset):
    """
    Dataset class for 2.5D MRI Segmentation.
    Loads 3 slices (t-1, t, t+1) as channels.
    """

    def __init__(self, df, transforms=None, mode="train"):
        """
        Args:
            df (pd.DataFrame): Dataframe containing metadata and paths.
            transforms (albumentations.Compose): Transforms to apply.
            mode (str): 'train', 'val', or 'test'.
        """
        self.df = df
        self.transforms = transforms
        self.mode = mode
        self.input_dir = Config.INPUT_DIR
        self.classes = Config.CLASSES

    def __len__(self):
        return len(self.df)

    def load_slice(self, rel_path, vol_max):
        """
        Loads a single slice, converts to float32, and applies Volume-based normalization.
        Cite solution_lesson_node_00002: Avoid per-slice min-max; use volume stats.
        """
        full_path = os.path.join(self.input_dir, rel_path)
        img = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)

        if img is None:
            # Fallback for missing files
            img = np.zeros((266, 266), dtype=np.uint8)

        # Handle 16-bit images or other depths
        img = img.astype(np.float32)

        # Volume-based Normalization
        # Normalize by the 99th percentile of the volume to preserve relative intensity
        if vol_max > 0:
            img = img / vol_max
            img = np.clip(img, 0, 1.0)
        else:
            img = np.zeros_like(img)

        return img

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. Load 2.5D Images
        # Paths are pre-calculated in the dataframe
        path_curr = row["image_path"]
        path_prev = row["prev_image_path"]
        path_next = row["next_image_path"]

        # Use pre-computed volume max for normalization
        vol_max = row["vol_max"]

        img_curr = self.load_slice(path_curr, vol_max)
        img_prev = self.load_slice(path_prev, vol_max)
        img_next = self.load_slice(path_next, vol_max)

        # Stack to (H, W, 3)
        # Resize logic is handled by Albumentations, so we stack first.
        img_stack = np.stack([img_prev, img_curr, img_next], axis=-1)

        # 2. Load Masks (if not test)
        mask_stack = None
        if self.mode != "test":
            h, w = img_curr.shape[:2]
            masks = []
            for cls in self.classes:
                rle = row[cls]
                mask = rle_decode(rle, shape=(h, w))
                masks.append(mask)
            mask_stack = np.stack(masks, axis=-1)  # (H, W, 3)

        # 3. Augmentations
        if self.transforms:
            if self.mode != "test":
                augmented = self.transforms(image=img_stack, mask=mask_stack)
                img_tensor = augmented["image"]
                mask_tensor = augmented["mask"]

                # Albumentations ToTensorV2 converts image to (C, H, W).
                # For mask (H, W, C), if passed as 'mask', it is returned as (H, W, C) tensor or numpy.
                # ToTensorV2 usually handles image transposition.
                # We ensure mask is (C, H, W) for PyTorch.
                if isinstance(mask_tensor, torch.Tensor):
                    if (
                        mask_tensor.shape[-1] == len(self.classes)
                        and mask_tensor.ndim == 3
                    ):
                        mask_tensor = mask_tensor.permute(2, 0, 1)
                elif isinstance(mask_tensor, np.ndarray):
                    mask_tensor = torch.from_numpy(mask_tensor).permute(2, 0, 1)

                # Ensure float32 for BCE loss
                mask_tensor = mask_tensor.float()

                return img_tensor, mask_tensor
            else:
                augmented = self.transforms(image=img_stack)
                img_tensor = augmented["image"]

                # Return ID for submission mapping
                return img_tensor, row["id"]

        # Fallback if no transforms (should not happen in this pipeline)
        img_tensor = torch.from_numpy(img_stack).permute(2, 0, 1).float()
        if self.mode != "test":
            mask_tensor = torch.from_numpy(mask_stack).permute(2, 0, 1).float()
            return img_tensor, mask_tensor

        return img_tensor, row["id"]
