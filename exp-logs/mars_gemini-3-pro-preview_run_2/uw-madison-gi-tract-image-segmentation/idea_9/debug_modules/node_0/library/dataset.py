import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import load_image, rle_decode


class UWMDataset(Dataset):
    """
    Dataset class for Stomach and Intestines Segmentation (2.5D BiSeNet).
    Handles 2.5D stack generation (prev, curr, next slices), resizing, and normalization.
    """

    def __init__(self, metadata, phase="train", load_cached_data=True):
        self.phase = phase
        self.classes = Config.CLASS_LABELS
        self.img_size = Config.IMG_SIZE  # (H, W)

        # Define cache path based on phase
        cache_name = f"processed_{phase}_dataframe.parquet"
        self.cache_path = os.path.join(Config.WORKING_DIR, cache_name)

        # Load data (with caching mechanism)
        self.df = self._load_data(metadata, load_cached_data)

        # Apply sampling strategy for training
        if self.phase == "train":
            self.df = self._apply_sampling(self.df)

    def _load_data(self, metadata, load_cached_data):
        """
        Loads metadata, pivots to wide format, adds neighbor paths, and caches result.
        """
        # 1. Try loading from cache
        if load_cached_data and os.path.exists(self.cache_path):
            return pd.read_parquet(self.cache_path)

        # 2. Process from scratch
        if isinstance(metadata, str):
            df = pd.read_csv(metadata)
        else:
            df = metadata.copy()

        # Ensure working directory exists
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

        # Columns required for processing
        required_cols = [
            "id",
            "case",
            "day",
            "slice",
            "file_path",
            "img_width",
            "img_height",
        ]

        # Pivot logic: Convert long format (multiple rows per id) to wide (one row per id)
        # Train/Val usually have 'class' and 'segmentation'
        if "class" in df.columns and "segmentation" in df.columns:
            pivot_df = df.pivot_table(
                index=required_cols,
                columns="class",
                values="segmentation",
                aggfunc="first",
            ).reset_index()

            # Ensure all class columns exist
            for cls in self.classes:
                if cls not in pivot_df.columns:
                    pivot_df[cls] = np.nan
        else:
            # Test set or inference mode (just unique slices)
            pivot_df = df[required_cols].drop_duplicates().reset_index(drop=True)

        # 3. Add Neighbors (2.5D Logic)
        # Sort to ensure correct sequential ordering
        pivot_df = pivot_df.sort_values(["case", "day", "slice"])

        # Create a group key to identify boundaries
        # We use numpy arrays for fast vectorized shifting
        cases = pivot_df["case"].values
        days = pivot_df["day"].values
        paths = pivot_df["file_path"].values

        # Identify groups (case + day)
        # We can detect boundaries where case or day changes
        group_ids = cases * 1000 + days  # Simple hash assuming day < 1000

        # Shift paths
        prev_paths = np.roll(paths, 1)
        next_paths = np.roll(paths, -1)

        prev_groups = np.roll(group_ids, 1)
        next_groups = np.roll(group_ids, -1)

        # Handle boundaries: if group changes, use current path (replicate padding)
        # Also fix first and last indices explicitly
        prev_paths[0] = paths[0]
        next_paths[-1] = paths[-1]

        # Mask invalid shifts (where group ID changed)
        invalid_prev = group_ids != prev_groups
        prev_paths[invalid_prev] = paths[invalid_prev]

        invalid_next = group_ids != next_groups
        next_paths[invalid_next] = paths[invalid_next]

        # Assign back to dataframe
        pivot_df["prev_path"] = prev_paths
        pivot_df["next_path"] = next_paths

        # Save to cache
        pivot_df.to_parquet(self.cache_path)

        return pivot_df

    def _apply_sampling(self, df):
        """
        Applies balanced sampling: Keep all positives, subsample negatives.
        """
        # Determine which columns contain masks
        available_classes = [c for c in self.classes if c in df.columns]
        if not available_classes:
            return df

        # Identify positive samples (at least one mask is not NaN/Empty)
        # Note: In the CSV, empty masks might be NaN.
        has_mask = df[available_classes].notna().any(axis=1)

        pos_df = df[has_mask]
        neg_df = df[~has_mask]

        # Subsample negatives
        if not neg_df.empty:
            neg_df = neg_df.sample(
                frac=Config.NEGATIVE_SAMPLING_RATIO, random_state=Config.SEED
            )

        # Combine and shuffle
        sampled_df = (
            pd.concat([pos_df, neg_df])
            .sample(frac=1.0, random_state=Config.SEED)
            .reset_index(drop=True)
        )
        return sampled_df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. Load 2.5D Image Stack
        curr_img = load_image(row["file_path"])
        prev_img = load_image(row["prev_path"])
        next_img = load_image(row["next_path"])

        # Stack: (H, W, 3)
        img_stack = np.stack([prev_img, curr_img, next_img], axis=-1)

        # 2. Normalize (Min-Max to [0, 1])
        mi, ma = np.min(img_stack), np.max(img_stack)
        if ma > mi:
            img_stack = (img_stack - mi) / (ma - mi)
        else:
            img_stack = np.zeros_like(img_stack)

        img_stack = img_stack.astype(np.float32)

        # 3. Resize
        # cv2.resize takes (width, height)
        target_h, target_w = self.img_size
        img_stack = cv2.resize(
            img_stack, (target_w, target_h), interpolation=cv2.INTER_LINEAR
        )

        # Transpose to (C, H, W)
        img_tensor = torch.from_numpy(img_stack.transpose(2, 0, 1))

        # 4. Load Masks (if not test)
        if self.phase != "test":
            masks = []
            orig_h = row["img_height"]
            orig_w = row["img_width"]

            for cls in self.classes:
                rle = row[cls] if cls in row else None
                # Decode using original shape
                mask = rle_decode(rle, (orig_h, orig_w))
                # Resize to target shape (Nearest Neighbor for masks)
                mask = cv2.resize(
                    mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST
                )
                masks.append(mask)

            # Stack masks: (3, H, W)
            mask_stack = np.stack(masks, axis=0)
            mask_tensor = torch.from_numpy(mask_stack.astype(np.float32))

            return img_tensor, mask_tensor, row["id"]

        else:
            return img_tensor, row["id"]


def get_loaders(
    train_df, val_df, test_df=None, batch_size=32, num_workers=4, load_cached_data=True
):
    """
    Creates DataLoaders for train, validation, and optionally test sets.
    Applies balanced sampling logic to the training set.
    """
    # Train Loader
    train_ds = UWMDataset(train_df, phase="train", load_cached_data=load_cached_data)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Val Loader
    val_ds = UWMDataset(val_df, phase="val", load_cached_data=load_cached_data)
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Test Loader
    test_loader = None
    if test_df is not None:
        test_ds = UWMDataset(test_df, phase="test", load_cached_data=load_cached_data)
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    return train_loader, val_loader, test_loader
