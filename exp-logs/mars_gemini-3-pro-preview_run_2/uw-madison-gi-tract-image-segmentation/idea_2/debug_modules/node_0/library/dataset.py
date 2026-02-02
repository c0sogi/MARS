import os
import numpy as np
import pandas as pd
import torch
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from library.config import Config
from library.utils import rle_decode


def prepare_data(csv_path, mode="train", load_cached_data=True):
    """
    Processes the metadata CSV to create a dataframe suitable for 2.5D training.
    Pivots the table to have one row per slice with columns for each class mask.
    Adds columns for previous and next slice file paths.

    Args:
        csv_path (str): Path to the metadata CSV.
        mode (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to load from cache if available.

    Returns:
        pd.DataFrame: Processed dataframe.
    """
    cache_filename = f"processed_{mode}_metadata.parquet"
    cache_path = os.path.join(Config.WORKING_DIR, cache_filename)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            # print(f"Loaded {mode} data from cache: {cache_path}")
            return df
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from scratch
    df = pd.read_csv(csv_path)

    # The train/val metadata is in long format (multiple rows per slice id).
    # Test metadata might already be one row per slice depending on submission format,
    # but based on provided metadata description, train.csv is long.

    # Pivot to wide format: one row per slice_id
    # We need to preserve file_path, case, day, slice, img_width, img_height
    # We want columns: large_bowel, small_bowel, stomach containing RLEs

    # Columns to keep constant per id
    meta_cols = [
        "id",
        "file_path",
        "case",
        "day",
        "slice",
        "img_width",
        "img_height",
        "pixel_spacing_w",
        "pixel_spacing_h",
    ]
    # Filter only existing columns
    meta_cols = [c for c in meta_cols if c in df.columns]

    # If 'segmentation' exists (train/val), pivot it.
    # If 'predicted' exists (test), it might be a placeholder, but usually test.csv is just a list of ids.
    # Based on description, test.csv is merged with file info.

    if "segmentation" in df.columns:
        # Pivot table
        df_pivot = df.pivot_table(
            index=meta_cols, columns="class", values="segmentation", aggfunc="first"
        ).reset_index()
        # Ensure all classes exist
        for c in ["large_bowel", "small_bowel", "stomach"]:
            if c not in df_pivot.columns:
                df_pivot[c] = np.nan
    else:
        # For test set, we might not need to pivot if it doesn't have class rows yet,
        # but the provided test_metadata.csv comes from test.csv which has 'class' column.
        # So we pivot similarly but values might be 'predicted' or empty.
        val_col = "predicted" if "predicted" in df.columns else None
        if val_col:
            df_pivot = df.pivot_table(
                index=meta_cols, columns="class", values=val_col, aggfunc="first"
            ).reset_index()
        else:
            # Just drop duplicates if no class info
            df_pivot = df[meta_cols].drop_duplicates().reset_index(drop=True)

        for c in ["large_bowel", "small_bowel", "stomach"]:
            if c not in df_pivot.columns:
                df_pivot[c] = ""

    # Sort by case, day, slice to find neighbors
    df_pivot = df_pivot.sort_values(["case", "day", "slice"]).reset_index(drop=True)

    # Identify neighbors
    # Shift file_path
    df_pivot["prev_file_path"] = df_pivot["file_path"].shift(1)
    df_pivot["next_file_path"] = df_pivot["file_path"].shift(-1)

    # Check boundaries: neighbor must belong to same case and day
    df_pivot["prev_case"] = df_pivot["case"].shift(1)
    df_pivot["prev_day"] = df_pivot["day"].shift(1)
    df_pivot["next_case"] = df_pivot["case"].shift(-1)
    df_pivot["next_day"] = df_pivot["day"].shift(-1)

    # If boundary, use current file path (replicate padding)
    mask_prev = (df_pivot["case"] == df_pivot["prev_case"]) & (
        df_pivot["day"] == df_pivot["prev_day"]
    )
    df_pivot.loc[~mask_prev, "prev_file_path"] = df_pivot.loc[~mask_prev, "file_path"]

    mask_next = (df_pivot["case"] == df_pivot["next_case"]) & (
        df_pivot["day"] == df_pivot["next_day"]
    )
    df_pivot.loc[~mask_next, "next_file_path"] = df_pivot.loc[~mask_next, "file_path"]

    # Clean up temp columns
    drop_cols = ["prev_case", "prev_day", "next_case", "next_day"]
    df_pivot = df_pivot.drop(columns=drop_cols)

    # 3. Save to cache
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    df_pivot.to_parquet(cache_path, index=False)
    # print(f"Saved processed {mode} data to {cache_path}")

    return df_pivot


class UWMadisonDataset(Dataset):
    def __init__(self, df, transforms=None, mode="train"):
        """
        Args:
            df (pd.DataFrame): Dataframe containing file paths and RLEs.
            transforms (albumentations.Compose): Augmentation pipeline.
            mode (str): 'train', 'val', or 'test'.
        """
        self.df = df
        self.transforms = transforms
        self.mode = mode
        self.input_dir = Config.INPUT_DIR

    def __len__(self):
        return len(self.df)

    def load_slice(self, rel_path):
        """Loads a single slice image."""
        full_path = os.path.join(self.input_dir, rel_path)
        img = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)

        # Handle 16-bit images (uint16) -> convert to float for normalization later
        # Or keep as is. Usually MRI is 16-bit.
        if img is None:
            # Fallback for missing files (should not happen with verified metadata)
            return np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE), dtype=np.float32)

        return img.astype(np.float32)

    def normalize(self, img_stack):
        """
        Applies Instance Min-Max Normalization.
        Args:
            img_stack: (H, W, C) numpy array
        Returns:
            normalized stack: (H, W, C) float32 in [0, 1]
        """
        min_val = img_stack.min()
        max_val = img_stack.max()

        if max_val - min_val > 0:
            img_stack = (img_stack - min_val) / (max_val - min_val)
        else:
            img_stack = np.zeros_like(img_stack)

        return img_stack.astype(np.float32)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. Load 2.5D Images
        # Paths are relative to input dir
        path_curr = row["file_path"]
        path_prev = row["prev_file_path"]
        path_next = row["next_file_path"]

        img_curr = self.load_slice(path_curr)
        img_prev = self.load_slice(path_prev)
        img_next = self.load_slice(path_next)

        # Stack: (H, W, 3)
        img_stack = np.stack([img_prev, img_curr, img_next], axis=-1)

        # 2. Normalize
        img_stack = self.normalize(img_stack)

        # 3. Handle Masks
        h, w = int(row["img_height"]), int(row["img_width"])

        # Resize logic is handled by albumentations, but we need original shape for decoding
        # The images loaded might be different sizes if the dataset varies,
        # but usually within a case they are consistent.
        # Note: The loaded images are original size.

        masks = np.zeros((img_stack.shape[0], img_stack.shape[1], 3), dtype=np.float32)

        if self.mode != "test":
            # Decode RLEs
            # Order: Large Bowel, Small Bowel, Stomach
            rle_lb = row["large_bowel"]
            rle_sb = row["small_bowel"]
            rle_st = row["stomach"]

            mask_lb = rle_decode(rle_lb, (h, w))
            mask_sb = rle_decode(rle_sb, (h, w))
            mask_st = rle_decode(rle_st, (h, w))

            masks[:, :, 0] = mask_lb
            masks[:, :, 1] = mask_sb
            masks[:, :, 2] = mask_st

        # 4. Augmentations / Resizing
        if self.transforms:
            data = self.transforms(image=img_stack, mask=masks)
            img_stack = data["image"]
            masks = data["mask"]

        # 5. Permute for PyTorch (H, W, C) -> (C, H, W)
        # Albumentations ToTensorV2 handles this for image, but sometimes we need manual control
        # If ToTensorV2 is used, it returns tensor. If not, numpy.
        # Assuming transforms includes ToTensorV2 or we do it manually.
        # Let's check if output is tensor.

        if not isinstance(img_stack, torch.Tensor):
            img_stack = np.transpose(img_stack, (2, 0, 1))
            img_stack = torch.from_numpy(img_stack)

        if not isinstance(masks, torch.Tensor):
            masks = np.transpose(masks, (2, 0, 1))
            masks = torch.from_numpy(masks)

        return {
            "image": img_stack,
            "mask": masks,
            "id": row["id"],
            # Return original size for post-processing if needed
            "orig_h": h,
            "orig_w": w,
        }


def get_transforms(mode="train"):
    """
    Returns albumentations transforms for train/val/test.
    """
    if mode == "train":
        return A.Compose(
            [
                A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                # Add more augmentations here if needed (e.g. ShiftScaleRotate, GridDistortion)
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose([A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE), ToTensorV2()])
