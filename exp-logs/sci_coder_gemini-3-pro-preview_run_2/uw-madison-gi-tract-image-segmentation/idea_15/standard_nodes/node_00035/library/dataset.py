import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import rle_decode


def load_image(path):
    """
    Loads a 16-bit PNG image.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")

    # Read as unchanged to preserve uint16 depth
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    # Ensure 2D (H, W) - handle cases where image might be saved as 3-channel
    if img is None:
        raise ValueError(f"Failed to load image: {path}")
    if img.ndim == 3:
        img = img[:, :, 0]

    return img


def normalize_image(img):
    """
    Normalizes image to 0-1 range based on instance max.
    """
    img = img.astype(np.float32)
    max_val = img.max()
    if max_val > 0:
        img /= max_val
    return img


def get_processed_dataframe(metadata_path, split_name="train", load_cached_data=True):
    """
    Loads metadata, pivots to wide format (one row per slice),
    adds neighbor file paths for 2.5D input, and caches the result.
    """
    cache_file = os.path.join(
        Config.WORKING_DIR, f"processed_dataset_{split_name}.parquet"
    )

    # 1. Load Cache if requested
    if load_cached_data and os.path.exists(cache_file):
        return pd.read_parquet(cache_file)

    # 2. Process from scratch
    df = pd.read_csv(metadata_path)

    # Define columns that are constant for a slice
    base_cols = [
        "id",
        "case",
        "day",
        "slice",
        "file_path",
        "img_width",
        "img_height",
        "pixel_spacing_w",
        "pixel_spacing_h",
    ]
    # Filter only existing columns
    base_cols = [c for c in base_cols if c in df.columns]

    # Deduplicate to get one row per slice
    df_base = df[base_cols].drop_duplicates(subset=["id"])

    # Pivot segmentation/class info if available
    if "class" in df.columns and "segmentation" in df.columns:
        # Pivot to get columns: segmentation_large_bowel, segmentation_small_bowel, segmentation_stomach
        df_pivot = df.pivot(
            index="id", columns="class", values="segmentation"
        ).reset_index()
        # Rename columns to avoid conflict
        df_pivot.columns = ["id"] + [
            f"segmentation_{c}" for c in df_pivot.columns if c != "id"
        ]
        # Merge back
        df_final = pd.merge(df_base, df_pivot, on="id", how="left")
    else:
        # Test set or other format
        df_final = df_base

    # Sort to ensure correct neighbor finding
    df_final = df_final.sort_values(["case", "day", "slice"]).reset_index(drop=True)

    # 3. Add Neighbors (2.5D Logic)
    # We need file_path for slice i-1 and i+1

    paths = df_final["file_path"]
    cases = df_final["case"]
    days = df_final["day"]
    slices = df_final["slice"]

    # Previous Slice
    # Logic: If prev row is (slice-1) and same case/day, use its path. Else use current path.
    prev_paths = paths.shift(1)
    prev_cases = cases.shift(1)
    prev_days = days.shift(1)
    prev_slices = slices.shift(1)

    mask_prev = (
        (cases == prev_cases) & (days == prev_days) & (slices == prev_slices + 1)
    )
    df_final["file_path_prev"] = np.where(mask_prev, prev_paths, paths)

    # Next Slice
    next_paths = paths.shift(-1)
    next_cases = cases.shift(-1)
    next_days = days.shift(-1)
    next_slices = slices.shift(-1)

    mask_next = (
        (cases == next_cases) & (days == next_days) & (slices == next_slices - 1)
    )
    df_final["file_path_next"] = np.where(mask_next, next_paths, paths)

    # 4. Save Cache
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    df_final.to_parquet(cache_file)

    return df_final


def balance_dataframe(df, random_state=42):
    """
    Balances the dataframe to have 50% positive (has mask) and 50% negative samples.
    """
    # Check for segmentation columns
    seg_cols = [c for c in df.columns if c.startswith("segmentation_")]
    if not seg_cols:
        return df

    # Vectorized check for mask existence
    # Fill NaN with empty string and check if any column has content
    df_filled = df[seg_cols].fillna("")
    mask_exists = (df_filled != "").any(axis=1)

    positives = df[mask_exists]
    negatives = df[~mask_exists]

    n_pos = len(positives)
    if n_pos == 0:
        return df

    # Sample negatives to match positive count
    # Use replacement if negatives are fewer than positives (unlikely here but safe)
    negatives_sampled = negatives.sample(
        n=n_pos,
        random_state=random_state,
        replace=True if len(negatives) < n_pos else False,
    )

    # Combine and shuffle
    df_balanced = (
        pd.concat([positives, negatives_sampled])
        .sample(frac=1.0, random_state=random_state)
        .reset_index(drop=True)
    )

    return df_balanced


class UWMapDataset(Dataset):
    def __init__(
        self, dataframe, mode="train", img_size=Config.IMG_SIZE, transforms=None
    ):
        self.df = dataframe
        self.mode = mode
        self.img_size = img_size

        # Setup Transforms
        if transforms is not None:
            self.transforms = transforms
        else:
            self.setup_transforms()

    def setup_transforms(self):
        """
        Defines Albumentations transforms based on current img_size and mode.
        """
        if self.mode == "train":
            self.transforms = A.Compose(
                [
                    A.Resize(self.img_size, self.img_size),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.ShiftScaleRotate(
                        shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5
                    ),
                    A.OneOf(
                        [
                            A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
                            A.ElasticTransform(
                                alpha=1, sigma=50, alpha_affine=50, p=1.0
                            ),
                        ],
                        p=0.25,
                    ),
                    ToTensorV2(),
                ]
            )
        else:
            # Val / Test
            self.transforms = A.Compose(
                [A.Resize(self.img_size, self.img_size), ToTensorV2()]
            )

    def update_img_size(self, new_size):
        """
        Updates the image size and rebuilds transforms. Used for Dynamic Scale Training.
        """
        self.img_size = new_size
        self.setup_transforms()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        # 1. Load Images (2.5D: Prev, Curr, Next)
        path_curr = os.path.join(Config.INPUT_DIR, row["file_path"])
        path_prev = os.path.join(Config.INPUT_DIR, row["file_path_prev"])
        path_next = os.path.join(Config.INPUT_DIR, row["file_path_next"])

        img_curr = load_image(path_curr)
        img_prev = load_image(path_prev)
        img_next = load_image(path_next)

        # Stack to create (H, W, 3)
        img = np.stack([img_prev, img_curr, img_next], axis=-1)

        # Normalize
        img = normalize_image(img)

        # 2. Handle Masks (Train/Val)
        if self.mode in ["train", "val"]:
            h, w = row["img_height"], row["img_width"]
            # Classes: large_bowel, small_bowel, stomach
            mask = np.zeros((h, w, 3), dtype=np.uint8)

            # Map columns to channel index
            # We assume fixed order: large_bowel, small_bowel, stomach
            class_map = {"large_bowel": 0, "small_bowel": 1, "stomach": 2}

            for cls_name, idx in class_map.items():
                col_name = f"segmentation_{cls_name}"
                if col_name in row and pd.notna(row[col_name]) and row[col_name] != "":
                    mask[..., idx] = rle_decode(row[col_name], shape=(h, w))

            # Apply Transforms
            transformed = self.transforms(image=img, mask=mask)
            img_tensor = transformed["image"]
            mask_tensor = transformed["mask"]

            # Permute mask to (C, H, W) as ToTensorV2 doesn't transpose mask automatically
            mask_tensor = mask_tensor.permute(2, 0, 1).float()

            return img_tensor, mask_tensor

        else:
            # Test Mode
            transformed = self.transforms(image=img)
            img_tensor = transformed["image"]
            return img_tensor
