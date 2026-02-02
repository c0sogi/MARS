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


def load_img(path):
    """
    Loads an image from a path.
    Handles 16-bit PNGs correctly by reading as unchanged.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")

    # Load as unchanged to preserve bit depth (likely 16-bit)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    # Handle multi-channel if present (though usually these are grayscale)
    if img.ndim == 3:
        img = img[:, :, 0]

    return img.astype(np.float32)


def normalize_slice(img):
    """
    Applies Robust Per-Slice Normalization.
    1. Percentile Clipping [1, 99]
    2. Noise Suppression check
    3. Min-Max Scaling to [0, 1]
    """
    # Calculate percentiles
    p1 = np.percentile(img, Config.NORM_MIN_PERCENTILE)
    p99 = np.percentile(img, Config.NORM_MAX_PERCENTILE)

    dynamic_range = p99 - p1

    # Noise Suppression: if dynamic range is too low, treat as empty background
    if dynamic_range < Config.NOISE_THRESHOLD:
        return np.zeros_like(img)

    # Clipping
    img = np.clip(img, p1, p99)

    # Min-Max Scaling
    img = (img - p1) / (dynamic_range + 1e-6)  # Add epsilon to avoid div by zero

    # Ensure strict 0-1 bounds
    img = np.clip(img, 0, 1)

    return img


def prepare_data(df, load_cached_data=True, split="train"):
    """
    Prepares the dataframe for 2.5D training/inference.
    Adds columns for 'prev_path' and 'next_path' representing t-1 and t+1 slices.
    Implements caching using parquet to save processing time.
    """
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    cache_path = os.path.join(Config.WORKING_DIR, f"{split}_25d.parquet")

    if load_cached_data and os.path.exists(cache_path):
        # print(f"Loading cached 2.5D data from {cache_path}")
        return pd.read_parquet(cache_path)

    # print(f"Processing 2.5D data for {split}...")

    # Ensure sorting by case, day, slice to correctly identify neighbors
    # 'slice' might be string in metadata, cast to int for sorting safety if needed,
    # but usually string sorting works for '0001', '0002'.
    # To be safe, we rely on the string sort order of zero-padded slice numbers.
    df = df.sort_values(["case", "day", "slice"]).reset_index(drop=True)

    path_col = "image_path"

    # Create shifted columns to identify potential neighbors
    df["prev_path"] = df[path_col].shift(1)
    df["next_path"] = df[path_col].shift(-1)

    df["prev_case"] = df["case"].shift(1)
    df["next_case"] = df["case"].shift(-1)

    df["prev_day"] = df["day"].shift(1)
    df["next_day"] = df["day"].shift(-1)

    # Logic: If case and day match, the shifted row is a valid neighbor.
    # Otherwise, it's a boundary (start/end of scan), so duplicate the current slice.

    # Fix prev_path (boundary at start of scan)
    mask_prev = (df["case"] == df["prev_case"]) & (df["day"] == df["prev_day"])
    df.loc[~mask_prev, "prev_path"] = df.loc[~mask_prev, path_col]

    # Fix next_path (boundary at end of scan)
    mask_next = (df["case"] == df["next_case"]) & (df["day"] == df["next_day"])
    df.loc[~mask_next, "next_path"] = df.loc[~mask_next, path_col]

    # Drop temporary columns
    cols_to_drop = ["prev_case", "next_case", "prev_day", "next_day"]
    df = df.drop(columns=cols_to_drop)

    # Save to cache
    df.to_parquet(cache_path, index=False)

    return df


def get_transforms(data="train"):
    """
    Returns the Albumentations transformation pipeline.
    """
    if data == "train":
        return A.Compose(
            [
                A.Resize(
                    Config.IMG_HEIGHT, Config.IMG_WIDTH, interpolation=cv2.INTER_LINEAR
                ),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5
                ),
                A.OneOf(
                    [
                        A.ElasticTransform(
                            alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5
                        ),
                        A.GridDistortion(p=0.5),
                    ],
                    p=0.3,
                ),
                A.CoarseDropout(
                    max_holes=8,
                    max_height=Config.IMG_HEIGHT // 20,
                    max_width=Config.IMG_WIDTH // 20,
                    min_holes=5,
                    fill_value=0,
                    mask_fill_value=0,
                    p=0.2,
                ),
                A.GaussNoise(var_limit=(0.001, 0.005), p=0.2),
                ToTensorV2(),
            ]
        )
    elif data == "valid" or data == "test":
        return A.Compose(
            [
                A.Resize(
                    Config.IMG_HEIGHT, Config.IMG_WIDTH, interpolation=cv2.INTER_LINEAR
                ),
                ToTensorV2(),
            ]
        )
    return A.Compose([ToTensorV2()])


class UWDataset(Dataset):
    """
    Dataset class for UW-Madison GI Tract Segmentation.
    Handles 2.5D input generation, normalization, and augmentation.
    """

    def __init__(self, df, transforms=None, mode="train"):
        self.df = df
        self.transforms = transforms
        self.mode = mode
        self.input_dir = Config.INPUT_DIR

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full paths for the 2.5D slices
        path_curr = os.path.join(self.input_dir, row["image_path"])
        path_prev = os.path.join(self.input_dir, row["prev_path"])
        path_next = os.path.join(self.input_dir, row["next_path"])

        # Load Images (16-bit -> float32)
        img_curr = load_img(path_curr)
        img_prev = load_img(path_prev)
        img_next = load_img(path_next)

        # Apply Robust Per-Slice Normalization
        img_curr = normalize_slice(img_curr)
        img_prev = normalize_slice(img_prev)
        img_next = normalize_slice(img_next)

        # Stack to form 2.5D input: (H, W, 3)
        img = np.stack([img_prev, img_curr, img_next], axis=-1)

        if self.mode in ["train", "valid"]:
            # Load Masks
            # Masks are RLE encoded in the dataframe
            h, w = int(row["height"]), int(row["width"])
            masks = []
            for cls in Config.CLASSES:
                rle = row[cls]
                mask = rle_decode(rle, (h, w))
                masks.append(mask)

            # Stack masks: (H, W, C)
            mask = np.stack(masks, axis=-1)

            # Apply Augmentations
            if self.transforms:
                augmented = self.transforms(image=img, mask=mask)
                img = augmented["image"]
                mask = augmented["mask"]

            # Ensure mask is float for BCE loss
            mask = mask.float()

            # Permute image is handled by ToTensorV2 (H, W, C) -> (C, H, W)

            return img, mask

        else:
            # Inference mode (no masks)
            if self.transforms:
                augmented = self.transforms(image=img)
                img = augmented["image"]

            # Return image and metadata for reconstruction
            return img, row["id"], row["height"], row["width"]
