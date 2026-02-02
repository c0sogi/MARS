import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import load_metadata, min_max_normalize


def rle_decode(mask_rle, shape):
    """
    Decodes a Run-Length Encoded string into a binary mask.

    Args:
        mask_rle (str): RLE string (start length start length ...).
        shape (tuple): (height, width) of the mask.

    Returns:
        np.ndarray: Binary mask of shape (height, width).
    """
    if pd.isna(mask_rle) or mask_rle == "":
        return np.zeros(shape, dtype=np.uint8)

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths

    # The dataset uses column-major (Fortran) flattening
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    return img.reshape(shape, order="F")


def process_metadata(split, load_cached_data=True):
    """
    Processes metadata to construct 2.5D inputs and handle sampling.
    Caches the result to parquet.

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: Processed dataframe with neighbor paths and sampling applied.
    """
    cache_path = os.path.join(Config.WORKING_DIR, f"processed_{split}_metadata.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Load raw metadata
    df_raw = load_metadata(split)

    # 3. Pivot to consolidate classes per slice
    # We need to preserve file info. Group by ID and take first for metadata,
    # and pivot for segmentations.

    # Columns that are constant per slice
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

    # Create base dataframe with unique slices
    df_slices = df_raw[meta_cols].drop_duplicates(subset=["id"]).set_index("id")

    # Pivot segmentation
    # If 'segmentation' column exists (train/val), pivot it.
    # Test set might have 'predicted' or empty segmentation.
    if "segmentation" in df_raw.columns:
        df_seg = df_raw.pivot(index="id", columns="class", values="segmentation")
        # Ensure all classes exist
        for cls in ["large_bowel", "small_bowel", "stomach"]:
            if cls not in df_seg.columns:
                df_seg[cls] = np.nan

        df = df_slices.join(df_seg)
    else:
        # For test set, we might not have segmentation column or it might be in a different format
        # If strictly following metadata format from description, test_metadata has 'predicted'
        # We initialize empty columns for consistency if needed, but mainly we need file paths
        df = df_slices.copy()
        for cls in ["large_bowel", "small_bowel", "stomach"]:
            df[cls] = np.nan

    # Reset index to make 'id' a column again
    df = df.reset_index()

    # 4. Sort to identify neighbors
    df = df.sort_values(["case", "day", "slice"]).reset_index(drop=True)

    # 5. Identify 2.5D neighbors (i-1, i+1)
    # We use file paths.
    file_paths = df["file_path"].values
    case_ids = df["case"].values
    day_ids = df["day"].values

    prev_paths = []
    next_paths = []

    n = len(df)
    for i in range(n):
        # Previous slice
        if i > 0 and case_ids[i] == case_ids[i - 1] and day_ids[i] == day_ids[i - 1]:
            prev_paths.append(file_paths[i - 1])
        else:
            # Boundary: replicate current
            prev_paths.append(file_paths[i])

        # Next slice
        if (
            i < n - 1
            and case_ids[i] == case_ids[i + 1]
            and day_ids[i] == day_ids[i + 1]
        ):
            next_paths.append(file_paths[i + 1])
        else:
            # Boundary: replicate current
            next_paths.append(file_paths[i])

    df["prev_path"] = prev_paths
    df["next_path"] = next_paths

    # 6. Sampling (Train only)
    if split == "train":
        # Calculate mask existence
        # Check if any of the class columns have a non-empty string
        def has_mask(row):
            for cls in ["large_bowel", "small_bowel", "stomach"]:
                if isinstance(row[cls], str) and len(row[cls]) > 0:
                    return True
            return False

        df["has_mask"] = df.apply(has_mask, axis=1)

        pos_df = df[df["has_mask"]].copy()
        neg_df = df[~df["has_mask"]].copy()

        # Sample negatives
        if len(neg_df) > 0:
            n_neg = int(len(neg_df) * Config.NEGATIVE_SAMPLE_RATIO)
            # Ensure at least some negatives if ratio > 0
            if n_neg == 0 and Config.NEGATIVE_SAMPLE_RATIO > 0:
                n_neg = min(len(neg_df), 100)  # Fallback

            if n_neg > 0:
                neg_df = neg_df.sample(n=n_neg, random_state=Config.SEED)
            else:
                neg_df = pd.DataFrame(columns=df.columns)

        df = (
            pd.concat([pos_df, neg_df], axis=0)
            .sample(frac=1, random_state=Config.SEED)
            .reset_index(drop=True)
        )

    # 7. Cache results
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    df.to_parquet(cache_path, index=False)

    return df


class UWMadisonDataset(Dataset):
    def __init__(self, split="train", load_cached_data=True):
        """
        Args:
            split (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to use cached metadata.
        """
        self.split = split
        self.df = process_metadata(split, load_cached_data=load_cached_data)
        self.classes = ["large_bowel", "small_bowel", "stomach"]

        # Transforms
        if split == "train":
            self.transforms = A.Compose(
                [
                    A.Resize(
                        height=Config.IMG_SIZE[0], width=Config.IMG_SIZE[1], p=1.0
                    ),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.Rotate(limit=15, p=0.5),
                    ToTensorV2(transpose_mask=True),
                ]
            )
        else:
            self.transforms = A.Compose(
                [
                    A.Resize(
                        height=Config.IMG_SIZE[0], width=Config.IMG_SIZE[1], p=1.0
                    ),
                    ToTensorV2(transpose_mask=True),
                ]
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. Load Images (2.5D: prev, curr, next)
        img_paths = [row["prev_path"], row["file_path"], row["next_path"]]
        images = []

        for path in img_paths:
            full_path = os.path.join(Config.INPUT_DIR, path)
            # Load as uint16 (unchanged)
            img = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                # Fallback for missing files (should not happen if metadata is verified)
                img = np.zeros((row["img_height"], row["img_width"]), dtype=np.uint16)

            # Handle if image is not 2D (e.g. already has channels, though unlikely for slice)
            if len(img.shape) > 2:
                img = img[:, :, 0]

            images.append(img)

        # Stack to create (H, W, 3)
        img_stack = np.stack(images, axis=-1)

        # 2. Load Masks
        # Shape for decoding
        h, w = row["img_height"], row["img_width"]
        mask_stack = np.zeros((h, w, len(self.classes)), dtype=np.float32)

        if self.split != "test":
            for i, cls in enumerate(self.classes):
                rle = row[cls]
                if isinstance(rle, str) and len(rle) > 0:
                    mask = rle_decode(rle, (h, w))
                    mask_stack[:, :, i] = mask

        # 3. Normalization
        # Apply min-max normalization to [0, 1]
        img_stack = min_max_normalize(img_stack)

        # 4. Augmentation & Tensor Conversion
        # Albumentations expects H, W, C
        augmented = self.transforms(image=img_stack, mask=mask_stack)
        img_tensor = augmented["image"]  # (3, H, W) via ToTensorV2
        mask_tensor = augmented["mask"]  # (3, H, W) via ToTensorV2(transpose_mask=True)

        # 5. Return
        return {
            "image": img_tensor.float(),
            "mask": mask_tensor.float(),
            "id": row["id"],
            "img_height": h,
            "img_width": w,
            "pixel_spacing_h": row["pixel_spacing_h"],
            "pixel_spacing_w": row["pixel_spacing_w"],
        }
