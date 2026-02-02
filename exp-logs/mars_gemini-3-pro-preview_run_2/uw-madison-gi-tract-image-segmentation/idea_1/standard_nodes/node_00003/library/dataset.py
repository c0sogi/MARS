import os
import cv2
import numpy as np
import pandas as pd
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from library.config import Config
from library.utils import rle_decode


def process_metadata(csv_path, split_name, load_cached_data=True):
    """
    Process metadata to create a wide-format dataframe (one row per slice)
    with neighbor file paths for 2.5D input. Implements caching.
    """
    cache_path = os.path.join(
        Config.CACHE_DIR, f"processed_{split_name}_metadata.parquet"
    )

    # 1. Try to load cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            print(f"Failed to load cache {cache_path}: {e}. Reprocessing...")

    # 2. Process from scratch
    raw_df = pd.read_csv(csv_path)

    # Pivot from long to wide format
    # We want one row per slice id.
    # Columns to keep constant per id: case, day, slice, file_path, img_width, img_height, pixel_spacing...
    # Columns to pivot: segmentation (based on class)

    # Identify static columns (metadata that doesn't change per class for the same slice)
    static_cols = [
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
    static_cols = [c for c in static_cols if c in raw_df.columns]

    # Create the wide dataframe
    df = raw_df.drop_duplicates(subset=["id"])[static_cols].copy()

    # Add segmentation columns for each class
    # We assume 3 classes: large_bowel, small_bowel, stomach
    if "segmentation" in raw_df.columns:
        for cls in Config.CLASS_LABELS:
            # Extract segmentation for this class
            cls_df = raw_df[raw_df["class"] == cls][["id", "segmentation"]]
            cls_df = cls_df.rename(columns={"segmentation": f"seg_{cls}"})
            df = pd.merge(df, cls_df, on="id", how="left")

    # Sort to find neighbors
    df = df.sort_values(["case", "day", "slice"]).reset_index(drop=True)

    # Vectorized neighbor finding
    # We want file_path of slice-1 and slice+1
    # Logic: Shift file_path column. Check if case and day match.

    file_paths = df["file_path"].values
    case_ids = df["case"].values
    day_ids = df["day"].values

    # Previous Slice
    prev_paths = np.roll(file_paths, 1)
    prev_cases = np.roll(case_ids, 1)
    prev_days = np.roll(day_ids, 1)

    # Mask where previous is not same scan (boundary condition)
    # If boundary, use current path (replicate padding)
    mask_prev = (prev_cases == case_ids) & (prev_days == day_ids)
    final_prev = np.where(mask_prev, prev_paths, file_paths)

    # Next Slice
    next_paths = np.roll(file_paths, -1)
    next_cases = np.roll(case_ids, -1)
    next_days = np.roll(day_ids, -1)

    mask_next = (next_cases == case_ids) & (next_days == day_ids)
    final_next = np.where(mask_next, next_paths, file_paths)

    df["prev_file_path"] = final_prev
    df["next_file_path"] = final_next

    # 3. Save to cache
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    df.to_parquet(cache_path, index=False)

    return df


class UWDataset(Dataset):
    def __init__(
        self, mode="train", transform=None, debug=False, load_cached_data=True
    ):
        """
        Args:
            mode (str): 'train', 'val', or 'test'.
            transform (albumentations.Compose): Augmentation pipeline.
            debug (bool): If True, use a small subset of data.
            load_cached_data (bool): Whether to use cached metadata.
        """
        self.mode = mode
        self.transform = transform
        self.debug = debug

        # Select correct metadata file
        if mode == "train":
            csv_path = Config.TRAIN_METADATA_PATH
        elif mode == "val":
            csv_path = Config.VAL_METADATA_PATH
        else:
            csv_path = Config.TEST_METADATA_PATH

        # Process metadata
        self.df = process_metadata(csv_path, mode, load_cached_data=load_cached_data)

        # Debug subset
        if self.debug:
            self.df = self.df.iloc[: Config.DEBUG_SAMPLE_SIZE].reset_index(drop=True)

        # Define default transforms if none provided
        if self.transform is None:
            if self.mode == "train":
                self.transform = A.Compose(
                    [
                        A.Resize(*Config.IMAGE_SIZE),
                        A.HorizontalFlip(p=0.5),
                        A.VerticalFlip(p=0.5),
                        A.ShiftScaleRotate(
                            shift_limit=0.0625, scale_limit=0.1, rotate_limit=10, p=0.5
                        ),
                        ToTensorV2(),
                    ]
                )
            else:
                self.transform = A.Compose([A.Resize(*Config.IMAGE_SIZE), ToTensorV2()])

    def __len__(self):
        return len(self.df)

    def load_slice(self, path):
        """
        Load a single slice, normalize with outlier clipping, and scale to [0, 1].
        """
        full_path = os.path.join(Config.INPUT_DIR, path)
        img = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)

        if img is None:
            # Fallback for missing files (should not happen with correct metadata)
            return np.zeros(Config.IMAGE_SIZE, dtype=np.float32)

        # Convert to float
        img = img.astype(np.float32)

        # Outlier Clipping (Top 1%)
        if img.max() > 0:
            val = np.percentile(img, 99)
            img = np.clip(img, 0, val)

            # Min-Max Normalization to [0, 1]
            img_min = img.min()
            img_max = img.max()
            if img_max > img_min:
                img = (img - img_min) / (img_max - img_min)
            else:
                img = np.zeros_like(img)

        return img

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. Load 2.5D Image (Slice-1, Slice, Slice+1)
        # Channels: [Prev, Current, Next]
        img_prev = self.load_slice(row["prev_file_path"])
        img_curr = self.load_slice(row["file_path"])
        img_next = self.load_slice(row["next_file_path"])

        # Stack to (H, W, 3)
        image = np.stack([img_prev, img_curr, img_next], axis=-1)

        # 2. Load Mask (if available)
        if self.mode in ["train", "val"]:
            # Original shape from metadata
            h, w = row["img_height"], row["img_width"]
            mask = np.zeros((h, w, Config.NUM_CLASSES), dtype=np.float32)

            for i, cls in enumerate(Config.CLASS_LABELS):
                seg_col = f"seg_{cls}"
                if seg_col in row and pd.notna(row[seg_col]):
                    mask[:, :, i] = rle_decode(row[seg_col], (h, w))

            # Apply Augmentations
            if self.transform:
                transformed = self.transform(image=image, mask=mask)
                image = transformed["image"]
                mask = transformed["mask"]

            # Ensure mask is channel-first for PyTorch: (C, H, W)
            # ToTensorV2 converts image to (C, H, W) but mask usually stays (H, W, C) or becomes (H, W)
            # Depending on Albumentations version, mask might need transpose if it's not handled by ToTensorV2 for multi-channel
            if mask.shape[0] == Config.IMAGE_SIZE[0]:  # Check if H is first dim
                mask = mask.permute(2, 0, 1)

            return image, mask

        else:
            # Test mode
            if self.transform:
                transformed = self.transform(image=image)
                image = transformed["image"]

            # Return original dimensions for resizing
            return image, row["id"], row["img_height"], row["img_width"]
