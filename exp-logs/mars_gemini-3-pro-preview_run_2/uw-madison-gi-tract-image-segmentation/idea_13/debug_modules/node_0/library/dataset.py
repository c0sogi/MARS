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


def get_transforms(mode="train"):
    """
    Returns the Albumentations transformation pipeline.
    """
    if mode == "train":
        return A.Compose(
            [
                # Ensure image is at least the patch size before cropping
                A.PadIfNeeded(
                    min_height=Config.PATCH_SIZE[0],
                    min_width=Config.PATCH_SIZE[1],
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                ),
                # Random Crop for patch-based learning
                A.RandomCrop(height=Config.PATCH_SIZE[0], width=Config.PATCH_SIZE[1]),
                # Spatial Augmentations
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5
                ),
                # Deformations
                A.OneOf(
                    [
                        A.ElasticTransform(
                            alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5
                        ),
                        A.GridDistortion(p=0.5),
                        A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=0.5),
                    ],
                    p=0.3,
                ),
                ToTensorV2(),
            ]
        )
    else:
        # For Validation and Test, we return the full image.
        # Tiling or resizing logic is handled by the inference loop or model wrapper.
        return A.Compose([ToTensorV2()])


def prepare_data(csv_path, mode="train", load_cached_data=True):
    """
    Loads metadata, processes it for 2.5D input (prev/next slice paths),
    pivots segmentation masks, and caches the result.
    """
    cache_filename = f"{mode}_processed.parquet"
    cache_path = os.path.join(Config.WORKING_DIR, cache_filename)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached {mode} data from {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"Processing {mode} data from {csv_path}...")

    # 2. Load CSV
    df = pd.read_csv(csv_path)

    # 3. Pivot and Aggregate
    # The metadata contains multiple rows per slice (one for each class).
    # We want one row per slice with columns for each class's segmentation.

    # Extract static columns (file info) - drop duplicates to get unique slices
    # We assume file_path, case, day, slice, etc. are consistent for the same ID
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
    # Filter only columns that actually exist in the input df
    available_meta_cols = [c for c in meta_cols if c in df.columns]

    df_meta = df[available_meta_cols].drop_duplicates(subset=["id"])

    # Handle Segmentation Masks
    if "segmentation" in df.columns:
        # Pivot to get columns: large_bowel, small_bowel, stomach
        df_seg = df.pivot(
            index="id", columns="class", values="segmentation"
        ).reset_index()
        # Rename columns to avoid confusion
        df_seg.columns.name = None  # Remove index name
        # Ensure all classes exist
        for cls in Config.CLASSES:
            if cls not in df_seg.columns:
                df_seg[cls] = ""
    else:
        # Test set might not have segmentation or class columns in the same way
        # Create empty placeholders
        df_seg = pd.DataFrame({"id": df_meta["id"]})
        for cls in Config.CLASSES:
            df_seg[cls] = ""

    # Merge metadata with segmentation
    df_processed = pd.merge(df_meta, df_seg, on="id", how="left")

    # 4. Generate 2.5D Context (Previous and Next Slice Paths)
    # Sort by case, day, slice to ensure correct ordering
    df_processed.sort_values(by=["case", "day", "slice"], ascending=True, inplace=True)

    # Group by case and day to avoid shifting across different scans
    groups = df_processed.groupby(["case", "day"])

    # Shift file paths
    df_processed["file_path_prev"] = groups["file_path"].shift(1)
    df_processed["file_path_next"] = groups["file_path"].shift(-1)

    # Fill NaNs (boundary slices) with the current slice's path
    df_processed["file_path_prev"] = df_processed["file_path_prev"].fillna(
        df_processed["file_path"]
    )
    df_processed["file_path_next"] = df_processed["file_path_next"].fillna(
        df_processed["file_path"]
    )

    # 5. Cache the result
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    df_processed.to_parquet(cache_path, index=False)
    print(f"Saved processed data to {cache_path}")

    return df_processed


class GIDataset(Dataset):
    def __init__(self, df, mode="train", transforms=None):
        """
        Args:
            df (pd.DataFrame): Processed dataframe from prepare_data.
            mode (str): 'train', 'val', or 'test'.
            transforms (albumentations.Compose): Transforms to apply.
        """
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.transforms = transforms
        self.input_dir = Config.INPUT_DIR

    def __len__(self):
        return len(self.df)

    def load_slice(self, rel_path):
        """Loads a single slice image."""
        full_path = os.path.join(self.input_dir, rel_path)
        img = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)

        if img is None:
            # Fallback for missing files (should not happen if metadata is correct)
            # Return a blank image of standard size or raise error
            raise FileNotFoundError(f"Image not found: {full_path}")

        # Ensure image is 16-bit or 8-bit, convert to float for processing
        img = img.astype(np.float32)
        return img

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. Load 2.5D Images (Channels: Prev, Curr, Next)
        # Paths are pre-calculated in the dataframe
        path_prev = row["file_path_prev"]
        path_curr = row["file_path"]
        path_next = row["file_path_next"]

        img_prev = self.load_slice(path_prev)
        img_curr = self.load_slice(path_curr)
        img_next = self.load_slice(path_next)

        # Stack to create (H, W, 3) image
        img = np.stack([img_prev, img_curr, img_next], axis=-1)

        # 2. Normalization (Instance Min-Max)
        # Normalize the entire stack based on its own statistics
        # Adding epsilon to avoid division by zero
        min_val = img.min()
        max_val = img.max()
        if max_val - min_val > 0:
            img = (img - min_val) / (max_val - min_val)
        else:
            img = np.zeros_like(img)

        # 3. Load Masks
        h, w = int(row["img_height"]), int(row["img_width"])
        masks = []

        for cls in Config.CLASSES:
            rle = row[cls]
            # If rle is NaN or empty, rle_decode handles it (returns zeros)
            mask = rle_decode(rle, (h, w))
            masks.append(mask)

        # Stack masks to (H, W, C) -> (H, W, 3)
        mask_stack = np.stack(masks, axis=-1).astype(np.float32)

        # 4. Augmentations
        if self.transforms:
            augmented = self.transforms(image=img, mask=mask_stack)
            img = augmented["image"]
            mask_stack = augmented["mask"]

        # 5. Prepare Output
        # Mask is (C, H, W) after ToTensorV2, Image is (3, H, W)

        return {
            "image": img,
            "mask": mask_stack,
            "id": row["id"],
            "img_height": h,
            "img_width": w,
            "pixel_spacing_w": row.get("pixel_spacing_w", 1.0),
            "pixel_spacing_h": row.get("pixel_spacing_h", 1.0),
        }
