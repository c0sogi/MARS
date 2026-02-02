import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import rle_decode


def process_metadata(csv_path, mode="train", load_cached_data=True):
    """
    Processes the metadata CSV from long format to wide format (one row per slice).
    Implements caching to parquet.
    """
    cache_path = os.path.join(Config.WORKING_DIR, f"{mode}_processed.parquet")

    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            print(f"Failed to load cache {cache_path}: {e}. Reprocessing...")

    # Load raw CSV
    df = pd.read_csv(csv_path)

    # Identify columns to group by (all metadata except class-specific info)
    # Common columns: id, case, day, slice, file_path, img_width, img_height, pixel_spacing_w, pixel_spacing_h
    # Variable columns: class, segmentation (train/val) or predicted (test)

    group_cols = [
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

    # Ensure columns exist (test set might differ slightly)
    available_group_cols = [c for c in group_cols if c in df.columns]

    # Determine value column for pivot
    value_col = "segmentation"
    if mode == "test":
        # Test set usually doesn't have ground truth, but might have 'predicted' placeholder
        # We don't need mask content for test, but we need the structure.
        # If 'segmentation' is missing, we just create the wide structure without mask data.
        if "segmentation" not in df.columns:
            # Drop duplicates to get unique slices
            df_wide = df[available_group_cols].drop_duplicates().reset_index(drop=True)
            # Save and return
            os.makedirs(Config.WORKING_DIR, exist_ok=True)
            df_wide.to_parquet(cache_path)
            return df_wide

    # Pivot logic for Train/Val
    df_wide = df.pivot_table(
        index=available_group_cols, columns="class", values=value_col, aggfunc="first"
    ).reset_index()

    # Ensure all classes are present as columns
    for cls in Config.CLASS_LABELS:
        if cls not in df_wide.columns:
            df_wide[cls] = np.nan

    # Save cache
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    df_wide.to_parquet(cache_path)

    return df_wide


class UWMadisonDataset(Dataset):
    def __init__(self, df, mode="train", transform=None, root_dir=Config.INPUT_DIR):
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.transform = transform
        self.root_dir = root_dir

        # Create a lookup for file paths: (case, day, slice) -> file_path
        # This allows O(1) access to neighbor slices
        self.path_map = dict(
            zip(
                zip(self.df["case"], self.df["day"], self.df["slice"]),
                self.df["file_path"],
            )
        )

        # Pre-check if mask columns exist
        self.has_masks = all(c in self.df.columns for c in Config.CLASS_LABELS) and (
            mode != "test"
        )

    def __len__(self):
        return len(self.df)

    def load_slice_img(self, case, day, slice_idx):
        """
        Loads a single slice image. Returns normalized float32 image.
        """
        key = (case, day, slice_idx)
        path = self.path_map.get(key)

        if path is None:
            return None

        full_path = os.path.join(self.root_dir, path)
        # Load 16-bit PNG
        img = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)

        if img is None:
            # Fallback for missing files (should not happen with valid metadata)
            return np.zeros((Config.IMG_SIZE[0], Config.IMG_SIZE[1]), dtype=np.float32)

        return img.astype(np.float32)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        case = row["case"]
        day = row["day"]
        slice_idx = row["slice"]

        # --- Load 2.5D Input (Slice i-1, i, i+1) ---
        img_curr = self.load_slice_img(case, day, slice_idx)
        img_prev = self.load_slice_img(case, day, slice_idx - 1)
        img_next = self.load_slice_img(case, day, slice_idx + 1)

        # Handle edge cases (start/end of scan) by replicating current slice
        if img_prev is None:
            img_prev = img_curr
        if img_next is None:
            img_next = img_curr

        # Stack channels: (H, W, 3)
        img_stack = np.stack([img_prev, img_curr, img_next], axis=-1)

        # --- Min-Max Normalization ---
        # Scale to [0, 1] per sample
        min_val = img_stack.min()
        max_val = img_stack.max()
        if max_val > min_val:
            img_stack = (img_stack - min_val) / (max_val - min_val)
        else:
            img_stack = np.zeros_like(img_stack)

        # --- Load Masks (Train/Val only) ---
        mask_stack = None
        if self.has_masks:
            h, w = int(row["img_height"]), int(row["img_width"])
            masks = []
            for cls in Config.CLASS_LABELS:
                rle = row[cls]
                mask = rle_decode(rle, shape=(h, w))
                masks.append(mask)

            # Stack masks: (H, W, C) -> C=3
            mask_stack = np.stack(masks, axis=-1).astype(np.float32)

        # --- Augmentation & Resizing ---
        if self.transform:
            if mask_stack is not None:
                transformed = self.transform(image=img_stack, mask=mask_stack)
                img_stack = transformed["image"]
                mask_stack = transformed["mask"]
            else:
                transformed = self.transform(image=img_stack)
                img_stack = transformed["image"]
        else:
            # If no transform provided, convert to tensor manually (fallback)
            img_stack = torch.from_numpy(img_stack.transpose(2, 0, 1))
            if mask_stack is not None:
                mask_stack = torch.from_numpy(mask_stack.transpose(2, 0, 1))

        # Prepare output
        result = {
            "image": img_stack,
            "id": row["id"],
            "case": case,
            "day": day,
            "slice": slice_idx,
            "img_width": row["img_width"],
            "img_height": row["img_height"],
        }

        if mask_stack is not None:
            # Ensure mask is (C, H, W)
            if isinstance(mask_stack, np.ndarray):
                mask_stack = torch.from_numpy(mask_stack.transpose(2, 0, 1))
            result["mask"] = mask_stack

        return result


def get_dataloaders(
    train_meta_path=Config.TRAIN_METADATA_PATH,
    val_meta_path=Config.VAL_METADATA_PATH,
    test_meta_path=Config.TEST_METADATA_PATH,
    batch_size=Config.BATCH_SIZE,
    load_cached_data=True,
    debug=Config.DEBUG,
):
    """
    Constructs DataLoaders for train, val, and test sets.
    Handles balanced sampling for training.
    """
    # 1. Define Transforms
    train_transform = A.Compose(
        [
            A.Resize(height=Config.IMG_SIZE[0], width=Config.IMG_SIZE[1]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            ToTensorV2(),
        ]
    )

    val_test_transform = A.Compose(
        [A.Resize(height=Config.IMG_SIZE[0], width=Config.IMG_SIZE[1]), ToTensorV2()]
    )

    # 2. Process Metadata
    df_train_full = process_metadata(
        train_meta_path, mode="train", load_cached_data=load_cached_data
    )
    df_val = process_metadata(
        val_meta_path, mode="val", load_cached_data=load_cached_data
    )
    df_test = process_metadata(
        test_meta_path, mode="test", load_cached_data=load_cached_data
    )

    # 3. Balanced Sampling for Training
    # Calculate mask area/existence to filter negatives
    # We check if any of the class columns have a non-null RLE
    mask_cols = [c for c in Config.CLASS_LABELS if c in df_train_full.columns]

    # Helper to check if row has any mask
    def has_mask(row):
        for c in mask_cols:
            if isinstance(row[c], str) and len(row[c]) > 0:
                return True
        return False

    # Vectorized check
    has_mask_mask = df_train_full[mask_cols].notna().any(axis=1)

    df_pos = df_train_full[has_mask_mask].copy()
    df_neg = df_train_full[~has_mask_mask].copy()

    # Sample negatives
    if len(df_neg) > 0:
        df_neg = df_neg.sample(
            frac=Config.NEGATIVE_SAMPLE_RATIO, random_state=Config.SEED
        )

    df_train = (
        pd.concat([df_pos, df_neg])
        .sample(frac=1.0, random_state=Config.SEED)
        .reset_index(drop=True)
    )

    # Debug Subsampling
    if debug:
        df_train = df_train.head(Config.DEBUG_SAMPLE_SIZE)
        df_val = df_val.head(Config.DEBUG_SAMPLE_SIZE)
        df_test = df_test.head(Config.DEBUG_SAMPLE_SIZE)

    print(
        f"Dataset Sizes - Train: {len(df_train)} (Pos: {len(df_pos)}, Neg Sampled: {len(df_neg)}), Val: {len(df_val)}, Test: {len(df_test)}"
    )

    # 4. Create Datasets
    train_dataset = UWMadisonDataset(df_train, mode="train", transform=train_transform)
    val_dataset = UWMadisonDataset(df_val, mode="val", transform=val_test_transform)
    test_dataset = UWMadisonDataset(df_test, mode="test", transform=val_test_transform)

    # 5. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
