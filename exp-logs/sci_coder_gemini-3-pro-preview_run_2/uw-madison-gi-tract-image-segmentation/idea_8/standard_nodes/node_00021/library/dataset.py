import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import rle_decode, set_seed


class MRIDataset(Dataset):
    """
    PyTorch Dataset for 2.5D MRI Segmentation.
    Loads a stack of 3 slices (i-1, i, i+1) as channels to provide volumetric context.
    """

    def __init__(self, df, path_lookup, transforms=None, mode="train"):
        self.df = df
        self.path_lookup = path_lookup
        self.transforms = transforms
        self.mode = mode

        # Initialize CLAHE (Contrast Limited Adaptive Histogram Equalization)
        if Config.USE_CLAHE:
            self.clahe = cv2.createCLAHE(
                clipLimit=Config.CLAHE_CLIP_LIMIT,
                tileGridSize=Config.CLAHE_TILE_GRID_SIZE,
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        case = row["case"]
        day = row["day"]
        slice_num = row["slice"]

        # ---------------------------
        # 1. Load 2.5D Image Stack
        # ---------------------------
        images = []
        # Retrieve neighbors: [slice-1, slice, slice+1]
        for s_offset in [-1, 0, 1]:
            target_slice = slice_num + s_offset

            # Lookup file path using (case, day, slice) tuple
            path = self.path_lookup.get((case, day, target_slice))

            # Handle boundary conditions: if neighbor doesn't exist, replicate current slice
            if path is None:
                path = self.path_lookup.get((case, day, slice_num))

            # Load Image
            # Use IMREAD_UNCHANGED to preserve 16-bit depth if present
            full_path = os.path.join(Config.INPUT_DIR, path)
            img = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)

            # Safety fallback for missing files
            if img is None:
                img = np.zeros((row["img_height"], row["img_width"]), dtype=np.uint8)

            # Normalize 16-bit/High-bit images to 8-bit [0, 255]
            # We use Min-Max normalization per image to maximize local contrast
            if img.max() > 0:
                img = img.astype(np.float32)
                img = (img - img.min()) / (img.max() - img.min()) * 255.0
                img = img.astype(np.uint8)
            else:
                img = img.astype(np.uint8)

            # Apply CLAHE if enabled
            if Config.USE_CLAHE:
                img = self.clahe.apply(img)

            images.append(img)

        # Stack slices along channel dimension -> (H, W, 3)
        img_stack = np.stack(images, axis=-1)

        # ---------------------------
        # 2. Load Masks (Train/Val)
        # ---------------------------
        if self.mode != "test":
            h, w = row["img_height"], row["img_width"]
            masks = []
            # Classes: large_bowel, small_bowel, stomach
            for cls in ["large_bowel", "small_bowel", "stomach"]:
                rle = row[cls]
                # Decode RLE to binary mask
                if pd.isna(rle) or rle == "":
                    mask = np.zeros((h, w), dtype=np.uint8)
                else:
                    mask = rle_decode(rle, (h, w))
                masks.append(mask)

            # Stack masks -> (H, W, 3)
            mask_stack = np.stack(masks, axis=-1)

            # ---------------------------
            # 3. Augmentations
            # ---------------------------
            if self.transforms:
                augmented = self.transforms(image=img_stack, mask=mask_stack)
                img_stack = augmented["image"]
                mask_stack = augmented["mask"]

            # Convert mask to (C, H, W) float tensor for PyTorch
            # ToTensorV2 usually converts image to (C, H, W) but mask stays (H, W, C)
            if isinstance(mask_stack, torch.Tensor):
                mask_stack = mask_stack.permute(2, 0, 1).float()
            else:
                mask_stack = torch.from_numpy(mask_stack).permute(2, 0, 1).float()

            return img_stack, mask_stack

        else:
            # ---------------------------
            # 4. Test Mode (No Masks)
            # ---------------------------
            if self.transforms:
                augmented = self.transforms(image=img_stack)
                img_stack = augmented["image"]

            return img_stack, row["id"]


def process_metadata(csv_path, cache_name, load_cached_data=True):
    """
    Loads metadata CSV, pivots it to wide format (one row per image),
    and creates a file path lookup dictionary.
    Handles caching to parquet.
    """
    cache_path = os.path.join(Config.CACHE_DIR, cache_name)

    # 1. Try Load Cache
    if load_cached_data and os.path.exists(cache_path):
        df = pd.read_parquet(cache_path)
    else:
        # 2. Process from Scratch
        df_raw = pd.read_csv(csv_path)

        # Fill NaNs in segmentation to allow pivoting
        df_raw["segmentation"] = df_raw["segmentation"].fillna("")

        # Define index columns that identify a unique image
        index_cols = [
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

        # Pivot table: Convert 'class' column values to separate columns
        # This consolidates the 3 rows per image into 1 row
        df = df_raw.pivot_table(
            index=index_cols, columns="class", values="segmentation", aggfunc="first"
        ).reset_index()

        # Clean up columns
        df.columns.name = None

        # Ensure all class columns exist (even if missing in data subset)
        for cls in ["large_bowel", "small_bowel", "stomach"]:
            if cls not in df.columns:
                df[cls] = ""

        # Save to cache
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        df.to_parquet(cache_path)

    # 3. Create Path Lookup Dictionary
    # Map (case, day, slice) -> file_path for O(1) neighbor retrieval
    path_lookup = dict(zip(zip(df["case"], df["day"], df["slice"]), df["file_path"]))

    return df, path_lookup


def get_transforms(data="train"):
    """
    Returns Albumentations transforms for training, validation, or testing.
    """
    if data == "train":
        return A.Compose(
            [
                A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                # Normalize to ImageNet statistics (compatible with EfficientNet)
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    elif data == "valid" or data == "test":
        return A.Compose(
            [
                A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


def get_loaders(load_cached_data=True):
    """
    Prepares DataLoaders for training and validation.
    Performs balanced sampling to handle class imbalance.
    """
    set_seed(Config.SEED)

    # 1. Load and Process Metadata
    train_df, train_lookup = process_metadata(
        Config.TRAIN_METADATA_PATH, "train_processed.parquet", load_cached_data
    )

    val_df, val_lookup = process_metadata(
        Config.VAL_METADATA_PATH, "val_processed.parquet", load_cached_data
    )

    # 2. Balanced Sampling for Training
    # Identify positive samples (at least one organ mask is non-empty)
    train_df["has_mask"] = (
        (train_df["large_bowel"] != "")
        | (train_df["small_bowel"] != "")
        | (train_df["stomach"] != "")
    )

    df_pos = train_df[train_df["has_mask"]].copy()
    df_neg = train_df[~train_df["has_mask"]].copy()

    # Sample negatives: Keep negatives equal to 50% of positives count
    # This reduces the overwhelming number of background slices
    n_neg = int(len(df_pos) * 0.5)
    if len(df_neg) > n_neg:
        df_neg = df_neg.sample(n=n_neg, random_state=Config.SEED)

    train_df_balanced = (
        pd.concat([df_pos, df_neg])
        .sample(frac=1, random_state=Config.SEED)
        .reset_index(drop=True)
    )

    print(
        f"Train Data: {len(train_df)} total -> {len(train_df_balanced)} balanced ({len(df_pos)} pos, {len(df_neg)} neg)"
    )
    print(f"Val Data: {len(val_df)} images")

    # 3. Create Datasets
    train_dataset = MRIDataset(
        train_df_balanced,
        train_lookup,
        transforms=get_transforms("train"),
        mode="train",
    )

    val_dataset = MRIDataset(
        val_df, val_lookup, transforms=get_transforms("valid"), mode="valid"
    )

    # 4. Create DataLoaders
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


def get_test_loader(load_cached_data=True):
    """
    Prepares DataLoader for testing/inference.
    Handles the specific structure of test metadata which lacks segmentation columns.
    """
    cache_path = os.path.join(Config.CACHE_DIR, "test_processed.parquet")

    if load_cached_data and os.path.exists(cache_path):
        df = pd.read_parquet(cache_path)
    else:
        # Load raw test metadata
        df_raw = pd.read_csv(Config.TEST_METADATA_PATH)

        # Extract unique images only (dropping class/prediction columns)
        cols = ["id", "case", "day", "slice", "file_path", "img_width", "img_height"]
        df = df_raw[cols].drop_duplicates().reset_index(drop=True)

        # Cache
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        df.to_parquet(cache_path)

    # Create path lookup
    path_lookup = dict(zip(zip(df["case"], df["day"], df["slice"]), df["file_path"]))

    dataset = MRIDataset(
        df, path_lookup, transforms=get_transforms("test"), mode="test"
    )

    loader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return loader
