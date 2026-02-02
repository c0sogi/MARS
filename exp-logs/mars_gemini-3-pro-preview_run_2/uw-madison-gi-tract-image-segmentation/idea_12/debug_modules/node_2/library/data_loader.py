import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import load_image, rle_decode, set_seed

# Set seed for reproducibility
set_seed(Config.SEED)


def preprocess_metadata(csv_path, load_cached_data=True):
    """
    Loads metadata, pivots it to wide format (one row per slice), and caches the result.
    """
    filename = os.path.basename(csv_path).replace(".csv", "_pivoted.parquet")
    cache_path = os.path.join(Config.WORKING_DIR, filename)

    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            print(f"Failed to load cache {cache_path}: {e}. Recomputing...")

    df = pd.read_csv(csv_path)

    # Check if we need to pivot (train/val have 'class' and 'segmentation' columns)
    if "class" in df.columns and "segmentation" in df.columns:
        # Pivot to wide format: one row per id, columns for each class segmentation
        # We also need to keep the metadata columns.
        # Strategy: Pivot segmentation, then join back with metadata (taking first)

        # 1. Pivot segmentation
        pivot_df = df.pivot(
            index="id", columns="class", values="segmentation"
        ).reset_index()

        # 2. Get metadata (drop duplicates since they are repeated for each class)
        meta_cols = [
            c for c in df.columns if c not in ["class", "segmentation", "predicted"]
        ]
        meta_df = df[meta_cols].drop_duplicates(subset=["id"])

        # 3. Merge
        df_wide = pd.merge(meta_df, pivot_df, on="id", how="left")

        # Fill NaN RLEs with empty strings
        for cls in Config.CLASS_LABELS:
            if cls in df_wide.columns:
                df_wide[cls] = df_wide[cls].fillna("")
            else:
                df_wide[cls] = ""

        df = df_wide

    # Save to cache
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    df.to_parquet(cache_path, index=False)

    return df


class Base25DDataset(Dataset):
    def __init__(self, df, transforms=None):
        self.df = df
        self.transforms = transforms
        self.ids = self.df["id"].values

        # Create a lookup for file paths: (case, day, slice) -> file_path
        # This is necessary for finding neighbors efficiently
        self.path_map = {}
        for _, row in self.df.iterrows():
            self.path_map[(row["case"], row["day"], row["slice"])] = row["file_path"]

    def __len__(self):
        return len(self.df)

    def load_25d_stack(self, case, day, slice_idx):
        """
        Loads a 2.5D stack of images (slice_idx-1, slice_idx, slice_idx+1).
        Handles boundary conditions by replicating the nearest valid slice.
        Normalizes to [0, 1].
        """
        slices = []
        for offset in [-1, 0, 1]:
            target_slice = slice_idx + offset
            key = (case, day, target_slice)

            # If neighbor doesn't exist (boundary), use the current slice
            if key not in self.path_map:
                key = (case, day, slice_idx)

            path = os.path.join(Config.INPUT_DIR, self.path_map[key])
            img = load_image(path)  # Returns (H, W, 1) or (H, W)

            if img.ndim == 3:
                img = img[:, :, 0]  # Drop channel dim if exists

            slices.append(img)

        # Stack along channel dimension -> (H, W, 3)
        stack = np.stack(slices, axis=-1).astype(np.float32)

        # Normalize Min-Max to [0, 1]
        mx = np.max(stack)
        mn = np.min(stack)
        if mx - mn > 0:
            stack = (stack - mn) / (mx - mn)
        else:
            stack = stack - mn  # Zero out if flat

        return stack

    def load_masks(self, row, shape):
        """
        Loads masks for all classes and stacks them.
        Returns (H, W, 3)
        """
        masks = []
        for cls in Config.CLASS_LABELS:
            rle = row[cls]
            mask = rle_decode(rle, shape)
            masks.append(mask)
        return np.stack(masks, axis=-1).astype(np.float32)


class CoarseDataset(Base25DDataset):
    """
    Dataset for Stage 1: Global Localization.
    Resizes full images to COARSE_IMG_SIZE.
    """

    def __init__(self, df, transforms=None):
        super().__init__(df, transforms)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load Image Stack
        image = self.load_25d_stack(row["case"], row["day"], row["slice"])
        h, w = image.shape[:2]

        # Load Masks
        mask = self.load_masks(row, (h, w))

        # Apply Transforms
        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # Ensure channel first for PyTorch (done by ToTensorV2 in transforms, usually)
        # If ToTensorV2 is not in transforms, do it manually
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image.transpose(2, 0, 1))
            mask = torch.from_numpy(mask.transpose(2, 0, 1))

        return image, mask


class FineDataset(Base25DDataset):
    """
    Dataset for Stage 2: Fine Segmentation.
    Crops ROI around Ground Truth masks with jitter and resizes to FINE_IMG_SIZE.
    """

    def __init__(self, df, mode="train", transforms=None):
        # Filter: For training/validation of Fine model, we only care about slices
        # that actually have relevant anatomy to refine.
        # We check if any of the class columns have a non-empty string.

        # Create a mask of valid rows
        has_mask = pd.Series([False] * len(df), index=df.index)
        for cls in Config.CLASS_LABELS:
            has_mask |= df[cls].notna() & (df[cls] != "")

        # Filter dataframe
        df_filtered = df[has_mask].copy().reset_index(drop=True)

        super().__init__(df_filtered, transforms)
        self.mode = mode

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load Image Stack
        image = self.load_25d_stack(row["case"], row["day"], row["slice"])
        h, w = image.shape[:2]

        # Load Masks
        mask = self.load_masks(row, (h, w))

        # 1. Calculate Bounding Box of the union of masks
        mask_union = np.max(mask, axis=-1)  # (H, W)
        rows, cols = np.where(mask_union > 0)

        if len(rows) > 0:
            y_min, y_max = np.min(rows), np.max(rows)
            x_min, x_max = np.min(cols), np.max(cols)
        else:
            # Fallback if mask is empty (shouldn't happen due to filtering, but for safety)
            y_min, y_max = h // 4, 3 * h // 4
            x_min, x_max = w // 4, 3 * w // 4

        # 2. Add Margin
        box_h = y_max - y_min
        box_w = x_max - x_min

        # Use max dim to keep aspect ratio roughly square-ish or just expand
        margin = max(box_h, box_w) * Config.ROI_MARGIN_RATIO

        y_min = max(0, int(y_min - margin))
        y_max = min(h, int(y_max + margin))
        x_min = max(0, int(x_min - margin))
        x_max = min(w, int(x_max + margin))

        # 3. Add Jitter (only during training)
        if self.mode == "train":
            # Randomly shift the box slightly, ensuring it stays within bounds
            jitter_range = int(margin * 0.5)
            if jitter_range > 0:
                y_shift = np.random.randint(-jitter_range, jitter_range + 1)
                x_shift = np.random.randint(-jitter_range, jitter_range + 1)

                # Apply shift and clamp
                new_y_min = max(0, min(h - (y_max - y_min), y_min + y_shift))
                new_x_min = max(0, min(w - (x_max - x_min), x_min + x_shift))

                y_max = new_y_min + (y_max - y_min)
                x_max = new_x_min + (x_max - x_min)
                y_min, x_min = new_y_min, new_x_min

        # 4. Crop
        crop_img = image[y_min:y_max, x_min:x_max, :]
        crop_mask = mask[y_min:y_max, x_min:x_max, :]

        # 5. Apply Transforms (Resize to FINE_IMG_SIZE)
        if self.transforms:
            augmented = self.transforms(image=crop_img, mask=crop_mask)
            crop_img = augmented["image"]
            crop_mask = augmented["mask"]

        if not isinstance(crop_img, torch.Tensor):
            crop_img = torch.from_numpy(crop_img.transpose(2, 0, 1))
            crop_mask = torch.from_numpy(crop_mask.transpose(2, 0, 1))

        return crop_img, crop_mask


def get_transforms(stage, mode="train"):
    """
    Returns Albumentations transforms for the specific stage and mode.
    """
    if stage == "coarse":
        target_size = Config.COARSE_IMG_SIZE
        if mode == "train":
            return A.Compose(
                [
                    A.Resize(target_size[0], target_size[1]),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.Rotate(limit=15, p=0.5),
                    ToTensorV2(transpose_mask=True),
                ]
            )
        else:
            return A.Compose(
                [
                    A.Resize(target_size[0], target_size[1]),
                    ToTensorV2(transpose_mask=True),
                ]
            )

    elif stage == "fine":
        target_size = Config.FINE_IMG_SIZE
        if mode == "train":
            return A.Compose(
                [
                    A.Resize(target_size[0], target_size[1]),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.Rotate(limit=30, p=0.5),
                    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.25),
                    ToTensorV2(transpose_mask=True),
                ]
            )
        else:
            return A.Compose(
                [
                    A.Resize(target_size[0], target_size[1]),
                    ToTensorV2(transpose_mask=True),
                ]
            )
    else:
        raise ValueError(f"Unknown stage: {stage}")


def get_dataloaders(stage, batch_size=None, debug=Config.DEBUG):
    """
    Factory function to create DataLoaders for Train and Validation.

    Args:
        stage (str): 'coarse' or 'fine'.
        batch_size (int): Batch size. If None, uses Config default.
        debug (bool): If True, subsamples data for quick testing.
    """
    # 1. Load Metadata
    train_df = preprocess_metadata(Config.TRAIN_META_PATH)
    val_df = preprocess_metadata(Config.VAL_META_PATH)

    # Debug Subsampling
    if debug:
        train_df = train_df.iloc[: Config.DEBUG_SAMPLE_SIZE]
        val_df = val_df.iloc[: Config.DEBUG_SAMPLE_SIZE]

    # 2. Select Dataset Class and Batch Size
    if stage == "coarse":
        DatasetClass = CoarseDataset
        bs = batch_size if batch_size else Config.COARSE_BATCH_SIZE
    elif stage == "fine":
        DatasetClass = FineDataset
        bs = batch_size if batch_size else Config.FINE_BATCH_SIZE
    else:
        raise ValueError("Stage must be 'coarse' or 'fine'")

    # 3. Create Datasets
    train_ds = DatasetClass(train_df, transforms=get_transforms(stage, "train"))
    # Note: FineDataset validation also needs mode='val' (or 'train' without jitter)
    # We pass mode='val' to disable jitter in FineDataset
    val_ds = DatasetClass(val_df, transforms=get_transforms(stage, "val"))

    if isinstance(train_ds, FineDataset):
        train_ds.mode = "train"
        val_ds.mode = "val"

    # 4. Create Loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader
