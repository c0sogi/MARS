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


class BuildDataset(Dataset):
    def __init__(self, df, label=True, transforms=None):
        """
        Args:
            df (pd.DataFrame): Dataframe containing metadata.
            label (bool): Whether to return masks (True for train/val, False for test).
            transforms (albumentations.Compose): Transforms to apply.
        """
        self.df = df
        self.label = label
        self.transforms = transforms

        # Pre-extract file paths and depth info to lists for faster indexing
        self.file_paths = df["file_path"].tolist()
        self.ids = df["id"].tolist()

        if self.label:
            self.rle_large = df["large_bowel"].tolist()
            self.rle_small = df["small_bowel"].tolist()
            self.rle_stomach = df["stomach"].tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # 1. Load Image
        img_path = os.path.join(Config.INPUT_DIR, self.file_paths[index])
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        # Handle potential read errors (though metadata check passed)
        if img is None:
            # Create a dummy black image of correct original size if read fails
            # This is a fallback to prevent crashing; ideally shouldn't happen
            w, h = self.df.iloc[index]["width"], self.df.iloc[index]["height"]
            img = np.zeros((h, w), dtype=np.uint16)

        # 2. Robust Normalization
        img = img.astype(np.float32)
        min_val = np.percentile(img, Config.NORM_MIN_PERCENTILE)
        max_val = np.percentile(img, Config.NORM_MAX_PERCENTILE)

        # Clip and Scale to [0, 1]
        img = np.clip(img, min_val, max_val)
        img = (img - min_val) / (max_val - min_val + 1e-6)

        # 3. Construct 3-Channel Input
        # Channels 1-3: Replicated Image
        img_rgb = np.stack([img, img, img], axis=-1)  # (H, W, 3)

        # 4. Handle Masks (if label=True)
        if self.label:
            msk_large = rle_decode(self.rle_large[index], (h, w))
            msk_small = rle_decode(self.rle_small[index], (h, w))
            msk_stomach = rle_decode(self.rle_stomach[index], (h, w))

            # Stack -> (H, W, 3)
            mask = np.stack([msk_large, msk_small, msk_stomach], axis=-1)
            mask = mask.astype(np.float32)

            # 5. Augmentations
            if self.transforms:
                data = self.transforms(image=img_rgb, mask=mask)
                img_rgb = data["image"]
                mask = data["mask"]

            # Ensure channel first (C, H, W)
            if not isinstance(img_rgb, torch.Tensor):
                img_rgb = np.transpose(img_rgb, (2, 0, 1))
                img_rgb = torch.from_numpy(img_rgb)

            if not isinstance(mask, torch.Tensor):
                mask = np.transpose(mask, (2, 0, 1))
                mask = torch.from_numpy(mask)

            return img_rgb, mask

        else:
            # Test mode
            if self.transforms:
                data = self.transforms(image=img_rgb)
                img_rgb = data["image"]

            if not isinstance(img_rgb, torch.Tensor):
                img_rgb = np.transpose(img_rgb, (2, 0, 1))
                img_rgb = torch.from_numpy(img_rgb)

            return img_rgb, self.ids[index]


def _process_dataframe(df_path, cache_name, load_cached_data=True):
    """
    Loads metadata CSV, calculates relative slice depth, and caches the result.
    """
    cache_path = os.path.join(Config.CACHE_DIR, cache_name)

    # 1. Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            print(f"Failed to load cache {cache_path}: {e}. Recomputing.")

    # 2. Process from scratch
    df = pd.read_csv(df_path, keep_default_na=False)

    # Ensure slice is int
    df["slice_idx"] = df["slice"].astype(int)

    # Calculate max slice per case+day group
    # We group by ['case', 'day'] and transform max
    # Note: 'case' and 'day' columns exist in metadata
    max_slices = df.groupby(["case", "day"])["slice_idx"].transform("max")

    # Calculate relative depth (avoid division by zero if max is 0, though unlikely)
    df["relative_depth"] = df["slice_idx"] / (max_slices + 1e-6)

    # 3. Save to cache
    try:
        df.to_parquet(cache_path, index=False)
    except Exception as e:
        print(f"Warning: Could not save cache to {cache_path}: {e}")

    return df


def get_transforms(data):
    """
    Returns Albumentations transforms for 'train' or 'valid'.
    """
    if data == "train":
        return A.Compose(
            [
                A.Resize(*Config.IMAGE_SIZE, interpolation=cv2.INTER_LINEAR),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.1,
                    rotate_limit=15,
                    p=0.5,
                    border_mode=cv2.BORDER_REFLECT_101,
                ),
                A.OneOf(
                    [
                        A.GridDistortion(
                            num_steps=5,
                            distort_limit=0.05,
                            p=1.0,
                            border_mode=cv2.BORDER_REFLECT_101,
                        ),
                        A.ElasticTransform(
                            alpha=1,
                            sigma=50,
                            alpha_affine=50,
                            p=1.0,
                            border_mode=cv2.BORDER_REFLECT_101,
                        ),
                    ],
                    p=0.25,
                ),
                # Note: We do not use A.Normalize here because we did custom normalization
                # and we have 4 channels.
                # ToTensorV2 converts to tensor and HWC->CHW
                ToTensorV2(transpose_mask=True),
            ]
        )

    elif data == "valid" or data == "test":
        return A.Compose(
            [
                A.Resize(*Config.IMAGE_SIZE, interpolation=cv2.INTER_LINEAR),
                ToTensorV2(transpose_mask=True),
            ]
        )


def prepare_loaders(load_cached_data=True, debug=False):
    """
    Prepares DataLoaders for training and validation.

    Args:
        load_cached_data (bool): Whether to use cached dataframes.
        debug (bool): If True, subsets data for debugging.

    Returns:
        train_loader, val_loader
    """
    # Load and process dataframes
    train_df = _process_dataframe(
        Config.TRAIN_CSV, "train_processed.parquet", load_cached_data
    )
    val_df = _process_dataframe(
        Config.VAL_CSV, "val_processed.parquet", load_cached_data
    )

    # Debug mode: subset
    if debug:
        train_df = train_df.head(Config.DEBUG_SAMPLE_SIZE).reset_index(drop=True)
        val_df = val_df.head(Config.DEBUG_SAMPLE_SIZE).reset_index(drop=True)

    # Create Datasets
    train_dataset = BuildDataset(
        train_df, label=True, transforms=get_transforms("train")
    )

    val_dataset = BuildDataset(val_df, label=True, transforms=get_transforms("valid"))

    # Create Loaders
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


def prepare_test_loader(load_cached_data=True):
    """
    Prepares DataLoader for testing/inference.
    """
    test_df = _process_dataframe(
        Config.TEST_CSV, "test_processed.parquet", load_cached_data
    )

    test_dataset = BuildDataset(test_df, label=False, transforms=get_transforms("test"))

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return test_loader
