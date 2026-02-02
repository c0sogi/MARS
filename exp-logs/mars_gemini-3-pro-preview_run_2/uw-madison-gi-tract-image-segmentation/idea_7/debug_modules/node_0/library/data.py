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


def load_img(path):
    """
    Loads an image from a path, ensuring it is read as 16-bit and converted to float32.
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Image not found at {path}")
    img = img.astype("float32")
    return img


class UWMDataset(Dataset):
    def __init__(self, df, lookup_dict, transforms=None, mode="train"):
        self.df = df.reset_index(drop=True)
        self.lookup_dict = lookup_dict
        self.transforms = transforms
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        case = row["case"]
        day = row["day"]
        slice_id = row["slice"]

        # 2.5D Input Generation: Stack [slice-1, slice, slice+1]
        imgs = []
        for offset in [-1, 0, 1]:
            target_slice = slice_id + offset
            key = (case, day, target_slice)

            # Boundary handling: Replicate current slice if neighbor is missing
            if key not in self.lookup_dict:
                key = (case, day, slice_id)

            path = os.path.join(Config.INPUT_DIR, self.lookup_dict[key])
            img = load_img(path)
            imgs.append(img)

        # Stack to (H, W, 3)
        img_stack = np.stack(imgs, axis=-1)

        # Min-Max Normalization per stack
        _min = img_stack.min()
        _max = img_stack.max()
        if _max > _min:
            img_stack = (img_stack - _min) / (_max - _min)
        else:
            img_stack = np.zeros_like(img_stack)

        if self.mode != "test":
            h, w = int(row["img_height"]), int(row["img_width"])
            masks = []
            for cls in Config.CLASSES:
                rle = row[cls]
                mask = rle_decode(rle, (h, w))
                masks.append(mask)
            mask_stack = np.stack(masks, axis=-1)  # (H, W, 3)

            # Apply Augmentations
            if self.transforms:
                augmented = self.transforms(image=img_stack, mask=mask_stack)
                img_stack = augmented["image"]
                mask_stack = augmented["mask"]

            # Ensure mask is channel-first float tensor if not handled by ToTensorV2
            if not isinstance(mask_stack, torch.Tensor):
                mask_stack = torch.from_numpy(mask_stack.transpose(2, 0, 1)).float()
            else:
                # ToTensorV2 converts to tensor and HWC->CHW, but keeps type.
                # We need float for BCE loss.
                mask_stack = mask_stack.float()

            return img_stack, mask_stack

        else:
            # Test mode
            if self.transforms:
                augmented = self.transforms(image=img_stack)
                img_stack = augmented["image"]

            return img_stack, str(row["id"])


def process_dataframe(
    csv_path, mode="train", cache_name="train_cache.parquet", load_cached_data=True
):
    """
    Loads metadata, pivots to wide format (one row per slice), performs subsampling, and caches result.
    """
    cache_path = os.path.join(Config.WORKING_DIR, cache_name)

    if load_cached_data and os.path.exists(cache_path):
        return pd.read_parquet(cache_path)

    df = pd.read_csv(csv_path)

    # Static columns that define a slice
    static_cols = ["id", "case", "day", "slice", "file_path", "img_height", "img_width"]

    # Get unique slices
    base_df = df[static_cols].drop_duplicates(subset=["id"])

    # Pivot segmentation columns
    # Result: id as index, columns are 'large_bowel', 'small_bowel', 'stomach'
    pivot_df = df.pivot(index="id", columns="class", values="segmentation")

    # Merge back to get file info with mask info in one row
    df_grouped = base_df.merge(pivot_df, on="id", how="left")

    # Fill NaNs
    for cls in Config.CLASSES:
        if cls not in df_grouped.columns:
            df_grouped[cls] = ""
        df_grouped[cls] = df_grouped[cls].fillna("")

    # Stratified Sampling for Training
    if mode == "train":
        # Identify slices with at least one mask
        has_mask = (df_grouped[Config.CLASSES] != "").any(axis=1)

        positives = df_grouped[has_mask]
        negatives = df_grouped[~has_mask]

        # Keep all positives, subsample 50% of negatives
        negatives_sampled = negatives.sample(frac=0.5, random_state=Config.SEED)

        df_final = (
            pd.concat([positives, negatives_sampled])
            .sample(frac=1.0, random_state=Config.SEED)
            .reset_index(drop=True)
        )
    else:
        df_final = df_grouped.reset_index(drop=True)

    # Cache the result
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df_final.to_parquet(cache_path)

    return df_final


def get_dataloaders(load_cached_data=True):
    """
    Creates training and validation DataLoaders.
    """
    # 1. Process DataFrames
    train_df = process_dataframe(
        Config.TRAIN_METADATA_PATH,
        mode="train",
        cache_name="train_processed.parquet",
        load_cached_data=load_cached_data,
    )

    val_df = process_dataframe(
        Config.VAL_METADATA_PATH,
        mode="val",
        cache_name="val_processed.parquet",
        load_cached_data=load_cached_data,
    )

    # 2. Build Global Lookup Dictionary
    # We need paths for ALL slices to find neighbors (2.5D), even those dropped during subsampling.
    full_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
    full_val = pd.read_csv(Config.VAL_METADATA_PATH)
    combined_meta = pd.concat([full_train, full_val])[
        ["case", "day", "slice", "file_path"]
    ].drop_duplicates()

    lookup_dict = {}
    for _, row in combined_meta.iterrows():
        lookup_dict[(row["case"], row["day"], row["slice"])] = row["file_path"]

    # 3. Define Transforms
    train_transforms = A.Compose(
        [
            A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            ToTensorV2(),
        ]
    )

    val_transforms = A.Compose(
        [A.Resize(Config.IMG_SIZE, Config.IMG_SIZE), ToTensorV2()]
    )

    # 4. Create Datasets
    train_dataset = UWMDataset(
        train_df, lookup_dict, transforms=train_transforms, mode="train"
    )
    val_dataset = UWMDataset(val_df, lookup_dict, transforms=val_transforms, mode="val")

    # Debug Subsetting
    if Config.DEBUG:
        train_dataset = torch.utils.data.Subset(
            train_dataset, range(min(len(train_dataset), Config.DEBUG_SAMPLE_SIZE))
        )
        val_dataset = torch.utils.data.Subset(
            val_dataset, range(min(len(val_dataset), Config.DEBUG_SAMPLE_SIZE))
        )

    # 5. Create Loaders
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


def get_test_dataloader():
    """
    Creates the test DataLoader.
    """
    df = pd.read_csv(Config.TEST_METADATA_PATH)

    # Group test dataframe to one row per slice
    static_cols = ["id", "case", "day", "slice", "file_path", "img_height", "img_width"]
    df_grouped = df[static_cols].drop_duplicates(subset=["id"]).reset_index(drop=True)

    # Build lookup dict for test set
    lookup_dict = {}
    for _, row in df_grouped.iterrows():
        lookup_dict[(row["case"], row["day"], row["slice"])] = row["file_path"]

    transforms = A.Compose([A.Resize(Config.IMG_SIZE, Config.IMG_SIZE), ToTensorV2()])

    dataset = UWMDataset(df_grouped, lookup_dict, transforms=transforms, mode="test")

    loader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return loader
