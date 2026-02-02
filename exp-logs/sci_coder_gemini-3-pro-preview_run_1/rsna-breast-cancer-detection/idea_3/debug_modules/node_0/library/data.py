import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

from library import config, utils


def process_metadata(load_cached_data=True):
    """
    Loads and processes metadata CSVs.
    Implements caching using Parquet to store processed dataframes with filled missing values.
    """
    cache_dir = config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    train_cache = os.path.join(cache_dir, "processed_train.parquet")
    val_cache = os.path.join(cache_dir, "processed_val.parquet")
    test_cache = os.path.join(cache_dir, "processed_test.parquet")

    # Check if cached files exist and loading is requested
    if (
        load_cached_data
        and os.path.exists(train_cache)
        and os.path.exists(val_cache)
        and os.path.exists(test_cache)
    ):
        print(f"Loading cached metadata from {cache_dir}...")
        df_train = pd.read_parquet(train_cache)
        df_val = pd.read_parquet(val_cache)
        df_test = pd.read_parquet(test_cache)
    else:
        print("Processing metadata from scratch...")
        df_train = pd.read_csv(config.TRAIN_METADATA_PATH)
        df_val = pd.read_csv(config.VAL_METADATA_PATH)
        df_test = pd.read_csv(config.TEST_METADATA_PATH)

        # Data Cleaning
        # 1. Age: Fill NaN with mean from Train set
        # Note: We compute mean from train only to avoid leakage
        age_mean = df_train["age"].mean()
        df_train["age"] = df_train["age"].fillna(age_mean)
        df_val["age"] = df_val["age"].fillna(age_mean)
        df_test["age"] = df_test["age"].fillna(age_mean)

        # 2. Implant: Fill NaN with 0, ensure numeric
        for df in [df_train, df_val, df_test]:
            if "implant" in df.columns:
                df["implant"] = df["implant"].fillna(0).astype(int)
            else:
                # Test set might not have this column if strictly hidden,
                # but per description it is available.
                df["implant"] = 0

        # Save to cache
        df_train.to_parquet(train_cache)
        df_val.to_parquet(val_cache)
        df_test.to_parquet(test_cache)
        print(f"Metadata saved to {cache_dir}")

    return df_train, df_val, df_test


class BreastCancerDataset(Dataset):
    def __init__(self, df, transforms=None, age_mean=0.0, age_std=1.0, mode="train"):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata.
            transforms (albumentations.Compose): Augmentations.
            age_mean (float): Mean age from training set for normalization.
            age_std (float): Std age from training set for normalization.
            mode (str): 'train', 'val', or 'test'.
        """
        self.df = df
        self.transforms = transforms
        self.age_mean = age_mean
        self.age_std = age_std
        self.mode = mode

        # Pre-extract columns to avoid dataframe overhead in __getitem__
        self.file_paths = df["file_path"].values
        self.ages = df["age"].values
        self.implants = df["implant"].values

        if self.mode != "test":
            self.labels = df["cancer"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = os.path.join(config.INPUT_DIR, self.file_paths[idx])

        # Load Image
        img = utils.load_dicom(path)

        # Handle load failure (robustness)
        if img is None:
            # Create a blank black image of correct size
            img = np.zeros((config.IMG_HEIGHT, config.IMG_WIDTH), dtype=np.uint8)
        else:
            # Crop ROI
            img = utils.crop_roi(img)

            # Resize
            # Using cv2 for speed.
            try:
                img = cv2.resize(
                    img,
                    (config.IMG_WIDTH, config.IMG_HEIGHT),
                    interpolation=cv2.INTER_LINEAR,
                )
            except Exception:
                img = np.zeros((config.IMG_HEIGHT, config.IMG_WIDTH), dtype=np.uint8)

        # Normalize to 0-1 float
        if img.max() > 0:
            img = img.astype(np.float32) / img.max()
        else:
            img = img.astype(np.float32)

        # Apply Transforms
        if self.transforms:
            # Albumentations expects image key
            augmented = self.transforms(image=img)
            img = augmented["image"]
        else:
            # Manual to tensor if no transforms
            img = torch.from_numpy(img).unsqueeze(0)  # (1, H, W)

        # Ensure correct tensor shape (1, H, W)
        # ToTensorV2 usually returns (C, H, W) if input is (H, W, C)
        # If input is (H, W), it returns (H, W). We need (1, H, W).
        if isinstance(img, torch.Tensor):
            if img.ndim == 2:
                img = img.unsqueeze(0)
            elif img.ndim == 3 and img.shape[2] == 1:
                img = img.permute(2, 0, 1)
        else:
            # Fallback if transforms didn't return tensor
            img = torch.from_numpy(img)
            if img.ndim == 2:
                img = img.unsqueeze(0)

        # Metadata Processing
        age = self.ages[idx]
        implant = self.implants[idx]

        # Normalize Age (Standard Scaling)
        if self.age_std > 0:
            age_norm = (age - self.age_mean) / self.age_std
        else:
            age_norm = 0.0

        # Convert to tensors
        age_tensor = torch.tensor(age_norm, dtype=torch.float32)
        implant_tensor = torch.tensor(implant, dtype=torch.float32)

        if self.mode != "test":
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return img, age_tensor, implant_tensor, label
        else:
            return img, age_tensor, implant_tensor


def get_transforms(mode="train"):
    """
    Returns the Albumentations transform pipeline.
    """
    if mode == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1,
                    rotate_limit=20,
                    p=0.5,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                ),
                # Resize ensures we are definitely at target size even after rotation/crop anomalies
                A.Resize(config.IMG_HEIGHT, config.IMG_WIDTH),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(config.IMG_HEIGHT, config.IMG_WIDTH),
                ToTensorV2(),
            ]
        )


def get_dataloaders(load_cached_data=True):
    """
    Creates and returns DataLoaders for train, val, and test sets.
    """
    # Load Metadata
    df_train, df_val, df_test = process_metadata(load_cached_data=load_cached_data)

    # Calculate Age Stats from Train
    age_mean = df_train["age"].mean()
    age_std = df_train["age"].std()

    # Transforms
    train_transforms = get_transforms("train")
    val_transforms = get_transforms("val")

    # Datasets
    train_dataset = BreastCancerDataset(
        df_train,
        transforms=train_transforms,
        age_mean=age_mean,
        age_std=age_std,
        mode="train",
    )

    val_dataset = BreastCancerDataset(
        df_val,
        transforms=val_transforms,
        age_mean=age_mean,
        age_std=age_std,
        mode="val",
    )

    test_dataset = BreastCancerDataset(
        df_test,
        transforms=val_transforms,
        age_mean=age_mean,
        age_std=age_std,
        mode="test",
    )

    # Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
