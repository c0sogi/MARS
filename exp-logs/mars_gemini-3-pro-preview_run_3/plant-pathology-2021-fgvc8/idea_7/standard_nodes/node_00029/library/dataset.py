import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from library.config import Config


def get_transforms(mode="train"):
    """
    Returns the Albumentations transform pipeline based on the mode.
    """
    if mode == "train":
        return A.Compose(
            [
                A.RandomResizedCrop(
                    size=(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
                    scale=Config.RANDOM_RESIZE_CROP_SCALE,
                ),
                A.HorizontalFlip(p=Config.FLIP_PROB),
                A.VerticalFlip(p=Config.FLIP_PROB),
                A.Normalize(),
                ToTensorV2(),
            ]
        )
    else:
        # Validation and Test
        return A.Compose(
            [
                A.Resize(height=Config.IMAGE_SIZE, width=Config.IMAGE_SIZE),
                A.Normalize(),
                ToTensorV2(),
            ]
        )


def process_metadata(csv_path, cache_name, load_cached_data=True):
    """
    Loads metadata CSV, processes labels into one-hot vectors, and caches the result.
    Strictly follows the caching logic requirement.
    """
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    cache_path = os.path.join(Config.CACHE_DIR, f"{cache_name}.parquet")

    # 1. IF load_cached_data is True: Try to load the file.
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception:
            # If loading fails, proceed to compute from scratch
            pass

    # 2. IF loading fails OR load_cached_data is False: Compute/process from scratch.
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Create mapping
    label_map = {lbl: i for i, lbl in enumerate(Config.LABELS)}

    def encode_labels(label_str):
        vec = np.zeros(Config.NUM_CLASSES, dtype=np.float32)
        if pd.isna(label_str) or label_str == "":
            return vec

        # Split by space
        labels = label_str.split()
        for lbl in labels:
            if lbl in label_map:
                vec[label_map[lbl]] = 1.0
        return vec

    # Apply encoding
    # We store as a list because parquet handles lists of primitives well
    df["target_vec"] = df["labels"].apply(lambda x: encode_labels(x).tolist())

    # Save to cache
    df.to_parquet(cache_path)

    # 3. Return the data.
    return df


class AppleDataset(Dataset):
    def __init__(self, df, transform=None, return_ids=False):
        self.df = df
        self.transform = transform
        self.return_ids = return_ids

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Image loading
        rel_path = row["file_path"]
        full_path = os.path.join(Config.INPUT_DIR, rel_path)

        image = cv2.imread(full_path)
        if image is None:
            # Safety fallback, though data checks passed
            image = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Retrieve target
        # Parquet stores lists, convert back to numpy tensor
        target = torch.tensor(row["target_vec"], dtype=torch.float32)

        if self.return_ids:
            return image, target, row["image"]
        else:
            return image, target


def get_loaders(load_cached_data=True):
    """
    Creates and returns training and validation DataLoaders.
    """
    # Train Data
    train_df = process_metadata(
        Config.TRAIN_METADATA, "train_processed", load_cached_data
    )
    train_dataset = AppleDataset(train_df, transform=get_transforms("train"))
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    # Validation Data
    val_df = process_metadata(Config.VAL_METADATA, "val_processed", load_cached_data)
    val_dataset = AppleDataset(val_df, transform=get_transforms("val"))
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
    Creates and returns the test DataLoader.
    """
    test_df = process_metadata(Config.TEST_METADATA, "test_processed", load_cached_data)
    test_dataset = AppleDataset(
        test_df, transform=get_transforms("test"), return_ids=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )
    return test_loader
