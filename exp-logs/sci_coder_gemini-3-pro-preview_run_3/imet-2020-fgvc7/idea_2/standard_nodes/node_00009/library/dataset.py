import os
import cv2
import torch
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from library.config import Config


class ArtworkDataset(Dataset):
    """
    Dataset class for Artwork Attribute Classification.
    Handles image loading and multi-hot label encoding.
    """

    def __init__(self, df, transforms=None, mode="train"):
        """
        Args:
            df (pd.DataFrame): DataFrame containing 'file_path' and 'attribute_ids'.
            transforms (albumentations.Compose): Transformations to apply.
            mode (str): 'train', 'val', or 'test'.
        """
        self.df = df
        self.transforms = transforms
        self.mode = mode
        self.file_paths = df["file_path"].values

        # Pre-process labels for train/val to avoid parsing strings in __getitem__
        # We keep them as a list of lists for efficiency
        if self.mode != "test":
            self.labels = []
            for attr_str in df["attribute_ids"].astype(str).values:
                if attr_str == "" or attr_str.lower() == "nan":
                    self.labels.append([])
                else:
                    self.labels.append([int(x) for x in attr_str.split()])
        else:
            self.ids = df["id"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Construct full image path
        img_path = os.path.join(Config.INPUT_DIR, self.file_paths[idx])

        # Load image
        image = cv2.imread(img_path)
        if image is None:
            # Fallback for missing images (though metadata check passed)
            # Create a black image of expected size
            image = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transformations
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]
        else:
            # Basic to tensor if no transforms provided (fallback)
            image = ToTensorV2()(image=image)["image"]

        if self.mode == "test":
            # Return image and ID for submission generation
            return image, self.ids[idx]
        else:
            # Create multi-hot target vector
            target = torch.zeros(Config.NUM_CLASSES, dtype=torch.float32)
            indices = self.labels[idx]
            if indices:
                target[indices] = 1.0

            return image, target


def get_transforms(mode="train"):
    """
    Returns the transformation pipeline for the specified mode.

    Args:
        mode (str): 'train' or 'val'/'test'.

    Returns:
        albumentations.Compose: The transformation pipeline.
    """
    if mode == "train":
        return A.Compose(
            [
                A.Resize(height=Config.IMAGE_SIZE, width=Config.IMAGE_SIZE),
                A.HorizontalFlip(p=0.5),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(height=Config.IMAGE_SIZE, width=Config.IMAGE_SIZE),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ]
        )


def load_and_process_metadata(csv_path, cache_name, load_cached_data=True):
    """
    Loads metadata from CSV, optionally using a parquet cache.

    Args:
        csv_path (str): Path to the source CSV file.
        cache_name (str): Name of the cache file (e.g., 'train_cache.parquet').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: The loaded dataframe.
    """
    cache_path = os.path.join(Config.WORKING_DIR, cache_name)

    # 1. Try to load cached data
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception:
            # If load fails, proceed to process from scratch
            pass

    # 2. Process data from scratch
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Source file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Ensure attribute_ids are strings (handle NaNs)
    if "attribute_ids" in df.columns:
        df["attribute_ids"] = df["attribute_ids"].fillna("").astype(str)

    # Save to cache
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    df.to_parquet(cache_path, index=False)

    return df


def get_dataloaders(
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    debug=Config.DEBUG,
    load_cached_data=True,
):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        batch_size (int): Batch size.
        num_workers (int): Number of worker processes.
        debug (bool): If True, uses a small subset of data.
        load_cached_data (bool): Whether to use cached metadata.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load DataFrames
    train_df = load_and_process_metadata(
        Config.TRAIN_CSV, "cached_train.parquet", load_cached_data
    )
    val_df = load_and_process_metadata(
        Config.VAL_CSV, "cached_val.parquet", load_cached_data
    )
    test_df = load_and_process_metadata(
        Config.TEST_CSV, "cached_test.parquet", load_cached_data
    )

    # Debug Mode: Sample subset
    if debug:
        train_df = train_df.head(batch_size * 2)
        val_df = val_df.head(batch_size)
        test_df = test_df.head(batch_size)

    # Create Datasets
    train_dataset = ArtworkDataset(
        train_df, transforms=get_transforms(mode="train"), mode="train"
    )
    val_dataset = ArtworkDataset(
        val_df, transforms=get_transforms(mode="val"), mode="val"
    )
    test_dataset = ArtworkDataset(
        test_df, transforms=get_transforms(mode="test"), mode="test"
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
