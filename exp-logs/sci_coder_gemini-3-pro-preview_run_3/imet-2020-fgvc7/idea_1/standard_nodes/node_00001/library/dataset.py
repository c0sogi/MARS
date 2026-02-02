import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from library.config import Config


def prepare_metadata(mode: str, load_cached_data: bool = True):
    """
    Loads and processes metadata, implementing a caching mechanism using Parquet.

    Args:
        mode (str): One of 'train', 'val', 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: Processed dataframe with 'full_path' and 'label_list'.
    """
    # Ensure working directory exists
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    cache_path = os.path.join(cache_dir, f"cached_{mode}.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            # Ensure label_list is read back correctly (parquet handles lists well)
            return df
        except Exception as e:
            print(f"Failed to load cache for {mode}: {e}. Recomputing...")

    # 2. Compute from scratch
    if mode == "train":
        csv_path = Config.TRAIN_CSV
    elif mode == "val":
        csv_path = Config.VAL_CSV
    elif mode == "test":
        csv_path = Config.TEST_CSV
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Resolve absolute file paths
    # Metadata contains relative paths like 'train/xxx.png'
    df["full_path"] = df["file_path"].apply(lambda x: os.path.join(Config.INPUT_DIR, x))

    # Parse attribute_ids into lists of integers
    def parse_labels(x):
        if pd.isna(x) or str(x).lower() == "nan" or str(x).strip() == "":
            return []
        return [int(i) for i in str(x).split()]

    # For test set, attribute_ids might be placeholders, but we parse them anyway
    # The Dataset class will handle creating the tensor
    df["label_list"] = df["attribute_ids"].apply(parse_labels)

    # 3. Save to cache
    try:
        df.to_parquet(cache_path, index=False)
    except Exception as e:
        print(f"Warning: Failed to save cache to {cache_path}: {e}")

    return df


def get_transforms(mode: str = "train"):
    """
    Returns the torchvision transforms for the given mode.

    Args:
        mode (str): 'train' or 'val'/'test'.
    """
    normalize = transforms.Normalize(mean=Config.IMAGENET_MEAN, std=Config.IMAGENET_STD)

    if mode == "train":
        return transforms.Compose(
            [
                transforms.Resize(Config.RESIZE_SIZE),
                transforms.RandomCrop(Config.IMG_SIZE),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        # Validation and Test
        return transforms.Compose(
            [
                transforms.Resize(Config.RESIZE_SIZE),
                transforms.CenterCrop(Config.IMG_SIZE),
                transforms.ToTensor(),
                normalize,
            ]
        )


class ArtworkDataset(Dataset):
    """
    PyTorch Dataset for Multi-Label Artwork Classification.
    """

    def __init__(
        self, df: pd.DataFrame, transform=None, num_classes: int = Config.NUM_CLASSES
    ):
        """
        Args:
            df (pd.DataFrame): DataFrame containing 'full_path' and 'label_list'.
            transform (callable, optional): Transform to be applied on a sample.
            num_classes (int): Total number of classes for multi-hot encoding.
        """
        self.data = df
        self.transform = transform
        self.num_classes = num_classes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = row["full_path"]
        label_indices = row["label_list"]

        # Load Image
        try:
            # Convert to RGB to handle grayscale or alpha channels consistently
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            # Fallback for corrupted images (though EDA showed 0 missing files)
            # Return a black image to prevent crashing
            print(f"Error loading image {image_path}: {e}")
            image = Image.new("RGB", (Config.IMG_SIZE, Config.IMG_SIZE))

        # Apply Transforms
        if self.transform:
            image = self.transform(image)

        # Create Multi-hot Label Vector
        # Initialize with zeros
        target = torch.zeros(self.num_classes, dtype=torch.float32)

        # Scatter ones at the specified indices
        if len(label_indices) > 0:
            # Ensure indices are within bounds
            valid_indices = [i for i in label_indices if 0 <= i < self.num_classes]
            target[valid_indices] = 1.0

        return image, target


def get_dataloaders(
    batch_size: int = Config.BATCH_SIZE,
    num_workers: int = Config.NUM_WORKERS,
    load_cached_data: bool = True,
    debug: bool = Config.DEBUG,
):
    """
    Creates and returns DataLoaders for train, validation, and test sets.

    Args:
        batch_size (int): Batch size for loading.
        num_workers (int): Number of subprocesses for data loading.
        load_cached_data (bool): Whether to use cached metadata.
        debug (bool): If True, limits dataset size for debugging.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Prepare Metadata
    train_df = prepare_metadata("train", load_cached_data=load_cached_data)
    val_df = prepare_metadata("val", load_cached_data=load_cached_data)
    test_df = prepare_metadata("test", load_cached_data=load_cached_data)

    # Debug Mode: Subsample data
    if debug:
        subset_size = Config.DEBUG_SUBSET_SIZE
        print(f"DEBUG mode enabled. Subsampling datasets to {subset_size} samples.")
        train_df = train_df.iloc[:subset_size]
        val_df = val_df.iloc[:subset_size]
        test_df = test_df.iloc[:subset_size]

    # Instantiate Datasets
    train_dataset = ArtworkDataset(
        train_df, transform=get_transforms("train"), num_classes=Config.NUM_CLASSES
    )

    val_dataset = ArtworkDataset(
        val_df, transform=get_transforms("val"), num_classes=Config.NUM_CLASSES
    )

    test_dataset = ArtworkDataset(
        test_df,
        transform=get_transforms("test"),  # Test uses same transforms as Val
        num_classes=Config.NUM_CLASSES,
    )

    # Instantiate DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=Config.PIN_MEMORY,
        drop_last=True,  # Drop last incomplete batch for training stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=Config.PIN_MEMORY,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=Config.PIN_MEMORY,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
