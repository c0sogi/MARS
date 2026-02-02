import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.augmentations import get_transforms


def load_metadata(split: str, load_cached_data: bool = True) -> pd.DataFrame:
    """
    Loads metadata for a specific split, implementing the required caching logic.

    Args:
        split (str): One of 'train', 'val', 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: The metadata dataframe.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    cache_path = os.path.join(Config.WORKING_DIR, f"metadata_{split}.parquet")

    # Logic Flow 1: Try to load cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception:
            # If load fails, proceed to compute from scratch
            pass

    # Logic Flow 2: Compute/Process from scratch
    csv_path = os.path.join(Config.METADATA_DIR, f"{split}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Save result to cache
    df.to_parquet(cache_path)

    return df


class CatDogDataset(Dataset):
    """
    Dataset class for Dog vs Cat classification.
    """

    def __init__(self, df: pd.DataFrame, transforms=None, mode: str = "train"):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata (filepath, label/id).
            transforms (albumentations.Compose): Transforms to apply.
            mode (str): 'train', 'val', or 'test'.
        """
        self.df = df
        self.transforms = transforms
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full path
        # filepath in metadata is relative to input dir (e.g., "train/cat.0.jpg")
        img_path = os.path.join(Config.INPUT_DIR, row["filepath"])

        # Load image
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found or corrupt: {img_path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        # Return based on mode
        if self.mode in ["train", "val"]:
            # Label is float for BCEWithLogitsLoss
            label = torch.tensor(row["label"], dtype=torch.float32)
            return image, label
        else:
            # Test mode returns image and ID for submission
            img_id = row["id"]
            return image, img_id


def worker_init_fn(worker_id):
    """
    Sets random seed for dataloader workers to ensure reproducibility.
    """
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def get_dataloaders(img_size: int, batch_size: int, load_cached_data: bool = True):
    """
    Constructs DataLoaders for train, validation, and test sets.

    Args:
        img_size (int): Target image size (e.g., 256 for CNN, 224 for ViT).
        batch_size (int): Batch size.
        load_cached_data (bool): Whether to use cached metadata.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load Metadata
    train_df = load_metadata("train", load_cached_data)
    val_df = load_metadata("val", load_cached_data)
    test_df = load_metadata("test", load_cached_data)

    # Get Transforms
    # Note: 'val' split transform is deterministic and suitable for test as well
    train_transforms = get_transforms(img_size, split="train")
    val_transforms = get_transforms(img_size, split="val")

    # Instantiate Datasets
    train_dataset = CatDogDataset(train_df, transforms=train_transforms, mode="train")
    val_dataset = CatDogDataset(val_df, transforms=val_transforms, mode="val")
    test_dataset = CatDogDataset(test_df, transforms=val_transforms, mode="test")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        drop_last=True,  # Drop last incomplete batch for training stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )

    return train_loader, val_loader, test_loader
