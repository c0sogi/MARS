import os
import cv2
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from library.config import Config


def get_dataframes(load_cached_data=True):
    """
    Loads metadata dataframes for train, val, and test sets.
    Implements caching using Parquet format to the working directory.

    Args:
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    cache_path_train = os.path.join(cache_dir, "train_metadata.parquet")
    cache_path_val = os.path.join(cache_dir, "val_metadata.parquet")
    cache_path_test = os.path.join(cache_dir, "test_metadata.parquet")

    # Attempt to load from cache
    if (
        load_cached_data
        and os.path.exists(cache_path_train)
        and os.path.exists(cache_path_val)
        and os.path.exists(cache_path_test)
    ):
        try:
            train_df = pd.read_parquet(cache_path_train)
            val_df = pd.read_parquet(cache_path_val)
            test_df = pd.read_parquet(cache_path_test)
            return train_df, val_df, test_df
        except Exception:
            # If load fails, fall back to processing from scratch
            pass

    # Load from source CSVs
    train_df = pd.read_csv(Config.TRAIN_METADATA)
    val_df = pd.read_csv(Config.VAL_METADATA)
    test_df = pd.read_csv(Config.TEST_METADATA)

    # Save to cache
    train_df.to_parquet(cache_path_train)
    val_df.to_parquet(cache_path_val)
    test_df.to_parquet(cache_path_test)

    return train_df, val_df, test_df


class CatDogDataset(Dataset):
    """
    Custom Dataset for Dog vs Cat classification.
    Loads images via OpenCV, converts to PIL, and applies transforms.
    """

    def __init__(self, df, transform=None, mode="train"):
        self.df = df
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rel_path = row["filepath"]
        full_path = os.path.join(Config.INPUT_DIR, rel_path)

        # Load image using OpenCV
        img = cv2.imread(full_path)

        if img is None:
            # Handle missing/corrupt images by returning a black image
            # This prevents the dataloader from crashing
            img = np.zeros((256, 256, 3), dtype=np.uint8)
        else:
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image for Torchvision transforms compatibility
        img_pil = Image.fromarray(img)

        # Apply transforms
        if self.transform:
            img_tensor = self.transform(img_pil)
        else:
            # Fallback to basic tensor conversion
            img_tensor = T.ToTensor()(img_pil)

        if self.mode == "test":
            # Return image and ID for submission generation
            return img_tensor, row["id"]
        else:
            # Return image and label (float32 for BCEWithLogitsLoss)
            label = torch.tensor(row["label"], dtype=torch.float32)
            return img_tensor, label


def get_transforms(size, mode="train"):
    """
    Generates the augmentation pipeline based on the mode and target size.
    Uses Bicubic interpolation as per requirements.
    """
    # ImageNet normalization statistics
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if mode == "train":
        return T.Compose(
            [
                # Context-Preserving Augmentation
                T.RandomResizedCrop(
                    size,
                    scale=Config.AUG_RRC_SCALE,
                    interpolation=T.InterpolationMode.BICUBIC,
                ),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(
                    brightness=Config.AUG_COLOR_JITTER,
                    contrast=Config.AUG_COLOR_JITTER,
                    saturation=Config.AUG_COLOR_JITTER,
                ),
                T.ToTensor(),
                T.Normalize(mean, std),
            ]
        )
    else:
        # Validation/Test: Deterministic Resize
        return T.Compose(
            [
                T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean, std),
            ]
        )


def get_dataloader(
    df, img_size, batch_size, mode="train", num_workers=Config.NUM_WORKERS
):
    """
    Factory function to create a DataLoader with specific image size and batch size.
    Essential for Progressive Resizing where img_size changes between phases.
    """
    transform = get_transforms(img_size, mode)
    dataset = CatDogDataset(df, transform=transform, mode=mode)

    # Shuffle only for training
    shuffle = mode == "train"

    # Drop last batch in training to maintain consistent batch statistics (optional but recommended)
    drop_last = mode == "train"

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,  # Faster data transfer to GPU
        drop_last=drop_last,
    )
