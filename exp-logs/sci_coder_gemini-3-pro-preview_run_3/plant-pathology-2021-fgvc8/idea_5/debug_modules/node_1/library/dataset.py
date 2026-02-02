import os
import cv2
import torch
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from library.config import Config


def load_processed_dataframe(csv_path, cache_name, load_cached_data=True):
    """
    Loads a dataframe from a CSV file, with caching to Parquet.

    Args:
        csv_path (str): Path to the source CSV file.
        cache_name (str): Name for the cached parquet file (e.g., 'train_cache').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: The loaded dataframe.
    """
    cache_path = os.path.join(Config.WORKING_DIR, f"{cache_name}.parquet")

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            print(
                f"Failed to load cache from {cache_path}: {e}. Reloading from source."
            )

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Source file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Save to cache
    try:
        df.to_parquet(cache_path, index=False)
    except Exception as e:
        print(f"Warning: Failed to save cache to {cache_path}: {e}")

    return df


def get_transforms(data="train"):
    """
    Returns the Albumentations augmentation pipeline based on the data mode.

    Args:
        data (str): 'train', 'valid', or 'test'.

    Returns:
        A.Compose: The transform pipeline.
    """
    if data == "train":
        return A.Compose(
            [
                # High-Resolution Input Strategy: RandomResizedCrop with min scale 0.5
                A.RandomResizedCrop(
                    size=(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
                    scale=(Config.AUG_SCALE_MIN, Config.AUG_SCALE_MAX),
                ),
                # Rotational Invariance
                A.HorizontalFlip(p=Config.AUG_HORIZONTAL_FLIP_PROB),
                A.VerticalFlip(p=Config.AUG_VERTICAL_FLIP_PROB),
                # Lighting Invariance
                A.ColorJitter(
                    brightness=Config.AUG_COLOR_JITTER,
                    contrast=Config.AUG_COLOR_JITTER,
                    saturation=0,
                    hue=0,
                    p=0.5,
                ),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )
    elif data in ["valid", "test"]:
        return A.Compose(
            [
                A.Resize(height=Config.IMAGE_SIZE, width=Config.IMAGE_SIZE),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )
    else:
        raise ValueError(f"Unknown data mode: {data}")


class AppleDataset(Dataset):
    """
    PyTorch Dataset for Apple Disease Detection.
    """

    def __init__(self, df, transforms=None):
        """
        Args:
            df (pd.DataFrame): Dataframe containing 'image', 'labels', and 'file_path'.
            transforms (A.Compose): Albumentations transforms.
        """
        self.df = df
        self.transforms = transforms
        self.file_paths = df["file_path"].values
        self.image_ids = df["image"].values
        self.labels = df["labels"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Resolve full path
        rel_path = self.file_paths[idx]
        full_path = os.path.join(Config.INPUT_DIR, rel_path)

        # Load Image
        image = cv2.imread(full_path)
        if image is None:
            # Fallback for missing images (though metadata check should prevent this)
            # Create a black image of expected size
            image = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply Transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        # Process Labels (Multi-hot encoding)
        label_str = self.labels[idx]
        target = torch.zeros(Config.NUM_CLASSES, dtype=torch.float32)

        # Handle test set placeholder or empty labels
        if isinstance(label_str, str) and label_str.strip():
            current_labels = label_str.split()
            for lbl in current_labels:
                if lbl in Config.LABEL2ID:
                    class_id = Config.LABEL2ID[lbl]
                    target[class_id] = 1.0

        image_id = self.image_ids[idx]

        return image, target, image_id


def get_loaders(
    load_cached_data=True, batch_size=Config.BATCH_SIZE, debug=Config.DEBUG
):
    """
    Creates and returns DataLoaders for train, validation, and test sets.

    Args:
        load_cached_data (bool): Whether to use cached dataframes.
        batch_size (int): Batch size for training/inference.
        debug (bool): If True, subsets the data for quick debugging.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load Dataframes
    train_df = load_processed_dataframe(
        Config.TRAIN_CSV, "train_processed", load_cached_data
    )
    val_df = load_processed_dataframe(Config.VAL_CSV, "val_processed", load_cached_data)
    test_df = load_processed_dataframe(
        Config.TEST_CSV, "test_processed", load_cached_data
    )

    # Debug Mode: Subset data
    if debug:
        train_df = train_df.head(Config.DEBUG_SUBSET_SIZE).reset_index(drop=True)
        val_df = val_df.head(Config.DEBUG_SUBSET_SIZE).reset_index(drop=True)
        test_df = test_df.head(Config.DEBUG_SUBSET_SIZE).reset_index(drop=True)

    # Initialize Datasets
    train_dataset = AppleDataset(train_df, transforms=get_transforms("train"))
    val_dataset = AppleDataset(val_df, transforms=get_transforms("valid"))
    test_dataset = AppleDataset(test_df, transforms=get_transforms("test"))

    # Initialize DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
