import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config


def load_metadata(split, load_cached_data=True):
    """
    Loads metadata for a specific split (train, val, test).
    Implements caching using parquet files in Config.WORKING_DIR.
    """
    cache_path = os.path.join(Config.WORKING_DIR, f"{split}_metadata.parquet")

    # 1. Try to load cached data
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            print(f"Failed to load cache for {split}: {e}. Reloading from source.")

    # 2. Load from source
    if split == "train":
        source_path = Config.TRAIN_METADATA_PATH
    elif split == "val":
        source_path = Config.VAL_METADATA_PATH
    elif split == "test":
        source_path = Config.TEST_METADATA_PATH
    else:
        raise ValueError(f"Unknown split: {split}")

    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Metadata file not found: {source_path}")

    df = pd.read_csv(source_path)

    # Process labels: Convert space-delimited string to list
    # Handle potential NaN or non-string labels gracefully
    df["label_list"] = df["labels"].apply(
        lambda x: x.split() if isinstance(x, str) else []
    )

    # 3. Save to cache
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    df.to_parquet(cache_path, index=False)

    return df


class AppleDataset(Dataset):
    """
    PyTorch Dataset for Apple Disease Detection.
    Handles image loading and multi-label target generation.
    """

    def __init__(self, df, transforms=None, debug=False):
        self.df = df
        self.transforms = transforms
        self.classes = Config.CLASSES
        self.num_classes = len(self.classes)
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        # In debug mode, reduce dataset size
        if debug:
            self.df = self.df.head(100).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full image path
        # Metadata file_path is relative to input dir (e.g., "train_images/xyz.jpg")
        image_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            # Fallback for missing images (should not happen based on metadata checks)
            # Create a black image of expected size
            image = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]
        else:
            # Minimal transform if none provided
            T = A.Compose(
                [
                    A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                    A.Normalize(),
                    ToTensorV2(),
                ]
            )
            image = T(image=image)["image"]

        # Generate Target (Multi-hot encoding)
        labels = row["label_list"]
        target = torch.zeros(self.num_classes, dtype=torch.float32)

        for label in labels:
            if label in self.class_to_idx:
                target[self.class_to_idx[label]] = 1.0

        # Return image ID for submission file generation
        image_id = row["image"]

        return image, target, image_id


def get_transforms(data="train"):
    """
    Returns Albumentations transforms for training or validation/testing.
    Includes CoarseDropout for training as per strategy.
    """
    if data == "train":
        return A.Compose(
            [
                A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                # CoarseDropout: Rectangular holes to force distributed feature learning
                A.CoarseDropout(
                    max_holes=8,
                    max_height=32,
                    max_width=32,
                    min_holes=1,
                    min_height=8,
                    min_width=8,
                    fill_value=0,
                    p=0.5,
                ),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )
    elif data == "valid" or data == "test":
        return A.Compose(
            [
                A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )
    else:
        raise ValueError(f"Unknown data mode: {data}")


def get_loaders(load_cached_data=True):
    """
    Creates and returns DataLoaders for train, val, and test sets.

    Args:
        load_cached_data (bool): Whether to use cached metadata.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load Metadata
    train_df = load_metadata("train", load_cached_data=load_cached_data)
    val_df = load_metadata("val", load_cached_data=load_cached_data)
    test_df = load_metadata("test", load_cached_data=load_cached_data)

    # Create Datasets
    train_dataset = AppleDataset(
        train_df, transforms=get_transforms("train"), debug=Config.DEBUG
    )
    val_dataset = AppleDataset(
        val_df, transforms=get_transforms("valid"), debug=Config.DEBUG
    )
    test_dataset = AppleDataset(
        test_df, transforms=get_transforms("test"), debug=Config.DEBUG
    )

    # Create DataLoaders
    # Drop last for train to maintain batch statistics stability
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

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
