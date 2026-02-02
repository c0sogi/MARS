import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import CFG
from library.utils import seed_everything


def process_data(metadata_path, cache_path, load_cached_data=True):
    """
    Loads metadata, processes targets for multi-label decomposition, and handles caching.

    Args:
        metadata_path (str): Path to the source metadata CSV.
        cache_path (str): Path to save/load the processed parquet file.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: The processed dataframe.
    """
    # Ensure working directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            # Verify essential columns exist, otherwise re-process
            if "file_path" in df.columns:
                return df
        except Exception:
            pass  # Fallback to processing

    # Load metadata
    df = pd.read_csv(metadata_path)

    # Process Targets if training data (check for label columns)
    # We need to decompose into 2 binary tasks: Rust and Scab.
    # Logic:
    #   Rust Present = 'rust' == 1 OR 'multiple_diseases' == 1
    #   Scab Present = 'scab' == 1 OR 'multiple_diseases' == 1

    # Check if label columns exist
    if (
        "rust" in df.columns
        and "multiple_diseases" in df.columns
        and "scab" in df.columns
    ):
        df["target_rust"] = df["rust"] + df["multiple_diseases"]
        df["target_scab"] = df["scab"] + df["multiple_diseases"]

        # Clip to ensure binary (though sums shouldn't exceed 1 given one-hot nature)
        df["target_rust"] = df["target_rust"].clip(upper=1.0)
        df["target_scab"] = df["target_scab"].clip(upper=1.0)

    # Save to cache
    df.to_parquet(cache_path, index=False)

    return df


class AppleDataset(Dataset):
    def __init__(self, df, transform=None, mode="train"):
        """
        Args:
            df (pd.DataFrame): DataFrame with file paths and labels.
            transform (A.Compose): Albumentations transforms.
            mode (str): 'train', 'val', or 'test'.
        """
        self.df = df
        self.transform = transform
        self.mode = mode

        # Pre-construct full paths
        # Metadata contains relative path e.g., "images/Train_0.jpg"
        # CFG.input_dir is "./input"
        self.file_paths = [
            os.path.join(CFG.input_dir, fp) for fp in df["file_path"].values
        ]
        self.image_ids = df["image_id"].values

        if self.mode != "test":
            self.labels = df[["target_rust", "target_scab"]].values.astype(np.float32)
        else:
            self.labels = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        image_id = self.image_ids[idx]

        # Load image
        image = cv2.imread(file_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {file_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Return based on mode
        if self.mode != "test":
            label = torch.tensor(self.labels[idx])
            return image, label, image_id
        else:
            return image, image_id


def get_transforms(data, img_size):
    """
    Returns the Albumentations transforms for the specific dataset split and image size.

    Args:
        data (str): 'train' or 'valid' (applies to test as well).
        img_size (int): Target image resolution.
    """
    if data == "train":
        return A.Compose(
            [
                A.Resize(img_size, img_size),
                # CoarseDropout as per Idea 10
                A.CoarseDropout(
                    max_holes=CFG.coarse_dropout_params["max_holes"],
                    max_height=CFG.coarse_dropout_params["max_height"],
                    max_width=CFG.coarse_dropout_params["max_width"],
                    min_holes=CFG.coarse_dropout_params["min_holes"],
                    min_height=CFG.coarse_dropout_params["min_height"],
                    min_width=CFG.coarse_dropout_params["min_width"],
                    fill_value=CFG.coarse_dropout_params["fill_value"],
                    p=CFG.coarse_dropout_params["p"],
                ),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    elif data == "valid":
        return A.Compose(
            [
                A.Resize(img_size, img_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    else:
        # Fallback
        return A.Compose(
            [
                A.Resize(img_size, img_size),
                A.Normalize(),
                ToTensorV2(),
            ]
        )


def get_loaders(train_df, val_df, test_df, img_size, batch_size=CFG.batch_size):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        train_df (pd.DataFrame): Processed training dataframe.
        val_df (pd.DataFrame): Processed validation dataframe.
        test_df (pd.DataFrame): Processed test dataframe.
        img_size (int): Image resolution for resizing.
        batch_size (int): Batch size.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """

    # Define Transforms
    train_transform = get_transforms("train", img_size)
    val_transform = get_transforms("valid", img_size)

    # Create Datasets
    train_dataset = AppleDataset(train_df, transform=train_transform, mode="train")
    val_dataset = AppleDataset(val_df, transform=val_transform, mode="val")
    test_dataset = AppleDataset(
        test_df, transform=val_transform, mode="test"
    )  # Use valid transform for test

    # Worker Init Function for Reproducibility
    def worker_init_fn(worker_id):
        seed_everything(CFG.seed + worker_id)

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
