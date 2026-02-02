import os
import cv2
import torch
import numpy as np
import pandas as pd
import random
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from library.config import Config
from library.augmentations import get_train_transforms, get_valid_transforms

# Prevent OpenCV from using multithreading to avoid contention with PyTorch DataLoader workers
cv2.setNumThreads(0)


def prepare_folds(load_cached_data: bool = True) -> pd.DataFrame:
    """
    Prepares the dataset for 5-Fold Stratified Cross-Validation.
    Merges existing train/val metadata and creates new fold assignments.
    Caches the result to ensure consistency.

    Args:
        load_cached_data (bool): If True, attempts to load from disk.

    Returns:
        pd.DataFrame: Dataframe containing image paths, labels, and fold assignments.
    """
    cache_path = os.path.join(Config.output_dir, "folds.parquet")

    # Ensure output directory exists
    os.makedirs(Config.output_dir, exist_ok=True)

    # 1. Try to load cached folds
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            # Basic validation to ensure cache isn't corrupted
            if "fold" in df.columns and len(df) > 0:
                return df
        except Exception:
            pass  # Fallback to recomputing if load fails

    # 2. Compute from scratch
    # Load existing metadata
    df_train = pd.read_csv(Config.train_metadata_path)
    df_val = pd.read_csv(Config.val_metadata_path)

    # Merge to form the "entire dataset"
    df = pd.concat([df_train, df_val], axis=0).reset_index(drop=True)

    # Create Stratified Folds
    skf = StratifiedKFold(
        n_splits=Config.n_folds, shuffle=True, random_state=Config.seed
    )
    df["fold"] = -1

    for fold, (_, val_idx) in enumerate(skf.split(df, df["label"])):
        df.loc[val_idx, "fold"] = fold

    # 3. Save to cache
    df.to_parquet(cache_path, index=False)

    return df


class CassavaDataset(Dataset):
    """
    PyTorch Dataset for Cassava Leaf Disease Classification.
    """

    def __init__(
        self, df: pd.DataFrame, transforms=None, data_root: str = Config.input_root
    ):
        """
        Args:
            df (pd.DataFrame): Dataframe containing 'file_path' and 'label'.
            transforms (albumentations.Compose): Transforms to apply.
            data_root (str): Root directory for image files.
        """
        self.df = df
        self.transforms = transforms
        self.data_root = data_root

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full path
        # Metadata file_path is relative to input_root (e.g., "train_images/123.jpg")
        path = os.path.join(self.data_root, row["file_path"])

        # Load Image
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Image not found at {path}")

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Apply Transforms
        if self.transforms:
            # Albumentations returns a dict
            img = self.transforms(image=img)["image"]
        else:
            # Fallback: Convert to tensor and normalize to [0, 1]
            img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0

        # Get Label
        label = torch.tensor(row["label"], dtype=torch.long)

        return img, label


def worker_init_fn(worker_id):
    """
    Sets random seeds for DataLoader workers to ensure reproducibility.
    """
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_loaders(
    fold: int,
    image_size: int = Config.p1_image_size,
    batch_size: int = Config.p1_batch_size,
    debug: bool = Config.debug,
):
    """
    Creates DataLoaders for the specified fold.

    Args:
        fold (int): The fold index to use for validation (0 to n_folds-1).
        image_size (int): Target image size for resizing/cropping.
        batch_size (int): Batch size.
        debug (bool): If True, uses a small subset of data for debugging.

    Returns:
        tuple: (train_loader, val_loader)
    """
    # Load and split data
    df = prepare_folds(load_cached_data=True)

    # Split into Train and Validation based on fold
    df_train = df[df["fold"] != fold].reset_index(drop=True)
    df_val = df[df["fold"] == fold].reset_index(drop=True)

    # Handle Debug Mode
    if debug:
        # Sample subsets deterministically
        df_train = df_train.sample(
            n=min(len(df_train), 200), random_state=Config.seed
        ).reset_index(drop=True)
        df_val = df_val.sample(
            n=min(len(df_val), 100), random_state=Config.seed
        ).reset_index(drop=True)

    # Create Datasets
    train_dataset = CassavaDataset(
        df_train,
        transforms=get_train_transforms(image_size),
        data_root=Config.input_root,
    )

    val_dataset = CassavaDataset(
        df_val, transforms=get_valid_transforms(image_size), data_root=Config.input_root
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_init_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=worker_init_fn,
    )

    return train_loader, val_loader


def get_test_loader(
    image_size: int = Config.p1_image_size, batch_size: int = Config.p1_batch_size
):
    """
    Creates a DataLoader for the test set.

    Args:
        image_size (int): Target image size.
        batch_size (int): Batch size.

    Returns:
        DataLoader: Test data loader.
    """
    # Load test metadata
    df_test = pd.read_csv(Config.test_metadata_path)

    # Create Dataset
    test_dataset = CassavaDataset(
        df_test,
        transforms=get_valid_transforms(image_size),
        data_root=Config.input_root,
    )

    # Create DataLoader
    # Shuffle must be False to preserve order for submission
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=worker_init_fn,
    )

    return test_loader
