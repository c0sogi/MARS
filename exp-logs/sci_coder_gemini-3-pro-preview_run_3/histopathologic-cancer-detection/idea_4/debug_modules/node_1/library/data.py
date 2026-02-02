import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold
from library.config import Config


def get_transforms(data_split: str):
    """
    Returns the Albumentations transform pipeline for a given data split.

    Args:
        data_split (str): 'train', 'val', or 'test'.

    Returns:
        A.Compose: The transform pipeline.
    """
    if data_split == "train":
        return A.Compose(
            [
                # Strictly maintain Center Crop of 64x64 pixels
                A.CenterCrop(height=Config.CROP_SIZE, width=Config.CROP_SIZE),
                # Geometric augmentations for rotational invariance
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                # Normalization and Tensor conversion
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        # Validation and Test: Center Crop + Normalize only
        return A.Compose(
            [
                A.CenterCrop(height=Config.CROP_SIZE, width=Config.CROP_SIZE),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


class PathologyDataset(Dataset):
    """
    PyTorch Dataset for Digital Pathology images.
    """

    def __init__(self, df: pd.DataFrame, transform=None):
        """
        Args:
            df (pd.DataFrame): DataFrame containing 'file_path' and 'label'.
            transform (A.Compose, optional): Albumentations transforms.
        """
        self.df = df
        self.transform = transform
        self.root_dir = Config.INPUT_DIR

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full file path
        # Metadata contains relative paths like "train/id.tif"
        file_path = os.path.join(self.root_dir, row["file_path"])

        # Read image
        image = cv2.imread(file_path)
        if image is None:
            # Fallback for missing images (though verification script passed)
            # Create a black image of expected size to prevent crash
            image = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Get label
        label = row["label"] if "label" in row else 0

        return image, torch.tensor(label, dtype=torch.float32)


def prepare_folds(load_cached_data: bool = True) -> pd.DataFrame:
    """
    Loads training data, merges provided splits, and creates Stratified K-Folds.
    Uses caching to store the fold assignments.

    Args:
        load_cached_data (bool): If True, attempts to load from parquet cache.

    Returns:
        pd.DataFrame: DataFrame with 'fold' column.
    """
    cache_path = os.path.join(Config.WORK_DIR, "folds.parquet")

    # 1. Try to load cached data
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception:
            pass  # Fallback to re-creation if read fails

    # 2. Compute from scratch
    # Load original metadata
    df_train = pd.read_csv(Config.TRAIN_META_PATH)
    df_val = pd.read_csv(Config.VAL_META_PATH)

    # Merge to create a full dataset for Cross-Validation
    df_full = pd.concat([df_train, df_val], axis=0).reset_index(drop=True)

    # Create Stratified K-Folds
    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )
    df_full["fold"] = -1

    for fold, (train_idx, val_idx) in enumerate(skf.split(df_full, df_full["label"])):
        df_full.loc[val_idx, "fold"] = fold

    # 3. Save to cache
    os.makedirs(Config.WORK_DIR, exist_ok=True)
    df_full.to_parquet(cache_path, index=False)

    return df_full


def load_test_metadata(load_cached_data: bool = True) -> pd.DataFrame:
    """
    Loads test metadata with caching.

    Args:
        load_cached_data (bool): If True, attempts to load from parquet cache.

    Returns:
        pd.DataFrame: Test metadata.
    """
    cache_path = os.path.join(Config.WORK_DIR, "cached_test_metadata.parquet")

    if load_cached_data and os.path.exists(cache_path):
        try:
            return pd.read_parquet(cache_path)
        except Exception:
            pass

    df_test = pd.read_csv(Config.TEST_META_PATH)

    os.makedirs(Config.WORK_DIR, exist_ok=True)
    df_test.to_parquet(cache_path, index=False)

    return df_test


def get_dataloaders(
    fold_id: int,
    batch_size: int = Config.BATCH_SIZE,
    num_workers: int = Config.NUM_WORKERS,
    load_cached_data: bool = True,
):
    """
    Creates train and validation DataLoaders for a specific fold.

    Args:
        fold_id (int): The fold index to use for validation (0 to NUM_FOLDS-1).
        batch_size (int): Batch size.
        num_workers (int): Number of worker threads.
        load_cached_data (bool): Whether to use cached fold data.

    Returns:
        tuple: (train_loader, val_loader)
    """
    df = prepare_folds(load_cached_data=load_cached_data)

    # Split based on fold
    train_df = df[df["fold"] != fold_id].reset_index(drop=True)
    val_df = df[df["fold"] == fold_id].reset_index(drop=True)

    # Create Datasets
    train_dataset = PathologyDataset(train_df, transform=get_transforms("train"))
    val_dataset = PathologyDataset(val_df, transform=get_transforms("val"))

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
    )

    return train_loader, val_loader


def get_test_dataloader(
    batch_size: int = Config.BATCH_SIZE,
    num_workers: int = Config.NUM_WORKERS,
    load_cached_data: bool = True,
):
    """
    Creates the test DataLoader.

    Args:
        batch_size (int): Batch size.
        num_workers (int): Number of worker threads.
        load_cached_data (bool): Whether to use cached metadata.

    Returns:
        DataLoader: The test data loader.
    """
    df_test = load_test_metadata(load_cached_data=load_cached_data)

    test_dataset = PathologyDataset(df_test, transform=get_transforms("test"))

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return test_loader
