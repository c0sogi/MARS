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
from library.utils import seed_everything


def process_train_data(load_cached_data=True):
    """
    Loads training and validation metadata, concatenates them for CV,
    computes binary targets, and caches the result.
    """
    cache_path = os.path.join(Config.WORKING_DIR, "train_cache.parquet")

    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # Load provided metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA)
    val_df = pd.read_csv(Config.VAL_METADATA)

    # Concatenate for Cross-Validation
    df = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)

    # Decompose targets into binary labels
    # Rust = 1 if 'rust' or 'multiple_diseases' is 1
    # Scab = 1 if 'scab' or 'multiple_diseases' is 1
    df["target_rust"] = df[["rust", "multiple_diseases"]].max(axis=1).astype(int)
    df["target_scab"] = df[["scab", "multiple_diseases"]].max(axis=1).astype(int)

    # Save to cache
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    df.to_parquet(cache_path, index=False)

    return df


def process_test_data(load_cached_data=True):
    """
    Loads test metadata and caches the result.
    """
    cache_path = os.path.join(Config.WORKING_DIR, "test_cache.parquet")

    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            print(f"Failed to load test cache: {e}. Recomputing...")

    df = pd.read_csv(Config.TEST_METADATA)

    # Save to cache
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    df.to_parquet(cache_path, index=False)

    return df


class AppleDataset(Dataset):
    def __init__(self, df, img_size, transforms=None, is_test=False):
        self.df = df
        self.img_size = img_size
        self.transforms = transforms
        self.is_test = is_test

        # Pre-construct file paths
        self.file_paths = (
            df["file_path"].apply(lambda x: os.path.join(Config.INPUT_DIR, x)).values
        )
        self.image_ids = df["image_id"].values

        if not self.is_test:
            self.targets = df[["target_rust", "target_scab"]].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = cv2.imread(path)

        if image is None:
            # Fallback for missing images (should not happen based on metadata check)
            image = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]
        else:
            # Fallback resize and normalize if no transform provided
            resizer = A.Compose(
                [A.Resize(self.img_size, self.img_size), A.Normalize(), ToTensorV2()]
            )
            image = resizer(image=image)["image"]

        if self.is_test:
            return image, self.image_ids[idx]
        else:
            target = torch.tensor(self.targets[idx])
            return image, target


def get_transforms(img_size, mode="train"):
    """
    Returns Albumentations transforms.
    Adapts CoarseDropout parameters based on image size.
    """
    if mode == "train":
        # Determine max hole size based on resolution
        if img_size == Config.IMG_SIZE_EFFNET:
            max_hole = Config.AUG_HOLE_SIZE_MAX_EFFNET
        else:
            max_hole = Config.AUG_HOLE_SIZE_MAX_CONVNEXT

        return A.Compose(
            [
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.CoarseDropout(
                    max_holes=Config.AUG_HOLES_NUM_MAX,
                    min_holes=Config.AUG_HOLES_NUM_MIN,
                    max_height=max_hole,
                    max_width=max_hole,
                    min_height=Config.AUG_HOLE_SIZE_MIN,
                    min_width=Config.AUG_HOLE_SIZE_MIN,
                    fill_value=0,
                    p=0.5,
                ),
                A.Normalize(),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(img_size, img_size),
                A.Normalize(),
                ToTensorV2(),
            ]
        )


def get_loaders(fold_idx, img_size, load_cached_data=True):
    """
    Creates DataLoaders for a specific fold using StratifiedKFold.
    Returns train_loader, val_loader, and positive class weights.
    """
    # Load full dataset
    df = process_train_data(load_cached_data=load_cached_data)

    # Stratified Split
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # We split based on the 'stratify_label' column which contains the multiclass label name
    folds = list(skf.split(df, df["stratify_label"]))
    train_idx, val_idx = folds[fold_idx]

    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()

    # Debug mode
    if Config.DEBUG:
        train_df = train_df.head(Config.DEBUG_SAMPLE_SIZE)
        val_df = val_df.head(Config.DEBUG_SAMPLE_SIZE)

    # Calculate Positive Class Weights for BCE
    # Weight = Num_Negative / Num_Positive
    # Targets: [Rust, Scab]
    targets = train_df[["target_rust", "target_scab"]].values
    pos_counts = np.sum(targets, axis=0)
    neg_counts = len(train_df) - pos_counts

    # Avoid division by zero
    pos_counts = np.clip(pos_counts, 1, None)
    pos_weights = neg_counts / pos_counts
    pos_weights = torch.tensor(pos_weights, dtype=torch.float32)

    # Datasets
    train_ds = AppleDataset(
        train_df,
        img_size=img_size,
        transforms=get_transforms(img_size, mode="train"),
        is_test=False,
    )

    val_ds = AppleDataset(
        val_df,
        img_size=img_size,
        transforms=get_transforms(img_size, mode="val"),
        is_test=False,
    )

    # Worker Init Fn for reproducibility
    def worker_init_fn(worker_id):
        seed_everything(Config.SEED + worker_id)

    # Loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_init_fn,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=worker_init_fn,
    )

    return train_loader, val_loader, pos_weights


def get_test_loader(img_size, load_cached_data=True):
    """
    Creates DataLoader for the test set.
    """
    df = process_test_data(load_cached_data=load_cached_data)

    test_ds = AppleDataset(
        df,
        img_size=img_size,
        transforms=get_transforms(img_size, mode="test"),
        is_test=True,
    )

    loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return loader
