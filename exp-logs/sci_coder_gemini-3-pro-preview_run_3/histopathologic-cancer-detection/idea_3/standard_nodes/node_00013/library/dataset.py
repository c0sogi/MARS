import os
import cv2
import torch
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from library.config import Config


class PathologyDataset(Dataset):
    """
    PyTorch Dataset for Pathology Tumor Detection.
    Loads images, applies transformations, and returns tensors.
    """

    def __init__(
        self, df: pd.DataFrame, transform: A.Compose = None, is_test: bool = False
    ):
        self.df = df
        self.transform = transform
        self.is_test = is_test

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        # Construct full file path
        # Metadata contains relative paths (e.g., "train/id.tif")
        img_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # Load image
        image = cv2.imread(img_path)
        if image is None:
            # Fallback for missing/corrupt images (though metadata check passed)
            # Create a black image of expected size
            image = np.zeros((Config.INPUT_SIZE, Config.INPUT_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Fallback to simple tensor conversion if no transform provided
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        # Return data
        if self.is_test:
            # For test set, we might need the ID for submission,
            # but standard loaders usually just return X.
            # We return image and a placeholder label.
            return image, torch.tensor(0.0, dtype=torch.float32)
        else:
            label = row["label"]
            return image, torch.tensor(label, dtype=torch.float32)


def get_transforms(phase: str = "train") -> A.Compose:
    """
    Returns the Albumentations transform pipeline for the specified phase.

    Args:
        phase (str): 'train', 'valid', or 'test'.
    """
    # Common normalization (ImageNet defaults are standard for ResNet/ConvNeXt)
    # If specific dataset stats were critical, we would use them, but ImageNet is robust.
    normalization = A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if phase == "train":
        return A.Compose(
            [
                # 1. Center Crop to 64x64 (Target ROI + Context)
                A.CenterCrop(height=Config.CROP_SIZE, width=Config.CROP_SIZE),
                # 2. Geometric Augmentations (Rotation/Flip Invariance)
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                # 3. Normalization & Tensor Conversion
                normalization,
                ToTensorV2(),
            ]
        )
    else:
        # Validation / Test
        return A.Compose(
            [
                A.CenterCrop(height=Config.CROP_SIZE, width=Config.CROP_SIZE),
                normalization,
                ToTensorV2(),
            ]
        )


def prepare_folds(load_cached_data: bool = True) -> pd.DataFrame:
    """
    Loads training and validation metadata, merges them, and creates stratified folds.
    Uses caching to ensure consistency across runs.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    cache_path = os.path.join(Config.WORKING_DIR, "folds.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception:
            pass  # Fallback to recreation if load fails

    # 2. Create from scratch
    # Load original split metadata
    df_train_meta = pd.read_csv(Config.TRAIN_METADATA_PATH)
    df_val_meta = pd.read_csv(Config.VAL_METADATA_PATH)

    # Merge to recover full dataset for 5-Fold CV
    df = pd.concat([df_train_meta, df_val_meta], ignore_index=True)

    # Create Stratified Folds
    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )
    df["fold"] = -1

    for fold_idx, (_, val_idx) in enumerate(skf.split(df, df["label"])):
        df.loc[val_idx, "fold"] = fold_idx

    # 3. Save to cache
    df.to_parquet(cache_path, index=False)

    return df


def get_dataloaders(
    fold_id: int,
    batch_size: int = Config.BATCH_SIZE,
    num_workers: int = Config.NUM_WORKERS,
    load_cached_data: bool = True,
    debug: bool = Config.DEBUG,
):
    """
    Creates training and validation DataLoaders for a specific fold.
    """
    # Load data with folds
    df = prepare_folds(load_cached_data=load_cached_data)

    # Split into train/val based on fold_id
    df_train = df[df["fold"] != fold_id].reset_index(drop=True)
    df_val = df[df["fold"] == fold_id].reset_index(drop=True)

    # Handle Debug Mode
    if debug:
        df_train = df_train.head(Config.DEBUG_SAMPLES)
        df_val = df_val.head(Config.DEBUG_SAMPLES)

    # Create Datasets
    train_dataset = PathologyDataset(df_train, transform=get_transforms(phase="train"))
    val_dataset = PathologyDataset(df_val, transform=get_transforms(phase="valid"))

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch for stable BN
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def get_test_dataloader(
    batch_size: int = Config.BATCH_SIZE,
    num_workers: int = Config.NUM_WORKERS,
    debug: bool = Config.DEBUG,
):
    """
    Creates the test DataLoader.
    """
    df_test = pd.read_csv(Config.TEST_METADATA_PATH)

    if debug:
        df_test = df_test.head(Config.DEBUG_SAMPLES)

    test_dataset = PathologyDataset(
        df_test, transform=get_transforms(phase="test"), is_test=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return test_loader
