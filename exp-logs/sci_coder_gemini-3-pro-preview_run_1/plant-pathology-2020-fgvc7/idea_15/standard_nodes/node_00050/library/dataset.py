import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold
from library.config import CFG


def get_transforms(data):
    """
    Returns the Albumentations transform pipeline based on the data mode.

    Args:
        data (str): 'train', 'valid', or 'test'.

    Returns:
        A.Compose: The composed transform pipeline.
    """
    if data == "train":
        return A.Compose(
            [
                A.Resize(CFG.img_size, CFG.img_size),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5
                ),
                A.RandomBrightnessContrast(p=0.5),
                # Explicit inclusion of flips as per strategy (Lesson 30)
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    elif data == "valid" or data == "test":
        return A.Compose(
            [
                A.Resize(CFG.img_size, CFG.img_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    else:
        raise ValueError(f"Unknown data mode: {data}")


class AppleDataset(Dataset):
    """
    Custom Dataset for Apple Disease Detection.
    """

    def __init__(self, df, transform=None):
        self.df = df
        self.file_names = df["file_path"].values
        self.image_ids = df["image_id"].values
        self.transform = transform

        # Determine if targets are present
        self.has_labels = set(CFG.target_cols).issubset(df.columns)
        if self.has_labels:
            self.labels = df[CFG.target_cols].values
        else:
            self.labels = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Construct full file path
        # file_path in metadata is relative (e.g., "images/Train_0.jpg")
        # CFG.input_dir is "./input"
        rel_path = self.file_names[idx]
        image_path = os.path.join(CFG.input_dir, rel_path)

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        image_id = self.image_ids[idx]

        if self.has_labels:
            label = torch.tensor(self.labels[idx]).float()
            return image, label, image_id
        else:
            # Return dummy label for test set to maintain signature consistency
            dummy_label = torch.zeros(len(CFG.target_cols)).float()
            return image, dummy_label, image_id


def prepare_folds(load_cached_data=True):
    """
    Loads training and validation metadata, combines them into a single dataset,
    and generates stratified folds for Phase 1 calibration.

    Implements caching using Parquet format.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: The full training dataframe with a 'fold' column.
    """
    cache_path = os.path.join(CFG.working_dir, "folds.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception:
            # If load fails, proceed to compute
            pass

    # 2. Compute from scratch
    # Ensure working directory exists
    os.makedirs(CFG.working_dir, exist_ok=True)

    # Load metadata
    if not os.path.exists(CFG.train_metadata_path) or not os.path.exists(
        CFG.val_metadata_path
    ):
        raise FileNotFoundError(
            "Metadata files not found. Ensure metadata generation is complete."
        )

    train_meta = pd.read_csv(CFG.train_metadata_path)
    val_meta = pd.read_csv(CFG.val_metadata_path)

    # Combine to form full training set (100% data)
    df = pd.concat([train_meta, val_meta], axis=0).reset_index(drop=True)

    # Create Stratified Folds
    skf = StratifiedKFold(n_splits=CFG.n_folds, shuffle=True, random_state=CFG.seed)

    # Identify stratification label
    if "stratify_label" in df.columns:
        y = df["stratify_label"]
    else:
        # Fallback: argmax of target columns
        y = df[CFG.target_cols].idxmax(axis=1)

    df["fold"] = -1
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, y)):
        df.loc[val_idx, "fold"] = fold

    # 3. Save to cache
    df.to_parquet(cache_path, index=False)

    return df
