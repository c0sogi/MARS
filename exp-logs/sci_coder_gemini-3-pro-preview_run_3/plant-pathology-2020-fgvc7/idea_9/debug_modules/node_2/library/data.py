import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold
from library.config import Config


def get_transforms(data, model_type):
    """
    Returns the albumentations augmentation pipeline based on the data split and model type.

    Args:
        data (str): 'train' or 'valid'.
        model_type (str): 'effnet' or 'maxvit'.

    Returns:
        A.Compose: The augmentation pipeline.
    """
    img_size = Config.get_image_size(model_type)

    if data == "train":
        return A.Compose(
            [
                A.Resize(img_size, img_size),
                # Strong Geometric Augmentation per strategy
                A.ShiftScaleRotate(
                    shift_limit=Config.AUG_SHIFT_LIMIT,
                    scale_limit=Config.AUG_SCALE_LIMIT,
                    rotate_limit=Config.AUG_ROTATE_LIMIT,
                    p=0.5,
                ),
                A.HorizontalFlip(p=0.5),
                # Vertical Flip is used in training but excluded in TTA
                A.VerticalFlip(p=0.5),
                A.Transpose(p=0.5),
                # Normalization
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    elif data == "valid":
        return A.Compose(
            [
                A.Resize(img_size, img_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        raise ValueError(f"Unknown data mode: {data}")


class AppleDataset(Dataset):
    """
    PyTorch Dataset for Apple Disease Detection.
    """

    def __init__(self, df, transform=None, output_label=True):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata (file_path, labels).
            transform (A.Compose, optional): Albumentations transforms.
            output_label (bool): Whether to return labels (True for train/val, False for test).
        """
        self.df = df
        self.transform = transform
        self.output_label = output_label
        self.file_paths = df["file_path"].values

        if self.output_label:
            # Extract labels in the order defined in Config
            self.labels = df[Config.CLASS_LABELS].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Construct full image path
        rel_path = self.file_paths[idx]
        img_path = os.path.join(Config.INPUT_DIR, rel_path)

        # Load image using OpenCV
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found at {img_path}")

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented["image"]

        # Return image and label (if requested)
        if self.output_label:
            label_vec = self.labels[idx]
            # Convert one-hot/probability vector to class index for CrossEntropy
            # The problem defines mutually exclusive classes for the primary task
            target = np.argmax(label_vec)
            return img, torch.tensor(target, dtype=torch.long)
        else:
            return img


def get_folds(load_cached_data=True, debug=False):
    """
    Loads the training data and creates stratified folds.
    Implements caching to 'folds.parquet'.

    Args:
        load_cached_data (bool): Whether to load from cache if available.
        debug (bool): If True, subsamples the data for debugging and does not cache.

    Returns:
        pd.DataFrame: DataFrame with a 'fold' column.
    """
    cache_path = os.path.join(Config.WORK_DIR, "folds.parquet")

    # 1. Debug Mode: Load raw, sample, split, return (no caching)
    if debug:
        df = pd.read_csv(Config.TRAIN_CSV)
        df = df.sample(
            n=min(len(df), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
        ).reset_index(drop=True)

        # Generate folds for the debug subset
        skf = StratifiedKFold(
            n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
        )
        df["fold"] = -1

        # Ensure stratify label exists
        if "stratify_label" not in df.columns:
            df["stratify_label"] = df[Config.CLASS_LABELS].idxmax(axis=1)

        for fold, (train_idx, val_idx) in enumerate(
            skf.split(df, df["stratify_label"])
        ):
            df.loc[val_idx, "fold"] = fold

        return df

    # 2. Normal Mode: Check Cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception:
            # If load fails, proceed to compute
            pass

    # 3. Compute Folds
    df = pd.read_csv(Config.TRAIN_CSV)

    # Ensure stratify label exists
    if "stratify_label" not in df.columns:
        df["stratify_label"] = df[Config.CLASS_LABELS].idxmax(axis=1)

    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )
    df["fold"] = -1

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["stratify_label"])):
        df.loc[val_idx, "fold"] = fold

    # 4. Save Cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df.to_parquet(cache_path, index=False)

    return df
