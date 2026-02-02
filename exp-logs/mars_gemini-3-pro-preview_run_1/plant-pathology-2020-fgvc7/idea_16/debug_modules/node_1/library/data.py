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


class AppleDataset(Dataset):
    """
    PyTorch Dataset for Apple Disease Detection.
    """

    def __init__(self, df, transforms=None, mode="train"):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe containing file paths and labels.
            transforms (albumentations.Compose): Transformations to apply.
            mode (str): 'train' (returns image, label) or 'test' (returns image, image_id).
        """
        self.df = df
        self.transforms = transforms
        self.mode = mode
        self.image_ids = df["image_id"].values
        # file_path in metadata is relative to input dir (e.g. "images/Train_0.jpg")
        self.file_paths = df["file_path"].values

        if self.mode == "train":
            self.labels = df[Config.TARGET_COLS].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Construct full path
        image_path = os.path.join(Config.INPUT_DIR, self.file_paths[idx])

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        if self.mode == "train":
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return image, label
        else:
            image_id = self.image_ids[idx]
            return image, image_id


def get_transforms(data="train"):
    """
    Returns albumentations transforms for training or validation/testing.
    """
    if data == "train":
        return A.Compose(
            [
                A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5
                ),
                A.RandomBrightnessContrast(p=0.5),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )


def calculate_class_weights(load_cached_data=True):
    """
    Calculates class weights to handle imbalance using inverse frequency.
    Implements caching to .npy file.

    Returns:
        torch.Tensor: Weights for each class on the configured device.
    """
    cache_path = os.path.join(Config.CACHE_DIR, "class_weights.npy")

    # 1. Try to load cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            weights = np.load(cache_path)
            return torch.tensor(weights, dtype=torch.float32).to(Config.DEVICE)
        except Exception:
            pass  # Fallback to calculation

    # 2. Calculate from scratch
    # Load all available training data (train + val metadata) to get global distribution
    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
    df_val = pd.read_csv(Config.VAL_METADATA_PATH)
    df_full = pd.concat([df_train, df_val], ignore_index=True)

    # Calculate counts for each target column
    # Assuming one-hot or soft labels, we sum the columns
    counts = df_full[Config.TARGET_COLS].sum().values
    total_samples = len(df_full)
    num_classes = len(Config.TARGET_COLS)

    # Inverse frequency weights: N / (K * count)
    # Add small epsilon to avoid division by zero if a class is missing (unlikely)
    weights = total_samples / (num_classes * (counts + 1e-6))

    # Normalize weights so they sum to num_classes (optional, but keeps scale consistent)
    # weights = weights / weights.sum() * num_classes

    # Save to cache
    np.save(cache_path, weights)

    return torch.tensor(weights, dtype=torch.float32).to(Config.DEVICE)


def get_loaders(mode="calibration", fold=0, load_cached_data=True):
    """
    Generates DataLoaders for training, validation, or testing.

    Args:
        mode (str): 'calibration' (K-Fold), 'production' (Full Data), or 'test'.
        fold (int): Fold index for calibration mode (0 to N_FOLDS-1).
        load_cached_data (bool): Whether to use cached folds/weights.

    Returns:
        tuple: (train_loader, val_loader) or (test_loader) depending on mode.
               val_loader is None in 'production' mode.
    """

    if mode == "test":
        df_test = pd.read_csv(Config.TEST_METADATA_PATH)
        test_dataset = AppleDataset(
            df_test, transforms=get_transforms(data="valid"), mode="test"
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )
        return test_loader

    # Load all training data
    df_train_meta = pd.read_csv(Config.TRAIN_METADATA_PATH)
    df_val_meta = pd.read_csv(Config.VAL_METADATA_PATH)
    df_full = pd.concat([df_train_meta, df_val_meta], ignore_index=True)

    if mode == "calibration":
        # Stratified K-Fold Split
        skf = StratifiedKFold(
            n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEEDS[0]
        )

        # We need a single label for stratification. The metadata has 'stratify_label'.
        # If not, we derive it.
        if "stratify_label" in df_full.columns:
            y = df_full["stratify_label"]
        else:
            y = df_full[Config.TARGET_COLS].idxmax(axis=1)

        # Get indices for the requested fold
        # list(skf.split) returns generator, we convert to list and pick the fold
        splits = list(skf.split(df_full, y))
        train_idx, val_idx = splits[fold]

        df_train = df_full.iloc[train_idx].reset_index(drop=True)
        df_val = df_full.iloc[val_idx].reset_index(drop=True)

        train_dataset = AppleDataset(
            df_train, transforms=get_transforms(data="train"), mode="train"
        )
        val_dataset = AppleDataset(
            df_val, transforms=get_transforms(data="valid"), mode="train"
        )

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
        )

        return train_loader, val_loader

    elif mode == "production":
        # Use 100% of data for training
        train_dataset = AppleDataset(
            df_full, transforms=get_transforms(data="train"), mode="train"
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )

        return train_loader, None

    else:
        raise ValueError(f"Unknown mode: {mode}")
