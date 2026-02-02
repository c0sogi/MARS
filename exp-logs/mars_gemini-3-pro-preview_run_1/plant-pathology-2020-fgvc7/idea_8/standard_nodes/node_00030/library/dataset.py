import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from library.config import Config


def get_transforms(data="train"):
    """
    Returns the Albumentations transformation pipeline.

    Args:
        data (str): 'train' for training augmentations, 'valid' or 'test' for inference.
    """
    if data == "train":
        return A.Compose(
            [
                A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                A.HorizontalFlip(p=0.5),
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
    elif data == "valid" or data == "test":
        return A.Compose(
            [
                A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    else:
        raise ValueError(f"Unknown data mode: {data}")


def load_dataframes(load_cached_data=True):
    """
    Loads and processes metadata dataframes with caching.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (train_df, test_df)
    """
    # Define cache paths
    train_cache_path = os.path.join(Config.WORKING_DIR, "train_df.parquet")
    test_cache_path = os.path.join(Config.WORKING_DIR, "test_df.parquet")

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 1. Try to load cached data
    if (
        load_cached_data
        and os.path.exists(train_cache_path)
        and os.path.exists(test_cache_path)
    ):
        try:
            train_df = pd.read_parquet(train_cache_path)
            test_df = pd.read_parquet(test_cache_path)
            return train_df, test_df
        except Exception:
            pass

    # 2. Compute/Process from scratch
    # Load metadata
    if not os.path.exists(Config.TRAIN_METADATA_PATH):
        raise FileNotFoundError(
            f"Train metadata not found at {Config.TRAIN_METADATA_PATH}"
        )

    df_train_part = pd.read_csv(Config.TRAIN_METADATA_PATH)
    df_val_part = pd.read_csv(Config.VAL_METADATA_PATH)
    df_test = pd.read_csv(Config.TEST_METADATA_PATH)

    # Merge train and val if configured to use full data
    if Config.USE_FULL_DATA:
        df_train = pd.concat([df_train_part, df_val_part], axis=0).reset_index(
            drop=True
        )
    else:
        df_train = df_train_part

    # 3. Save to cache
    df_train.to_parquet(train_cache_path, index=False)
    df_test.to_parquet(test_cache_path, index=False)

    return df_train, df_test


class AppleDataset(Dataset):
    def __init__(self, df, transforms=None, root_dir=Config.INPUT_DIR, is_test=False):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe.
            transforms (albumentations.Compose): Transformations to apply.
            root_dir (str): Root directory for images (usually input dir).
            is_test (bool): If True, does not look for target columns.
        """
        self.df = df
        self.transforms = transforms
        self.root_dir = root_dir
        self.is_test = is_test
        self.class_labels = Config.CLASS_LABELS

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct image path
        # file_path in metadata is relative to input (e.g., "images/Train_0.jpg")
        img_path = os.path.join(self.root_dir, row["file_path"])

        # Read image
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        # Get targets
        if self.is_test:
            # Return dummy targets for test set
            targets = torch.zeros(len(self.class_labels), dtype=torch.float32)
        else:
            # Extract targets from dataframe columns
            # Assumes columns exist and are float/int
            label_values = row[self.class_labels].values.astype(np.float32)
            targets = torch.tensor(label_values, dtype=torch.float32)

        return image, targets


def get_loaders(load_cached_data=True, batch_size=Config.BATCH_SIZE):
    """
    Creates and returns DataLoaders for training and testing.

    Args:
        load_cached_data (bool): Whether to use cached dataframes.
        batch_size (int): Batch size for loaders.

    Returns:
        tuple: (train_loader, test_loader)
    """
    # Load dataframes
    train_df, test_df = load_dataframes(load_cached_data=load_cached_data)

    # Create datasets
    train_dataset = AppleDataset(
        df=train_df,
        transforms=get_transforms("train"),
        root_dir=Config.INPUT_DIR,
        is_test=False,
    )

    test_dataset = AppleDataset(
        df=test_df,
        transforms=get_transforms("test"),
        root_dir=Config.INPUT_DIR,
        is_test=True,
    )

    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, test_loader
