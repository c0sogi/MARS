import os
import cv2
import torch
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from typing import Tuple, Optional, List

from library.config import Config


class AppleDataset(Dataset):
    """
    Custom Dataset for Apple Disease Detection.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        config: Config,
        transform: Optional[A.Compose] = None,
        test: bool = False,
    ):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata.
            config (Config): Configuration object.
            transform (A.Compose, optional): Albumentations transformations.
            test (bool): Whether this is a test dataset (no labels).
        """
        self.df = df
        self.config = config
        self.transform = transform
        self.test = test

        # Pre-compute full file paths
        # Metadata file_path is relative to input_dir
        self.file_paths = [
            os.path.join(config.input_dir, fp) for fp in df["file_path"].values
        ]

        # Extract labels if not test set
        if not self.test:
            self.labels = df[config.target_cols].values.astype(np.float32)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        file_path = self.file_paths[idx]

        # Load image using OpenCV
        image = cv2.imread(file_path)
        if image is None:
            # Handle missing images gracefully by returning a black image
            # This prevents the dataloader from crashing during training
            image = np.zeros(
                (self.config.img_size, self.config.img_size, 3), dtype=np.uint8
            )
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Default transform: Resize -> Normalize -> ToTensor
            default_transform = A.Compose(
                [
                    A.Resize(self.config.img_size, self.config.img_size),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ]
            )
            image = default_transform(image=image)["image"]

        if self.test:
            # Return dummy label for test set
            return image, torch.tensor(0.0)
        else:
            label = torch.tensor(self.labels[idx])
            return image, label


def get_transforms(config: Config, data: str = "train") -> A.Compose:
    """
    Generates Albumentations transformation pipelines.

    Args:
        config (Config): Configuration object.
        data (str): Type of data ('train', 'valid', 'test').

    Returns:
        A.Compose: Composed transformations.
    """
    if data == "train":
        return A.Compose(
            [
                A.Resize(config.img_size, config.img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.5
                ),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    elif data in ["valid", "test"]:
        return A.Compose(
            [
                A.Resize(config.img_size, config.img_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    else:
        raise ValueError(f"Unknown data mode: {data}")


def load_full_train_data(config: Config) -> pd.DataFrame:
    """
    Loads and concatenates train and validation metadata to form the full dataset
    for Cross-Validation.
    """
    if not os.path.exists(config.train_metadata_path) or not os.path.exists(
        config.val_metadata_path
    ):
        raise FileNotFoundError(
            "Metadata files not found. Please ensure metadata generation was successful."
        )

    train_df = pd.read_csv(config.train_metadata_path)
    val_df = pd.read_csv(config.val_metadata_path)

    # Concatenate to use all available labeled data for CV
    full_df = pd.concat([train_df, val_df], ignore_index=True)

    if config.debug:
        full_df = full_df.sample(
            n=min(100, len(full_df)), random_state=config.seed
        ).reset_index(drop=True)

    return full_df


def get_loaders(
    fold: int, config: Config, df: Optional[pd.DataFrame] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Creates train and validation DataLoaders for a specific fold using Stratified K-Fold.

    Args:
        fold (int): The fold index (0 to n_folds-1).
        config (Config): Configuration object.
        df (pd.DataFrame, optional): The dataframe to split. If None, loads full combined data.

    Returns:
        train_loader (DataLoader): Loader for training subset.
        val_loader (DataLoader): Loader for validation subset.
    """
    if df is None:
        df = load_full_train_data(config)

    # Ensure stratify label exists
    if "stratify_label" not in df.columns:
        df["stratify_label"] = df[config.target_cols].idxmax(axis=1)

    # Initialize Stratified K-Fold
    skf = StratifiedKFold(
        n_splits=config.n_folds, shuffle=True, random_state=config.seed
    )

    # Get indices for the specific fold
    # list(skf.split) is deterministic given the seed
    splits = list(skf.split(df, df["stratify_label"]))

    if fold >= len(splits):
        raise ValueError(f"Fold {fold} out of range for {config.n_folds} splits.")

    train_idx, val_idx = splits[fold]

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    # Create Datasets
    train_dataset = AppleDataset(
        train_df, config, transform=get_transforms(config, "train")
    )
    val_dataset = AppleDataset(
        val_df, config, transform=get_transforms(config, "valid")
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,  # Important for batch normalization and mixup stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def get_test_loader(config: Config) -> Tuple[DataLoader, pd.DataFrame]:
    """
    Creates the test DataLoader.

    Args:
        config (Config): Configuration object.

    Returns:
        test_loader (DataLoader): Loader for test data.
        test_df (pd.DataFrame): DataFrame containing test metadata.
    """
    if not os.path.exists(config.test_metadata_path):
        raise FileNotFoundError(
            f"Test metadata not found at {config.test_metadata_path}"
        )

    test_df = pd.read_csv(config.test_metadata_path)

    if config.debug:
        test_df = test_df.sample(
            n=min(50, len(test_df)), random_state=config.seed
        ).reset_index(drop=True)

    test_dataset = AppleDataset(
        test_df, config, transform=get_transforms(config, "test"), test=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return test_loader, test_df
