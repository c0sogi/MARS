import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import seed_everything

# Ensure the working directory exists
os.makedirs(Config.working_dir, exist_ok=True)


def crop_image_from_gray(img, tol=7):
    """
    Crops the black borders from a fundus image.
    If the image is too dark (all pixels < tol), returns the original image.
    """
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = gray_img > tol

        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:  # Image is too dark
            return img
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img
    return img


def get_transforms(phase: str):
    """
    Returns the Albumentations transformation pipeline for the specified phase.
    """
    if phase == "train":
        return A.Compose(
            [
                A.Resize(Config.image_size, Config.image_size),
                A.CLAHE(p=Config.clahe_prob),
                A.HorizontalFlip(p=Config.flip_prob),
                A.VerticalFlip(p=Config.flip_prob),
                A.Rotate(limit=Config.rotation_degrees, p=0.5),
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
                A.Resize(Config.image_size, Config.image_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )


def load_dataframe(csv_path, cache_name, load_cached_data=True):
    """
    Loads a dataframe from a CSV file, with caching support using Parquet.
    """
    cache_path = os.path.join(Config.working_dir, f"{cache_name}.parquet")

    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception:
            pass  # Fallback to loading from CSV if cache is corrupt

    # Load from source
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Source file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Save to cache
    try:
        df.to_parquet(cache_path, index=False)
    except Exception as e:
        print(f"Warning: Failed to save cache to {cache_path}. Error: {e}")

    return df


class RetinopathyDataset(Dataset):
    def __init__(self, df, transform=None, mode="train"):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata.
            transform (albumentations.Compose): Transformations to apply.
            mode (str): 'train' (returns label) or 'test' (no label).
        """
        self.df = df
        self.transform = transform
        self.mode = mode

        # Pre-check columns
        self.has_diagnosis = "diagnosis" in self.df.columns

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full file path
        # Note: file_path in metadata is relative (e.g., "train_images/id.png")
        # We join it with input_dir ("./input")
        image_path = os.path.join(Config.input_dir, row["file_path"])

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            # Fallback for missing images (should not happen based on metadata check)
            # Create a black image of default size
            image = np.zeros((Config.image_size, Config.image_size, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply Circle Crop
        if Config.use_circle_crop:
            image = crop_image_from_gray(image)

        # Apply Albumentations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Return data based on mode
        if self.mode in ["train", "val"]:
            label = row["diagnosis"] if self.has_diagnosis else 0
            # Return label as float for regression (MSE Loss)
            return image, torch.tensor(label, dtype=torch.float)
        else:
            # Test mode
            return image


def get_dataloaders(load_cached_data=True):
    """
    Factory function to create DataLoaders for train, val, and test sets.
    """
    seed_everything(Config.seed)

    # Load DataFrames
    train_df = load_dataframe(Config.train_csv_path, "train_df", load_cached_data)
    val_df = load_dataframe(Config.val_csv_path, "val_df", load_cached_data)
    test_df = load_dataframe(Config.test_csv_path, "test_df", load_cached_data)

    # Debug mode: subset data
    if Config.debug:
        train_df = train_df.head(Config.debug_samples)
        val_df = val_df.head(Config.debug_samples)
        test_df = test_df.head(Config.debug_samples)

    # Create Datasets
    train_dataset = RetinopathyDataset(
        train_df, transform=get_transforms("train"), mode="train"
    )

    val_dataset = RetinopathyDataset(
        val_df, transform=get_transforms("valid"), mode="val"
    )

    test_dataset = RetinopathyDataset(
        test_df, transform=get_transforms("test"), mode="test"
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
