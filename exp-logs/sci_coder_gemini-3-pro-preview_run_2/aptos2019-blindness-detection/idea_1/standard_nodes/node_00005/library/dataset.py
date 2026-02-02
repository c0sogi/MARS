import os
import cv2
import torch
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from library.config import Config


def get_transforms(phase: str):
    """
    Creates the transformation pipeline for the given phase.

    Args:
        phase (str): One of 'train', 'valid', 'test'.

    Returns:
        albumentations.Compose: The transformation pipeline.
    """
    # ImageNet normalization statistics
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    transforms = []

    # Resize is applied to all splits
    transforms.append(A.Resize(height=Config.IMG_SIZE, width=Config.IMG_SIZE))

    if phase == "train":
        # Augmentations for training
        transforms.append(A.HorizontalFlip(p=0.5))
        # Aggressive augmentations to mitigate overfitting (Cite solution_lesson_node_00002)
        transforms.append(A.VerticalFlip(p=0.5))
        transforms.append(A.Rotate(limit=30, p=0.5))
        transforms.append(A.RandomBrightnessContrast(p=0.5))

    # Normalize and convert to tensor
    transforms.append(A.Normalize(mean=mean, std=std))
    transforms.append(ToTensorV2())

    return A.Compose(transforms)


class RetinopathyDataset(Dataset):
    """
    Dataset class for loading and preprocessing Diabetic Retinopathy images.
    """

    def __init__(self, df: pd.DataFrame, transforms=None, mode: str = "train"):
        """
        Args:
            df (pd.DataFrame): DataFrame containing image metadata.
            transforms (albumentations.Compose, optional): Transformations to apply.
            mode (str): 'train', 'valid', or 'test'.
        """
        self.df = df
        self.transforms = transforms
        self.mode = mode

        # Validation of dataframe columns
        required_cols = ["file_path"]
        if self.mode != "test":
            required_cols.append("diagnosis")

        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"DataFrame missing required column: {col}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct absolute file path
        # Metadata contains relative paths (e.g., "train_images/id.png")
        image_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # Load image using OpenCV
        image = cv2.imread(image_path)

        # Handle potential loading errors
        if image is None:
            # Create a blank image to prevent crashing
            image = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
        else:
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transformations
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]
        else:
            # Fallback manual conversion if no transforms provided
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        # Return format depends on mode
        if self.mode == "test":
            # For inference, we need the image and the ID to map predictions
            return image, row["id_code"]
        else:
            # For training/validation, we need the image and the regression target
            label = torch.tensor(row["diagnosis"], dtype=torch.float)
            return image, label


def load_data(load_cached_data: bool = True):
    """
    Loads dataset metadata, handling caching and debug sampling.

    Args:
        load_cached_data (bool): If True, attempts to load from parquet cache.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Define cache filenames based on debug state to avoid mixing full/debug data
    suffix = "_debug" if Config.DEBUG else ""
    train_cache = os.path.join(Config.WORKING_DIR, f"train_df{suffix}.parquet")
    val_cache = os.path.join(Config.WORKING_DIR, f"val_df{suffix}.parquet")
    test_cache = os.path.join(Config.WORKING_DIR, f"test_df{suffix}.parquet")

    # 1. Try to load from cache
    if load_cached_data:
        if (
            os.path.exists(train_cache)
            and os.path.exists(val_cache)
            and os.path.exists(test_cache)
        ):
            try:
                train_df = pd.read_parquet(train_cache)
                val_df = pd.read_parquet(val_cache)
                test_df = pd.read_parquet(test_cache)
                return train_df, val_df, test_df
            except Exception:
                # If load fails, proceed to re-create
                pass

    # 2. Load from source metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    # Apply Debug Sampling
    if Config.DEBUG:
        train_df = train_df.iloc[: Config.DEBUG_SAMPLE_SIZE]
        val_df = val_df.iloc[: Config.DEBUG_SAMPLE_SIZE]
        test_df = test_df.iloc[: Config.DEBUG_SAMPLE_SIZE]

    # 3. Save to cache
    train_df.to_parquet(train_cache, index=False)
    val_df.to_parquet(val_cache, index=False)
    test_df.to_parquet(test_cache, index=False)

    return train_df, val_df, test_df
