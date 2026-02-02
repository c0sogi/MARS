import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold
import library.config as cfg


def crop_image_from_gray(img, tol=7):
    """
    Crops the black borders from a fundus image to focus on the retina.
    Converts to grayscale, applies a threshold to find the mask,
    and crops to the bounding box of the mask.
    """
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
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


def get_dataframes(load_cached_data=True):
    """
    Loads train, validation, and test metadata.
    Implements caching using Parquet files to speed up subsequent runs.
    """
    # Define cache paths
    train_cache = os.path.join(cfg.WORKING_DIR, "train_df.parquet")
    val_cache = os.path.join(cfg.WORKING_DIR, "val_df.parquet")
    test_cache = os.path.join(cfg.WORKING_DIR, "test_df.parquet")

    # Ensure working directory exists
    os.makedirs(cfg.WORKING_DIR, exist_ok=True)

    # Helper to load or create
    def load_or_create(csv_path, cache_path):
        if load_cached_data and os.path.exists(cache_path):
            return pd.read_parquet(cache_path)
        else:
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"Metadata file not found: {csv_path}")
            df = pd.read_csv(csv_path)
            # Save to cache
            df.to_parquet(cache_path, index=False)
            return df

    train_df = load_or_create(cfg.TRAIN_CSV, train_cache)
    val_df = load_or_create(cfg.VAL_CSV, val_cache)
    test_df = load_or_create(cfg.TEST_CSV, test_cache)

    # Debug mode: sample data
    if cfg.DEBUG:
        train_df = train_df.head(cfg.DEBUG_SAMPLE_SIZE)
        val_df = val_df.head(cfg.DEBUG_SAMPLE_SIZE)
        test_df = test_df.head(cfg.DEBUG_SAMPLE_SIZE)

    return train_df, val_df, test_df


class RetinopathyDataset(Dataset):
    def __init__(self, df, transform=None, mode="train"):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata.
            transform (albumentations.Compose): Augmentation pipeline.
            mode (str): 'train', 'val', or 'test'.
        """
        self.df = df
        self.transform = transform
        self.mode = mode
        self.input_dir = cfg.INPUT_DIR

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full image path
        img_path = os.path.join(self.input_dir, row["file_path"])

        # Load image
        image = cv2.imread(img_path)
        if image is None:
            # Fallback for missing images (should not happen given metadata check)
            # Create a black image of default size
            image = np.zeros((cfg.IMG_SIZE, cfg.IMG_SIZE, 3), dtype=np.uint8)
        else:
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Apply Circle Crop
            image = crop_image_from_gray(image)

        # Apply Augmentations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Handle Labels
        if self.mode in ["train", "val"]:
            label = row["diagnosis"]
            # Return label as float for regression (MSE Loss)
            return image, torch.tensor(label, dtype=torch.float)
        else:
            # For test, we might not have diagnosis, or we just need ID
            return image, row["id_code"]


def get_transforms(img_size):
    """
    Returns augmentation pipelines for training and validation/testing.
    """
    # ImageNet normalization statistics
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transforms = A.Compose(
        [
            A.Resize(height=img_size, width=img_size),
            A.CLAHE(p=1.0),  # Cite solution_lesson_node_00012: Improve contrast
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            # Rotation up to 30 degrees
            A.Rotate(limit=30, p=0.5),
            # Random Brightness and Contrast
            A.RandomBrightnessContrast(p=0.5),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=img_size, width=img_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )

    return train_transforms, val_transforms


def get_loaders(fold=0, load_cached_data=True):
    """
    Creates and returns DataLoaders for train, val, and test sets.
    Supports 5-Fold Cross Validation.
    """
    # Load DataFrames
    train_df_orig, val_df_orig, test_df = get_dataframes(
        load_cached_data=load_cached_data
    )

    # Merge train and val to create a full dataset for CV
    full_df = pd.concat([train_df_orig, val_df_orig]).reset_index(drop=True)

    # Create Stratified K-Fold
    skf = StratifiedKFold(n_splits=cfg.NUM_FOLDS, shuffle=True, random_state=cfg.SEED)

    # Get indices for the specific fold
    train_idx, val_idx = list(skf.split(full_df, full_df["diagnosis"]))[fold]

    train_df = full_df.iloc[train_idx].reset_index(drop=True)
    val_df = full_df.iloc[val_idx].reset_index(drop=True)

    # Get Transforms
    train_tf, val_tf = get_transforms(cfg.IMG_SIZE)

    # Create Datasets
    train_dataset = RetinopathyDataset(train_df, transform=train_tf, mode="train")
    val_dataset = RetinopathyDataset(val_df, transform=val_tf, mode="val")
    test_dataset = RetinopathyDataset(test_df, transform=val_tf, mode="test")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
