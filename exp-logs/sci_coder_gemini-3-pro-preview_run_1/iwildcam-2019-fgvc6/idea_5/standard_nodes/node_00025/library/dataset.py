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
    Returns the data augmentation pipeline for the specified phase.

    Args:
        phase (str): 'train', 'val', or 'test'.

    Returns:
        A.Compose: Albumentations composition of transforms.
    """
    if phase == "train":
        return A.Compose(
            [
                # High resolution input with aggressive cropping to learn local features
                # Scale 0.08 forces model to look at small parts, critical for small animals
                A.RandomResizedCrop(
                    size=(Config.IMG_SIZE, Config.IMG_SIZE), scale=(0.08, 1.0)
                ),
                A.HorizontalFlip(p=0.5),
                # ColorJitter to handle variable lighting conditions in camera traps
                A.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.5
                ),
                # Blur to simulate out-of-focus shots common in motion-triggered cameras
                A.GaussianBlur(p=0.3),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        # Deterministic resize for validation and testing
        return A.Compose(
            [
                A.Resize(height=Config.IMG_SIZE, width=Config.IMG_SIZE),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


def load_metadata(mode: str, load_cached_data: bool = True, sample_size: int = None):
    """
    Loads metadata for the dataset, implementing caching logic.

    Args:
        mode (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.
        sample_size (int, optional): If provided, subsets the data for debugging.

    Returns:
        pd.DataFrame: The loaded metadata.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    cache_path = os.path.join(Config.WORKING_DIR, f"{mode}_meta.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            # Apply sampling if requested even on cached data
            if sample_size is not None and sample_size < len(df):
                df = df.iloc[:sample_size]
            return df
        except Exception:
            # If load fails, fall through to process from scratch
            pass

    # 2. Process from scratch
    if mode == "train":
        source_path = Config.TRAIN_META
    elif mode == "val":
        source_path = Config.VAL_META
    elif mode == "test":
        source_path = Config.TEST_META
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Metadata file not found: {source_path}")

    df = pd.read_csv(source_path)

    # Save to cache for future runs
    try:
        df.to_parquet(cache_path, index=False)
    except Exception as e:
        print(f"Warning: Failed to save cache to {cache_path}: {e}")

    # Apply sampling
    if sample_size is not None and sample_size < len(df):
        df = df.iloc[:sample_size]

    return df


class AnimalDataset(Dataset):
    def __init__(
        self,
        mode: str,
        transform=None,
        load_cached_data: bool = True,
        sample_size: int = None,
    ):
        """
        Args:
            mode (str): 'train', 'val', or 'test'.
            transform (A.Compose, optional): Albumentations transforms.
            load_cached_data (bool): Whether to use cached metadata.
            sample_size (int, optional): Limit dataset size for debugging.
        """
        self.mode = mode
        self.transform = transform
        self.df = load_metadata(mode, load_cached_data, sample_size)
        self.root_dir = Config.INPUT_DIR

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full image path
        # Metadata 'file_path' is relative to INPUT_DIR (e.g., "train_images/xyz.jpg")
        img_path = os.path.join(self.root_dir, row["file_path"])

        # Load image
        image = cv2.imread(img_path)
        if image is None:
            # Handle missing images gracefully by returning a black image
            # This prevents training from crashing due to a single corrupt file
            image = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Fallback to tensor conversion if no transform provided
            image = ToTensorV2()(image=image)["image"]

        # Extract labels
        # Test set might not have 'Category', handle accordingly
        if "Category" in row:
            species_label = int(row["Category"])
            # Detection label: 0 if empty (class 0), 1 if any animal (class > 0)
            detection_label = 0 if species_label == 0 else 1
        else:
            species_label = -1  # Dummy value for test
            detection_label = -1

        return {
            "image": image,
            "species_label": torch.tensor(species_label, dtype=torch.long),
            "detection_label": torch.tensor(
                detection_label, dtype=torch.float
            ),  # Float for BCEWithLogitsLoss
            "id": row["Id"],
        }
