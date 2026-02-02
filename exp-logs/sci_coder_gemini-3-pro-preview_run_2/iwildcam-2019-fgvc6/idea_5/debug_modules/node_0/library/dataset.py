import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from library.config import Config


def load_and_cache_data(
    csv_path: str, cache_name: str, load_cached_data: bool = True
) -> pd.DataFrame:
    """
    Loads data from CSV or Parquet cache.

    Args:
        csv_path: Path to the original CSV file.
        cache_name: Name of the cache file (e.g., 'train_cache.parquet').
        load_cached_data: Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: The loaded dataframe.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    cache_path = os.path.join(Config.WORKING_DIR, cache_name)

    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception:
            # If cache load fails, fall back to CSV
            pass

    # Load from CSV
    df = pd.read_csv(csv_path)

    # Ensure Id is string to match file paths
    if "Id" in df.columns:
        df["Id"] = df["Id"].astype(str)

    # Cache the result
    df.to_parquet(cache_path, index=False)

    return df


class AnimalDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transforms=None):
        """
        Args:
            df: DataFrame containing metadata (must have 'file_path' and 'Category').
            transforms: Albumentations transforms to apply.
        """
        self.df = df
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full image path
        # Metadata file_path is relative (e.g., "train_images/id.jpg")
        img_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # Load image
        image = cv2.imread(img_path)

        if image is None:
            # Fallback for missing images (though EDA showed none)
            # Create a black image of the target size
            image = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8)
        else:
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        # Get label
        label = row["Category"]

        return image, torch.tensor(label, dtype=torch.long)


def get_transforms(data: str = "train"):
    """
    Returns the albumentations transform pipeline.

    Args:
        data: 'train' or 'valid'/'test'.
    """
    if data == "train":
        return A.Compose(
            [
                A.Resize(height=Config.IMAGE_SIZE, width=Config.IMAGE_SIZE),
                A.HorizontalFlip(p=0.5),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(height=Config.IMAGE_SIZE, width=Config.IMAGE_SIZE),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ]
        )


def create_dataloaders(load_cached_data: bool = True):
    """
    Creates DataLoaders for train, validation, and test sets.
    Implements WeightedRandomSampler for the training set to handle class imbalance.

    Args:
        load_cached_data: Whether to use cached dataframes.

    Returns:
        train_loader, val_loader, test_loader
    """
    # 1. Load DataFrames
    df_train = load_and_cache_data(
        Config.TRAIN_METADATA, "train_meta.parquet", load_cached_data
    )
    df_val = load_and_cache_data(
        Config.VAL_METADATA, "val_meta.parquet", load_cached_data
    )
    df_test = load_and_cache_data(
        Config.TEST_METADATA, "test_meta.parquet", load_cached_data
    )

    # 2. Handle Debug Mode
    if Config.DEBUG:
        df_train = df_train.iloc[: Config.DEBUG_SAMPLE_SIZE].reset_index(drop=True)
        df_val = df_val.iloc[: Config.DEBUG_SAMPLE_SIZE].reset_index(drop=True)
        df_test = df_test.iloc[: Config.DEBUG_SAMPLE_SIZE].reset_index(drop=True)

    # 3. Create Datasets
    train_dataset = AnimalDataset(df_train, transforms=get_transforms("train"))
    val_dataset = AnimalDataset(df_val, transforms=get_transforms("valid"))
    test_dataset = AnimalDataset(df_test, transforms=get_transforms("valid"))

    # 4. Create WeightedRandomSampler for Training
    # Calculate weights for each class
    class_counts = df_train["Category"].value_counts().sort_index()
    # Add small epsilon to avoid division by zero if a class is missing in debug mode
    class_weights = 1.0 / (class_counts + 1e-6)

    # Map weights to each sample
    # We use map to create a weight for every row in the dataframe
    sample_weights = df_train["Category"].map(class_weights).fillna(0).values
    sample_weights = torch.from_numpy(sample_weights).double()

    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    # 5. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        sampler=sampler,  # Sampler is mutually exclusive with shuffle
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
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
