import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from library.config import Config
from library.utils import get_logger

logger = get_logger("dataset")


class AnimalDataset(Dataset):
    """
    Dataset class for Animal Classification.
    Reads images via OpenCV and applies Albumentations transforms.
    """

    def __init__(self, df, transform=None, mode="train"):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata (Id, file_path, Category).
            transform (A.Compose): Albumentations transforms.
            mode (str): 'train', 'val', or 'test'.
        """
        self.df = df
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full image path
        img_path = os.path.join(Config.INPUT_ROOT, row["file_path"])

        # Load image
        image = cv2.imread(img_path)
        if image is None:
            # Fallback or error handling for missing images
            raise FileNotFoundError(f"Image not found at {img_path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Return data based on mode
        if self.mode in ["train", "val"]:
            label = row["Category"]
            return image, torch.tensor(label, dtype=torch.long)
        else:
            # For test, return image and Id
            return image, row["Id"]


def get_transforms(mode="train"):
    """
    Returns Albumentations transforms for the specified mode.

    Args:
        mode (str): 'train', 'val', or 'test'.
    """
    if mode == "train":
        return A.Compose(
            [
                A.Resize(height=Config.IMG_SIZE, width=Config.IMG_SIZE),
                # Geometric Augmentations
                A.ShiftScaleRotate(
                    shift_limit=Config.AUG_SHIFT_LIMIT,
                    scale_limit=Config.AUG_SCALE_LIMIT,
                    rotate_limit=Config.AUG_ROTATE_LIMIT,
                    p=0.5,
                ),
                # Color/Intensity Augmentations
                A.RandomBrightnessContrast(
                    brightness_limit=Config.AUG_BRIGHTNESS_LIMIT,
                    contrast_limit=Config.AUG_CONTRAST_LIMIT,
                    p=0.5,
                ),
                # Regularization: CoarseDropout (Distributed Occlusion)
                A.CoarseDropout(
                    max_holes=Config.AUG_MAX_HOLES,
                    min_holes=Config.AUG_MIN_HOLES,
                    max_height=Config.AUG_HOLE_HEIGHT,
                    max_width=Config.AUG_HOLE_WIDTH,
                    min_height=Config.AUG_HOLE_HEIGHT,
                    min_width=Config.AUG_HOLE_WIDTH,
                    fill_value=0,
                    p=Config.AUG_DROPOUT_PROB,
                ),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        # Validation / Test transforms
        return A.Compose(
            [
                A.Resize(height=Config.IMG_SIZE, width=Config.IMG_SIZE),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


def _load_cached_df(csv_path, cache_name, load_cached_data=True):
    """
    Loads a DataFrame from a CSV file with caching support using Parquet.

    Args:
        csv_path (str): Path to the source CSV file.
        cache_name (str): Name of the cache file (e.g., 'train_processed.parquet').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    cache_path = os.path.join(Config.WORKING_DIR, cache_name)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            logger.info(f"Loading cached data from {cache_path}")
            return pd.read_parquet(cache_path)
        except Exception as e:
            logger.warning(
                f"Failed to load cache {cache_path}: {e}. Reloading from source."
            )

    # 2. Load from source and cache
    logger.info(f"Loading data from {csv_path}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Source metadata file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Ensure cache directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Save to cache
    try:
        df.to_parquet(cache_path, index=False)
        logger.info(f"Cached data to {cache_path}")
    except Exception as e:
        logger.warning(f"Failed to save cache to {cache_path}: {e}")

    return df


def get_dataloaders(debug=Config.DEBUG, load_cached_data=True):
    """
    Creates DataLoaders for train, validation, and test sets.
    Handles class balancing using WeightedRandomSampler.

    Args:
        debug (bool): If True, subsets data for debugging.
        load_cached_data (bool): Whether to use cached dataframes.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load DataFrames
    train_df = _load_cached_df(
        Config.TRAIN_METADATA, "train_processed.parquet", load_cached_data
    )
    val_df = _load_cached_df(
        Config.VAL_METADATA, "val_processed.parquet", load_cached_data
    )
    test_df = _load_cached_df(
        Config.TEST_METADATA, "test_processed.parquet", load_cached_data
    )

    # Handle Debug Mode
    if debug:
        logger.info(
            f"Debug mode enabled. Subsetting to {Config.DEBUG_SAMPLE_SIZE} samples."
        )
        train_df = train_df.iloc[: Config.DEBUG_SAMPLE_SIZE]
        val_df = val_df.iloc[: Config.DEBUG_SAMPLE_SIZE]
        test_df = test_df.iloc[: Config.DEBUG_SAMPLE_SIZE]

    # --- Weighted Random Sampler for Class Imbalance ---
    # Calculate weights for each class
    # Get all labels
    targets = train_df["Category"].values

    # Count frequency of each class
    class_counts = np.bincount(targets, minlength=Config.NUM_CLASSES)

    # Avoid division by zero for classes that might be missing in a small debug split
    class_counts = np.maximum(class_counts, 1)

    # Compute class weights (inverse frequency)
    class_weights = 1.0 / class_counts

    # Assign a weight to each sample based on its class
    sample_weights = class_weights[targets]

    # Convert to tensor
    sample_weights = torch.from_numpy(sample_weights).double()

    # Create Sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    logger.info("WeightedRandomSampler initialized for training set.")

    # --- Datasets ---
    train_dataset = AnimalDataset(
        train_df, transform=get_transforms(mode="train"), mode="train"
    )

    val_dataset = AnimalDataset(
        val_df, transform=get_transforms(mode="val"), mode="val"
    )

    test_dataset = AnimalDataset(
        test_df, transform=get_transforms(mode="test"), mode="test"
    )

    # --- DataLoaders ---
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        sampler=sampler,  # Use sampler, so shuffle must be False
        shuffle=False,
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

    logger.info(
        f"DataLoaders created: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}"
    )

    return train_loader, val_loader, test_loader
