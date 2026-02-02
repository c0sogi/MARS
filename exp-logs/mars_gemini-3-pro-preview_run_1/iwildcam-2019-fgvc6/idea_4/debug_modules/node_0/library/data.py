import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import get_logger

# Initialize logger
logger = get_logger("data_module")


def load_and_cache_df(
    csv_path: str, cache_name: str, load_cached_data: bool = True
) -> pd.DataFrame:
    """
    Loads a dataframe from a CSV file with a caching mechanism using Parquet.

    Args:
        csv_path (str): Path to the source CSV file.
        cache_name (str): Name for the cached file (without extension).
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: The loaded dataframe.
    """
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{cache_name}.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            logger.info(f"Loading cached dataframe from {cache_path}")
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}. Reloading from source.")

    # 2. Load from source
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Source file not found: {csv_path}")

    logger.info(f"Loading dataframe from {csv_path}")
    df = pd.read_csv(csv_path)

    # 3. Save to cache
    try:
        df.to_parquet(cache_path, index=False)
        logger.info(f"Cached dataframe to {cache_path}")
    except Exception as e:
        logger.warning(f"Failed to save cache: {e}")

    return df


class AnimalDataset(Dataset):
    def __init__(
        self, df: pd.DataFrame, transforms: A.Compose = None, mode: str = "train"
    ):
        """
        Args:
            df (pd.DataFrame): Dataframe containing metadata.
            transforms (A.Compose): Albumentations transforms.
            mode (str): 'train', 'val', or 'test'. Determines return values.
        """
        self.df = df
        self.transforms = transforms
        self.mode = mode
        self.input_dir = Config.INPUT_DIR

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full image path
        # Metadata file_path is relative to input directory (e.g., "train_images/id.jpg")
        img_path = os.path.join(self.input_dir, row["file_path"])

        # Read image using OpenCV
        image = cv2.imread(img_path)

        # Handle missing or corrupt images
        if image is None:
            # In a real scenario, we might log this or skip.
            # For this task, we assume data integrity but provide a fallback black image to prevent crash.
            # 224x224 black image
            image = np.zeros((Config.INPUT_SIZE, Config.INPUT_SIZE, 3), dtype=np.uint8)
        else:
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        # Return based on mode
        if self.mode == "test":
            # For test, we need the Id to map predictions
            return image, row["Id"]
        else:
            # For train/val, we return the label
            # Ensure label is long tensor for CrossEntropy/FocalLoss
            label = torch.tensor(row["Category"], dtype=torch.long)
            return image, label


def get_transforms(mode: str = "train") -> A.Compose:
    """
    Returns the Albumentations transform pipeline for the specified mode.

    Args:
        mode (str): 'train' or 'val'/'test'.

    Returns:
        A.Compose: Composed transforms.
    """
    img_size = Config.INPUT_SIZE

    # Standard ImageNet normalization
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if mode == "train":
        return A.Compose(
            [
                # RandomResizedCrop: Essential for scale invariance and "zooming in" on animals
                A.RandomResizedCrop(
                    height=img_size,
                    width=img_size,
                    scale=(0.08, 1.0),
                    ratio=(0.75, 1.333),
                    p=1.0,
                ),
                # Horizontal Flip: Standard augmentation
                A.HorizontalFlip(p=0.5),
                # ColorJitter: Robustness to lighting conditions
                A.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.4
                ),
                # GaussianBlur: Slight blurring to handle focus issues / noise
                A.GaussianBlur(blur_limit=(3, 7), p=0.1),
                # Normalize and convert to tensor
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    else:
        # Validation/Test: Resize (squish) to input size, then normalize
        return A.Compose(
            [
                A.Resize(height=img_size, width=img_size),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )


def get_loaders(debug: bool = Config.DEBUG, load_cached_data: bool = True):
    """
    Creates and returns DataLoaders for train, validation, and test sets.

    Args:
        debug (bool): If True, subsamples the dataset for debugging.
        load_cached_data (bool): Whether to use cached dataframes.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # 1. Load DataFrames
    df_train = load_and_cache_df(Config.TRAIN_META_PATH, "train_meta", load_cached_data)
    df_val = load_and_cache_df(Config.VAL_META_PATH, "val_meta", load_cached_data)
    df_test = load_and_cache_df(Config.TEST_META_PATH, "test_meta", load_cached_data)

    # 2. Handle Debug Mode
    if debug:
        logger.info(
            f"Debug mode enabled. Subsampling to {Config.DEBUG_SAMPLE_SIZE} samples."
        )
        df_train = df_train.head(Config.DEBUG_SAMPLE_SIZE)
        df_val = df_val.head(Config.DEBUG_SAMPLE_SIZE)
        df_test = df_test.head(Config.DEBUG_SAMPLE_SIZE)

    logger.info(f"Train samples: {len(df_train)}")
    logger.info(f"Val samples:   {len(df_val)}")
    logger.info(f"Test samples:  {len(df_test)}")

    # 3. Create Datasets
    train_dataset = AnimalDataset(
        df_train, transforms=get_transforms(mode="train"), mode="train"
    )

    val_dataset = AnimalDataset(
        df_val, transforms=get_transforms(mode="val"), mode="val"
    )

    test_dataset = AnimalDataset(
        df_test, transforms=get_transforms(mode="test"), mode="test"
    )

    # 4. Create DataLoaders
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
