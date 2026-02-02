import os
import cv2
import torch
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from library.config import Config
from library.utils import get_logger

# Initialize logger
logger = get_logger("dataset")


def get_metadata(mode: str, load_cached_data: bool = True) -> pd.DataFrame:
    """
    Loads metadata for the specified mode (train, val, test).
    Implements strict caching logic using Parquet files.

    Args:
        mode (str): One of 'train', 'val', 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: The loaded metadata.
    """
    # Define cache path
    cache_filename = f"meta_{mode}_data.parquet"
    cache_path = os.path.join(Config.cache_dir, cache_filename)

    # Ensure cache directory exists
    os.makedirs(Config.cache_dir, exist_ok=True)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            logger.info(f"Loading cached {mode} metadata from {cache_path}")
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}. Re-processing.")

    # 2. Process from scratch
    logger.info(f"Loading {mode} metadata from CSV...")
    if mode == "train":
        source_path = Config.train_metadata_path
    elif mode == "val":
        source_path = Config.val_metadata_path
    elif mode == "test":
        source_path = Config.test_metadata_path
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Metadata file not found: {source_path}")

    df = pd.read_csv(source_path)

    # 3. Save to cache
    try:
        logger.info(f"Saving {mode} metadata to cache at {cache_path}")
        df.to_parquet(cache_path, index=False)
    except Exception as e:
        logger.warning(f"Failed to save cache: {e}")

    return df


def get_transforms(mode: str = "train"):
    """
    Returns the Albumentations transform pipeline for the specified mode.

    Args:
        mode (str): 'train', 'val', or 'test'.

    Returns:
        A.Compose: The transform pipeline.
    """
    if mode == "train":
        return A.Compose(
            [
                A.RandomResizedCrop(
                    height=Config.img_size,
                    width=Config.img_size,
                    scale=Config.resize_crop_scale,
                ),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        # Validation and Test
        return A.Compose(
            [
                A.Resize(height=Config.img_size, width=Config.img_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


class PetDataset(Dataset):
    """
    PyTorch Dataset for Dog vs Cat classification.
    """

    def __init__(self, df: pd.DataFrame, mode: str = "train", transforms=None):
        """
        Args:
            df (pd.DataFrame): Dataframe containing metadata.
            mode (str): 'train', 'val', or 'test'.
            transforms (A.Compose): Albumentations transforms.
        """
        self.df = df
        self.mode = mode
        self.transforms = transforms
        self.input_dir = Config.input_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full file path
        # Metadata contains relative paths like "train/cat.0.jpg"
        img_path = os.path.join(self.input_dir, row["filepath"])

        # Read image
        image = cv2.imread(img_path)
        if image is None:
            # Handle missing/corrupt images gracefully by returning a black image
            # This prevents crashing during training, though ideally data is clean
            logger.error(f"Could not read image at {img_path}")
            image = np.zeros((Config.img_size, Config.img_size, 3), dtype=np.uint8)
        else:
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        # Return data based on mode
        if self.mode in ["train", "val"]:
            label = row["label"]
            # Ensure label is a tensor (long for classification, float for BCE)
            # Using float for BCEWithLogitsLoss or Mixup compatibility usually requires float
            # But standard CrossEntropy expects Long.
            # Given config uses Mixup, targets will likely be mixed in training loop.
            # We return standard types here.
            return image, torch.tensor(label, dtype=torch.long)

        elif self.mode == "test":
            img_id = row["id"]
            return image, img_id

        else:
            return image
