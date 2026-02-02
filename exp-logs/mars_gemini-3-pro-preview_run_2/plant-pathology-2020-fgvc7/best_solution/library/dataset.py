import os
import cv2
import torch
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

from library.config import Config
from library.utils import worker_init_fn


def get_transforms(data_type: str):
    """
    Returns the Albumentations transformation pipeline based on the data type.

    Args:
        data_type (str): One of 'train', 'valid', or 'test'.
    """
    if data_type == "train":
        return A.Compose(
            [
                A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5
                ),
                A.CoarseDropout(
                    max_holes=Config.COARSE_DROPOUT_MAX_HOLES,
                    max_height=Config.COARSE_DROPOUT_MAX_HEIGHT,
                    max_width=Config.COARSE_DROPOUT_MAX_WIDTH,
                    min_holes=1,
                    min_height=Config.COARSE_DROPOUT_MIN_HEIGHT,
                    min_width=Config.COARSE_DROPOUT_MIN_WIDTH,
                    fill_value=0,
                    p=0.5,
                ),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    elif data_type in ["valid", "test"]:
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
        raise ValueError(f"Unknown data_type: {data_type}")


class AppleDataset(Dataset):
    """
    Dataset class for Apple Disease Detection.
    Handles image loading and multi-label target engineering.
    """

    def __init__(self, df: pd.DataFrame, transforms=None, mode="train"):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata.
            transforms (albumentations.Compose): Transformations to apply.
            mode (str): 'train', 'valid', or 'test'.
        """
        self.df = df
        self.transforms = transforms
        self.mode = mode

        # Pre-construct full file paths
        # df['file_path'] is relative, e.g., "images/Train_0.jpg"
        # Config.INPUT_DIR is "./input"
        self.file_paths = [
            os.path.join(Config.INPUT_DIR, fp) for fp in df["file_path"].values
        ]
        self.image_ids = df["image_id"].values

        # Prepare targets if not in test mode
        if self.mode != "test":
            self.targets = self._prepare_targets(df)
        else:
            self.targets = None

    def _prepare_targets(self, df: pd.DataFrame) -> np.ndarray:
        """
        Converts one-hot encoded columns into a 2-label binary format: [Rust, Scab].

        Mapping Logic:
        - Healthy (0,0,0,1) -> [0, 0]
        - Rust (1,0,0,0)    -> [1, 0]
        - Scab (0,1,0,0)    -> [0, 1]
        - Multiple (0,0,1,0)-> [1, 1]

        Note: The input DataFrame columns are: 'healthy', 'multiple_diseases', 'rust', 'scab'.
        """
        # Ensure columns exist
        required_cols = ["rust", "scab", "multiple_diseases"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing column {col} in dataframe")

        # Vectorized operation
        # Rust presence: explicitly labeled 'rust' OR 'multiple_diseases'
        rust_target = np.maximum(df["rust"].values, df["multiple_diseases"].values)

        # Scab presence: explicitly labeled 'scab' OR 'multiple_diseases'
        scab_target = np.maximum(df["scab"].values, df["multiple_diseases"].values)

        # Stack into (N, 2) array
        targets = np.stack([rust_target, scab_target], axis=1).astype(np.float32)
        return targets

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image_id = self.image_ids[idx]

        # Load image
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        # Return based on mode
        if self.mode == "test":
            return image, image_id
        else:
            target = torch.tensor(self.targets[idx], dtype=torch.float32)
            return image, target


def load_cached_dataframe(
    csv_path: str, cache_name: str, load_cached: bool = True
) -> pd.DataFrame:
    """
    Loads a dataframe from a CSV, caching it as a Parquet file for faster subsequent access.

    Args:
        csv_path (str): Path to the source CSV file.
        cache_name (str): Name of the cache file (e.g., 'train_cache.parquet').
        load_cached (bool): Whether to attempt loading from cache.
    """
    cache_path = os.path.join(Config.WORKING_DIR, cache_name)

    if load_cached and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception:
            # If cache is corrupt, fall back to CSV
            pass

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Source file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Save to cache
    try:
        df.to_parquet(cache_path, index=False)
    except Exception as e:
        print(f"Warning: Failed to save cache to {cache_path}. Error: {e}")

    return df


def get_dataloaders(load_cached_data: bool = True):
    """
    Creates and returns DataLoaders for train, validation, and test sets.

    Args:
        load_cached_data (bool): If True, attempts to load metadata from Parquet cache.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # 1. Load DataFrames
    train_df = load_cached_dataframe(
        Config.TRAIN_METADATA_PATH, "train_cache.parquet", load_cached_data
    )
    val_df = load_cached_dataframe(
        Config.VAL_METADATA_PATH, "val_cache.parquet", load_cached_data
    )
    test_df = load_cached_dataframe(
        Config.TEST_METADATA_PATH, "test_cache.parquet", load_cached_data
    )

    # 2. Create Datasets
    train_dataset = AppleDataset(
        train_df, transforms=get_transforms("train"), mode="train"
    )
    val_dataset = AppleDataset(val_df, transforms=get_transforms("valid"), mode="valid")
    test_dataset = AppleDataset(test_df, transforms=get_transforms("test"), mode="test")

    # 3. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )

    return train_loader, val_loader, test_loader
