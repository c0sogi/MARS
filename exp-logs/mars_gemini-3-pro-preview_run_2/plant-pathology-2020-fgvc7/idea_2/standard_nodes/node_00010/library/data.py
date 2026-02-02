import os
import cv2
import random
import torch
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

from library.config import Config
from library.utils import seed_everything


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_dataset_df(metadata_path, cache_name, load_cached_data=True):
    """
    Loads the dataset dataframe, using a cache mechanism for deterministic processing.

    Args:
        metadata_path (str): Path to the original metadata CSV.
        cache_name (str): Name of the cache file (e.g., 'train_cache.parquet').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: The loaded dataframe.
    """
    # Ensure cache directory exists
    os.makedirs(Config.IDEA_DIR, exist_ok=True)
    cache_path = os.path.join(Config.IDEA_DIR, cache_name)

    # 1. IF load_cached_data is True: Try to load the file.
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            # Verify file_path column exists, if not, re-process
            if "full_path" in df.columns:
                return df
        except Exception:
            # If load fails, proceed to process from scratch
            pass

    # 2. IF loading fails OR load_cached_data is False: Process from scratch
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

    df = pd.read_csv(metadata_path)

    # Construct full path if not already present or if we want to ensure correctness
    # The metadata script generates 'file_path' relative to input dir.
    # We construct 'full_path' for easy loading.
    if "file_path" in df.columns:
        df["full_path"] = df["file_path"].apply(
            lambda x: os.path.join(Config.INPUT_DIR, x)
        )
    else:
        # Fallback if file_path isn't there, though metadata script guarantees it
        df["full_path"] = df["image_id"].apply(
            lambda x: os.path.join(Config.IMAGES_DIR, f"{x}.jpg")
        )

    # Save to cache
    try:
        df.to_parquet(cache_path, index=False)
    except Exception as e:
        print(f"Warning: Failed to save cache to {cache_path}. Error: {e}")

    return df


class AppleDataset(Dataset):
    def __init__(self, df, transforms=None, mode="train"):
        """
        Args:
            df (pd.DataFrame): DataFrame containing image paths and labels.
            transforms (albumentations.Compose): Albumentations transforms.
            mode (str): 'train', 'val', or 'test'.
        """
        self.df = df
        self.transforms = transforms
        self.mode = mode
        self.image_paths = df["full_path"].values

        # Extract labels if not in test mode
        if self.mode != "test":
            # Ensure columns exist and are in the correct order defined in Config
            self.labels = df[Config.CLASS_LABELS].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.image_paths[idx]

        # Read image
        image = cv2.imread(path)
        if image is None:
            # Handle missing image gracefully (though metadata check should prevent this)
            # Create a black image of expected size
            image = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]
        else:
            # Fallback to simple tensor conversion if no transforms provided
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        if self.mode == "test":
            return image, self.df.iloc[idx]["image_id"]
        else:
            label = torch.tensor(self.labels[idx])
            return image, label


def get_transforms(data="train"):
    """
    Returns the Albumentations transform pipeline.

    Args:
        data (str): 'train' or 'valid'/'test'.
    """
    if data == "train":
        return A.Compose(
            [
                A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                # CoarseDropout for information dropping strategy
                A.CoarseDropout(
                    max_holes=Config.COARSE_DROPOUT_MAX_HOLES,
                    max_height=Config.COARSE_DROPOUT_MAX_HEIGHT,
                    max_width=Config.COARSE_DROPOUT_MAX_WIDTH,
                    min_holes=Config.COARSE_DROPOUT_MIN_HOLES,
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
    elif data == "valid" or data == "test":
        return A.Compose(
            [
                A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    else:
        raise ValueError(f"Unknown data mode: {data}")


def get_loaders(load_cached_data=True):
    """
    Creates and returns DataLoaders for training and validation.

    Args:
        load_cached_data (bool): Whether to use cached dataframes.

    Returns:
        tuple: (train_loader, val_loader)
    """
    seed_everything(Config.SEED)

    # Load DataFrames
    train_df = load_dataset_df(
        Config.TRAIN_METADATA_PATH, "train_cache.parquet", load_cached_data
    )
    val_df = load_dataset_df(
        Config.VAL_METADATA_PATH, "val_cache.parquet", load_cached_data
    )

    # Debug Mode: Subsample data
    if Config.DEBUG:
        train_df = train_df.sample(
            n=min(len(train_df), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
        ).reset_index(drop=True)
        val_df = val_df.sample(
            n=min(len(val_df), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
        ).reset_index(drop=True)

    # Create Datasets
    train_dataset = AppleDataset(
        train_df, transforms=get_transforms(data="train"), mode="train"
    )

    val_dataset = AppleDataset(
        val_df, transforms=get_transforms(data="valid"), mode="val"
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=seed_worker,
    )

    return train_loader, val_loader


def get_test_loader(load_cached_data=True):
    """
    Creates and returns DataLoader for testing/inference.

    Args:
        load_cached_data (bool): Whether to use cached dataframes.

    Returns:
        DataLoader: test_loader
    """
    seed_everything(Config.SEED)

    test_df = load_dataset_df(
        Config.TEST_METADATA_PATH, "test_cache.parquet", load_cached_data
    )

    test_dataset = AppleDataset(
        test_df, transforms=get_transforms(data="test"), mode="test"
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=seed_worker,
    )

    return test_loader
