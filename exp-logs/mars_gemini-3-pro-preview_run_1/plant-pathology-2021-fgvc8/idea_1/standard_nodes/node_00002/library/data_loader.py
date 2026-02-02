import os
import cv2
import torch
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import get_label_map

# Normalization constants derived from EDA
MEAN = [0.4871, 0.6273, 0.4093]
STD = [0.1901, 0.1673, 0.1984]


def get_transforms(data="train", size=256):
    """
    Returns Albumentations transforms for training or validation/testing.

    Args:
        data (str): 'train' for augmentation, 'val' or 'test' for deterministic resizing.
        size (int): Target image size.
    """
    if data == "train":
        return A.Compose(
            [
                A.Resize(size, size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                # Enhanced Augmentations for better generalization
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5
                ),
                A.OneOf(
                    [
                        A.RandomBrightnessContrast(
                            brightness_limit=0.1, contrast_limit=0.1, p=1
                        ),
                        A.HueSaturationValue(
                            hue_shift_limit=10,
                            sat_shift_limit=15,
                            val_shift_limit=10,
                            p=1,
                        ),
                    ],
                    p=0.5,
                ),
                A.CoarseDropout(
                    max_holes=8,
                    max_height=size // 10,
                    max_width=size // 10,
                    min_holes=1,
                    fill_value=0,
                    p=0.5,
                ),
                A.Normalize(mean=MEAN, std=STD),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [A.Resize(size, size), A.Normalize(mean=MEAN, std=STD), ToTensorV2()]
        )


def process_metadata(csv_path, mode, load_cached_data=True):
    """
    Loads and processes metadata with a strict caching mechanism using Parquet.

    Args:
        csv_path (str): Path to the source CSV file.
        mode (str): Dataset mode ('train', 'val', 'test') for naming the cache file.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: Processed dataframe with 'full_path' and 'target_vector' columns.
    """
    cache_path = os.path.join(Config.WORKING_DIR, f"{mode}_processed.parquet")

    # 1. Try to load cached data
    if load_cached_data and os.path.exists(cache_path):
        try:
            # PyArrow handles list columns in parquet efficiently
            df = pd.read_parquet(cache_path)
            return df
        except Exception:
            pass  # Fallback to processing from scratch if cache is corrupt

    # 2. Process data from scratch
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Construct absolute file paths
    # Metadata file_path is relative to INPUT_DIR (e.g., "train_images/abc.jpg")
    df["full_path"] = df["file_path"].apply(lambda x: os.path.join(Config.INPUT_DIR, x))

    # Pre-process labels into binary vectors (Multi-Hot Encoding)
    str2int, _ = get_label_map()

    def encode_labels(label_str):
        # label_str is space delimited, e.g., "scab rust"
        # For test set, labels might be a placeholder, but we process it safely
        labels = str(label_str).split()
        binary_vector = [0] * Config.NUM_CLASSES
        for l in labels:
            if l in str2int:
                binary_vector[str2int[l]] = 1
        return binary_vector

    df["target_vector"] = df["labels"].apply(encode_labels)

    # 3. Save to cache
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    df.to_parquet(cache_path, index=False)

    return df


class AppleDataset(Dataset):
    """
    PyTorch Dataset for Apple Disease Detection.
    """

    def __init__(self, df, transforms=None):
        self.df = df
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row["full_path"]

        # Load Image using OpenCV
        image = cv2.imread(image_path)
        if image is None:
            # Fallback for missing/corrupt images (creates a black image)
            image = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply Transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]
        else:
            t = ToTensorV2()
            image = t(image=image)["image"]

        # Get Label
        # target_vector is stored as a list in the dataframe
        target = torch.tensor(row["target_vector"], dtype=torch.float32)

        return image, target


def get_dataloaders(
    train_batch_size=Config.BATCH_SIZE,
    val_batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    load_cached_data=True,
    debug=Config.DEBUG,
):
    """
    Creates and returns DataLoaders for train, validation, and test sets.

    Args:
        train_batch_size (int): Batch size for training.
        val_batch_size (int): Batch size for validation/inference.
        num_workers (int): Number of subprocesses for data loading.
        load_cached_data (bool): Whether to use cached metadata.
        debug (bool): If True, uses a small subset of data.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """

    # Load and Process Metadata
    train_df = process_metadata(Config.TRAIN_METADATA_PATH, "train", load_cached_data)
    val_df = process_metadata(Config.VAL_METADATA_PATH, "val", load_cached_data)
    test_df = process_metadata(Config.TEST_METADATA_PATH, "test", load_cached_data)

    # Handle Debug Mode
    if debug:
        train_df = train_df.head(Config.DEBUG_SAMPLE_SIZE)
        val_df = val_df.head(Config.DEBUG_SAMPLE_SIZE)
        test_df = test_df.head(Config.DEBUG_SAMPLE_SIZE)

    # Define Transforms
    train_transforms = get_transforms("train", Config.IMG_SIZE)
    val_transforms = get_transforms("val", Config.IMG_SIZE)

    # Instantiate Datasets
    train_dataset = AppleDataset(train_df, transforms=train_transforms)
    val_dataset = AppleDataset(val_df, transforms=val_transforms)
    test_dataset = AppleDataset(test_df, transforms=val_transforms)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
