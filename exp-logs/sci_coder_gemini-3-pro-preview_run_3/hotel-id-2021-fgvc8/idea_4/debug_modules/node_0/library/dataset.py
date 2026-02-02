import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import seed_everything


def get_transforms(phase: str):
    """
    Returns the Albumentations transformation pipeline based on the phase.

    Args:
        phase (str): 'train', 'val', or 'test'.

    Strategy:
        - Resize to 256x256.
        - CenterCrop to 224x224.
        - Train: Add HorizontalFlip.
        - Normalize using ImageNet mean/std.
    """
    transforms = []

    # Resize logic as per "Idea"
    transforms.append(A.Resize(height=Config.RESIZE_SIZE, width=Config.RESIZE_SIZE))
    transforms.append(A.CenterCrop(height=Config.IMAGE_SIZE, width=Config.IMAGE_SIZE))

    if phase == "train":
        transforms.append(A.HorizontalFlip(p=0.5))

    transforms.append(
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    )
    transforms.append(ToTensorV2())

    return A.Compose(transforms)


def get_label_encoding(train_df, load_cached_data=True):
    """
    Generates or loads the mapping between raw hotel_ids and class indices.

    Args:
        train_df (pd.DataFrame): Training metadata containing 'hotel_id'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (hotel_to_idx (dict), idx_to_hotel (np.array))
    """
    cache_path = os.path.join(Config.WORKING_DIR, "label_encoder.npy")

    unique_hotels = None

    # 1. Try to load cached data
    if load_cached_data and os.path.exists(cache_path):
        try:
            unique_hotels = np.load(cache_path)
            # Verify consistency if not in debug mode
            if not Config.DEBUG and len(unique_hotels) != Config.NUM_CLASSES:
                # If cache doesn't match config expectation, force recompute
                unique_hotels = None
        except Exception:
            unique_hotels = None

    # 2. Compute if not loaded
    if unique_hotels is None:
        unique_hotels = np.sort(train_df["hotel_id"].unique())

        # Save to cache
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        np.save(cache_path, unique_hotels)

    # Create mappings
    # idx_to_hotel is just the array itself (index -> value)
    # hotel_to_idx is the reverse map
    hotel_to_idx = {hotel_id: idx for idx, hotel_id in enumerate(unique_hotels)}

    return hotel_to_idx, unique_hotels


class HotelDataset(Dataset):
    def __init__(self, df, root_dir, transform=None, hotel_to_idx=None, is_test=False):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe.
            root_dir (str): Root directory containing image folders.
            transform (albumentations.Compose): Transformations to apply.
            hotel_to_idx (dict): Mapping from hotel_id to class index.
            is_test (bool): If True, returns dummy label.
        """
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        self.hotel_to_idx = hotel_to_idx
        self.is_test = is_test

        # Pre-compute full paths to avoid string concat in loop
        # The metadata 'file_path' is relative to INPUT_DIR
        self.file_paths = df["file_path"].values

        if not self.is_test:
            self.hotel_ids = df["hotel_id"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        rel_path = self.file_paths[idx]
        full_path = os.path.join(self.root_dir, rel_path)

        # Load image
        image = cv2.imread(full_path)
        if image is None:
            # Fallback for missing/corrupt images: return black image
            image = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Handle Label
        if self.is_test:
            # For test, we don't have ground truth, return dummy or raw hotel_id string if needed
            # Returning -1 as placeholder
            target = -1
        else:
            raw_id = self.hotel_ids[idx]
            target = self.hotel_to_idx[raw_id]

        return image, target


def get_dataloaders(load_cached_data=True):
    """
    Prepares DataLoaders for Train, Validation, and Test sets.

    Args:
        load_cached_data (bool): Whether to use cached label encoding.

    Returns:
        train_loader, val_loader, test_loader, idx_to_hotel_id (np.array)
    """
    seed_everything(Config.SEED)

    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # Debug Mode: Sample subset
    if Config.DEBUG:
        train_df = train_df.sample(
            n=min(len(train_df), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
        ).reset_index(drop=True)
        val_df = val_df.sample(
            n=min(len(val_df), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
        ).reset_index(drop=True)
        test_df = test_df.sample(
            n=min(len(test_df), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
        ).reset_index(drop=True)

    # Get Label Encoding
    # We derive this from the full training set (or the debug subset if debugging)
    hotel_to_idx, idx_to_hotel = get_label_encoding(
        train_df, load_cached_data=load_cached_data
    )

    # Define Transforms
    train_transform = get_transforms(phase="train")
    val_transform = get_transforms(phase="val")  # Same for test

    # Instantiate Datasets
    # Root dir is Config.INPUT_DIR because file_paths in csv are like "train_images/..."
    train_dataset = HotelDataset(
        train_df,
        Config.INPUT_DIR,
        transform=train_transform,
        hotel_to_idx=hotel_to_idx,
        is_test=False,
    )

    val_dataset = HotelDataset(
        val_df,
        Config.INPUT_DIR,
        transform=val_transform,
        hotel_to_idx=hotel_to_idx,
        is_test=False,
    )

    test_dataset = HotelDataset(
        test_df,
        Config.INPUT_DIR,
        transform=val_transform,
        hotel_to_idx=None,
        is_test=True,
    )

    # Instantiate DataLoaders
    # Strategy: Standard Random Sampling (shuffle=True) for training
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

    return train_loader, val_loader, test_loader, idx_to_hotel
