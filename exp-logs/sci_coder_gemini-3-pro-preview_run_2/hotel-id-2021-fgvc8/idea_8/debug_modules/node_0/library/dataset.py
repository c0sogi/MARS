import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.augmentations import get_transforms


def get_class_mapping(train_df, load_cached_data=True):
    """
    Generates or loads a mapping from hotel_id to class index (0..N-1).
    Caches the mapping to a parquet file to ensure consistency.
    """
    cache_path = os.path.join(Config.WORKING_DIR, "class_mapping.parquet")

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    if load_cached_data and os.path.exists(cache_path):
        try:
            mapping_df = pd.read_parquet(cache_path)
            # Convert to dictionary: hotel_id -> class_idx
            class_mapping = dict(zip(mapping_df["hotel_id"], mapping_df["class_idx"]))
            return class_mapping
        except Exception as e:
            print(f"Failed to load cached class mapping: {e}. Recomputing...")

    # Compute mapping from scratch
    unique_ids = sorted(train_df["hotel_id"].unique())
    class_mapping = {hotel_id: idx for idx, hotel_id in enumerate(unique_ids)}

    # Save to cache
    mapping_df = pd.DataFrame(
        {
            "hotel_id": list(class_mapping.keys()),
            "class_idx": list(class_mapping.values()),
        }
    )
    mapping_df.to_parquet(cache_path, index=False)

    return class_mapping


class HotelDataset(Dataset):
    """
    PyTorch Dataset for Hotel ID classification.
    """

    def __init__(self, df, transform=None, mode="train", class_mapping=None):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe containing 'file_path' and 'hotel_id' (for train/val).
            transform (albumentations.Compose): Transformations to apply.
            mode (str): 'train', 'valid', or 'test'.
            class_mapping (dict): Mapping from hotel_id to class index. Required for train/valid.
        """
        self.df = df
        self.transform = transform
        self.mode = mode
        self.class_mapping = class_mapping

        # Pre-compute paths to avoid joining strings in __getitem__
        # Config.INPUT_DIR is "./input", file_path in df is relative like "train_images/..."
        self.file_paths = [
            os.path.join(Config.INPUT_DIR, fp) for fp in df["file_path"].values
        ]

        if self.mode in ["train", "valid"]:
            if self.class_mapping is None:
                raise ValueError("class_mapping must be provided for train/valid modes")
            # Map hotel_ids to indices
            self.labels = [self.class_mapping[hid] for hid in df["hotel_id"].values]
        elif self.mode == "test":
            # For test, we need image IDs for submission
            self.image_ids = df["image"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]

        # Load image
        image = cv2.imread(file_path)
        if image is None:
            # Handle missing/corrupt images gracefully by creating a black image
            # This prevents crashing during long training runs
            # Assuming standard size, though transform will resize anyway
            image = np.zeros((256, 256, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Fallback to tensor conversion if no transform provided
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        if self.mode in ["train", "valid"]:
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return image, label
        else:
            # For test, return image and its ID (filename)
            image_id = self.image_ids[idx]
            return image, image_id


def get_dataloaders(
    img_size=256,
    batch_size=32,
    load_cached_data=True,
    debug=False,
    debug_sample_size=1000,
):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        img_size (int): Target image resolution (e.g., 256 for Phase 1, 384 for Phase 2).
        batch_size (int): Batch size.
        load_cached_data (bool): Whether to use cached class mapping.
        debug (bool): If True, subsets the data for quick debugging.
        debug_sample_size (int): Number of samples to use in debug mode.

    Returns:
        tuple: (train_loader, val_loader, test_loader, num_classes)
    """
    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    # Debugging: Subsample data
    if debug:
        train_df = train_df.iloc[:debug_sample_size]
        val_df = val_df.iloc[:debug_sample_size]
        test_df = test_df.iloc[:debug_sample_size]

    # Get Class Mapping
    class_mapping = get_class_mapping(train_df, load_cached_data=load_cached_data)
    num_classes = len(class_mapping)

    # Get Transforms
    train_transform = get_transforms(image_size=img_size, mode="train")
    val_transform = get_transforms(image_size=img_size, mode="valid")
    # Test transform is same as valid usually (resize + normalize)
    test_transform = get_transforms(image_size=img_size, mode="test")

    # Instantiate Datasets
    train_dataset = HotelDataset(
        train_df, transform=train_transform, mode="train", class_mapping=class_mapping
    )

    val_dataset = HotelDataset(
        val_df, transform=val_transform, mode="valid", class_mapping=class_mapping
    )

    test_dataset = HotelDataset(test_df, transform=test_transform, mode="test")

    # Instantiate DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader, num_classes
