import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from library.config import Config


def get_transforms(data: str):
    """
    Returns the Albumentations transformation pipeline for the specified data split.

    Args:
        data (str): One of 'train', 'valid', or 'test'.

    Returns:
        A.Compose: The transformation pipeline.
    """
    if data == "train":
        return A.Compose(
            [
                A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1,
                    rotate_limit=15,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=0.5,
                ),
                A.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5
                ),
                A.CoarseDropout(
                    max_holes=4,
                    max_height=Config.IMAGE_SIZE // 10,
                    max_width=Config.IMAGE_SIZE // 10,
                    min_holes=1,
                    fill_value=0,
                    p=0.5,
                ),
                A.Normalize(mean=Config.MEAN, std=Config.STD),
                ToTensorV2(),
            ]
        )
    else:
        # Validation and Test transforms
        return A.Compose(
            [
                A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
                A.Normalize(mean=Config.MEAN, std=Config.STD),
                ToTensorV2(),
            ]
        )


class HotelDataset(Dataset):
    """
    Custom Dataset for loading Hotel images.
    """

    def __init__(self, df, transforms=None, is_test=False, class_to_idx=None):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe containing 'file_path' and 'hotel_id' (for train/val).
            transforms (A.Compose): Albumentations transforms.
            is_test (bool): Flag to indicate if this is the test set.
            class_to_idx (dict): Mapping from hotel_id to label index. Required if is_test is False.
        """
        self.df = df
        self.transforms = transforms
        self.is_test = is_test
        self.class_to_idx = class_to_idx
        self.input_dir = Config.INPUT_DIR

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Construct full path
        file_path = os.path.join(self.input_dir, row["file_path"])

        # Load image
        image = cv2.imread(file_path)

        # Handle missing or corrupt images gracefully (though metadata check should prevent this)
        if image is None:
            # Create a black image placeholder to allow training to continue
            image = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        if self.is_test:
            # For test, return image and the image filename (ID)
            return image, row["image"]
        else:
            # For train/val, return image and the mapped integer label
            hotel_id = row["hotel_id"]
            label_idx = self.class_to_idx[hotel_id]
            return image, torch.tensor(label_idx, dtype=torch.long)


def get_dataloaders(load_cached_data: bool = True):
    """
    Prepares and returns DataLoaders for Train, Validation, and Test sets.
    Handles class mapping generation and caching.

    Args:
        load_cached_data (bool): If True, attempts to load class mapping from cache.

    Returns:
        tuple: (train_loader, val_loader, test_loader, class_to_idx, idx_to_class)
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    # Class Mapping Caching Logic
    mapping_path = os.path.join(Config.WORKING_DIR, "class_mapping.parquet")

    class_to_idx = {}
    idx_to_class = {}

    mapping_loaded = False

    if load_cached_data and os.path.exists(mapping_path):
        try:
            mapping_df = pd.read_parquet(mapping_path)
            class_to_idx = dict(zip(mapping_df["hotel_id"], mapping_df["label_idx"]))
            idx_to_class = dict(zip(mapping_df["label_idx"], mapping_df["hotel_id"]))
            mapping_loaded = True
        except Exception:
            # If load fails, recompute
            mapping_loaded = False

    if not mapping_loaded:
        # Generate mapping from training data
        unique_ids = sorted(train_df["hotel_id"].unique())
        class_to_idx = {cls_id: idx for idx, cls_id in enumerate(unique_ids)}
        idx_to_class = {idx: cls_id for idx, cls_id in enumerate(unique_ids)}

        # Save as parquet
        mapping_df = pd.DataFrame(
            {
                "hotel_id": list(class_to_idx.keys()),
                "label_idx": list(class_to_idx.values()),
            }
        )
        mapping_df.to_parquet(mapping_path, index=False)

    # Initialize Datasets
    train_dataset = HotelDataset(
        train_df,
        transforms=get_transforms("train"),
        is_test=False,
        class_to_idx=class_to_idx,
    )

    val_dataset = HotelDataset(
        val_df,
        transforms=get_transforms("valid"),
        is_test=False,
        class_to_idx=class_to_idx,
    )

    test_dataset = HotelDataset(
        test_df, transforms=get_transforms("test"), is_test=True, class_to_idx=None
    )

    # Initialize DataLoaders
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

    return train_loader, val_loader, test_loader, class_to_idx, idx_to_class
