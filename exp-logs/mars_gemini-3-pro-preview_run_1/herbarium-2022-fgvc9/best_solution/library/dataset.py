import os
import cv2
import torch
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

from library import config
from library import utils


class PlantDataset(Dataset):
    """
    Custom Dataset for Plant Classification.
    """

    def __init__(self, df, root_dir, transform=None, label_to_idx=None, is_test=False):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata (file_path, category_id, image_id).
            root_dir (str): Root directory for images (e.g., ./input).
            transform (A.Compose): Albumentations transformations.
            label_to_idx (dict): Mapping from raw category_id to model index (0..N-1).
            is_test (bool): If True, returns (image, image_id). If False, returns (image, label).
        """
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        self.label_to_idx = label_to_idx
        self.is_test = is_test

        # Pre-extract lists to avoid pandas overhead in __getitem__
        self.file_paths = self.df["file_path"].values

        if not self.is_test:
            # Ensure category_id exists
            if "category_id" not in self.df.columns:
                raise ValueError(
                    "DataFrame must contain 'category_id' for training/validation."
                )
            self.category_ids = self.df["category_id"].values
        else:
            self.image_ids = self.df["image_id"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Construct full file path
        rel_path = self.file_paths[idx]
        full_path = os.path.join(self.root_dir, rel_path)

        # Load image using OpenCV
        image = cv2.imread(full_path)
        if image is None:
            # Fallback for missing images (though metadata validation should prevent this)
            # Create a black image of expected size
            image = np.zeros((config.IMG_SIZE, config.IMG_SIZE, 3), dtype=np.uint8)
        else:
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        if self.is_test:
            # Return image and image_id for submission generation
            image_id = self.image_ids[idx]
            return image, str(image_id)
        else:
            # Return image and mapped label index
            raw_cat_id = self.category_ids[idx]
            label_idx = self.label_to_idx[raw_cat_id]
            return image, torch.tensor(label_idx, dtype=torch.long)


def get_transforms(data_type="train"):
    """
    Returns Albumentations transformations.

    Args:
        data_type (str): 'train', 'val', or 'test'.
    """
    if data_type == "train":
        return A.Compose(
            [
                A.RandomResizedCrop(
                    size=(config.IMG_SIZE, config.IMG_SIZE), scale=(0.8, 1.0)
                ),
                A.HorizontalFlip(p=0.5),
                A.Normalize(
                    mean=config.MEAN,
                    std=config.STD,
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(height=config.IMG_SIZE, width=config.IMG_SIZE),
                A.Normalize(
                    mean=config.MEAN,
                    std=config.STD,
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ]
        )


def get_dataloaders():
    """
    Creates and returns DataLoaders for train, val, and test sets.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # 1. Load Metadata
    train_df = pd.read_csv(config.TRAIN_CSV)
    val_df = pd.read_csv(config.VAL_CSV)
    test_df = pd.read_csv(config.TEST_CSV)

    # 2. Handle Debugging (Sample subset)
    if config.DEBUG_SAMPLE_SIZE is not None:
        train_df = train_df.iloc[: config.DEBUG_SAMPLE_SIZE]
        val_df = val_df.iloc[: config.DEBUG_SAMPLE_SIZE]
        test_df = test_df.iloc[: config.DEBUG_SAMPLE_SIZE]

    # 3. Get Class Mappings
    # We need to map the raw category_id to a 0-indexed range for CrossEntropyLoss
    label_to_idx, _ = utils.get_class_mappings(config.TRAIN_CSV)

    # 4. Create Transforms
    # For this baseline (linear probing), we use the same deterministic transform for all splits
    common_transform = get_transforms(data_type="train")

    # 5. Instantiate Datasets
    train_dataset = PlantDataset(
        df=train_df,
        root_dir=config.INPUT_DIR,
        transform=common_transform,
        label_to_idx=label_to_idx,
        is_test=False,
    )

    val_dataset = PlantDataset(
        df=val_df,
        root_dir=config.INPUT_DIR,
        transform=common_transform,
        label_to_idx=label_to_idx,
        is_test=False,
    )

    test_dataset = PlantDataset(
        df=test_df,
        root_dir=config.INPUT_DIR,
        transform=common_transform,
        label_to_idx=None,
        is_test=True,
    )

    # 6. Instantiate DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch to stabilize training
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
