import os
import cv2
import torch
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

from library.config import Config
from library.utils import get_id_encoder


class WhaleDataset(Dataset):
    """
    Custom Dataset for loading Whale images and labels.
    """

    def __init__(self, df, transforms=None, id_encoder=None, is_test=False):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata (Image, file_path, Id).
            transforms (albumentations.Compose): Augmentation pipeline.
            id_encoder (IdEncoder): Encoder to convert string Ids to integers.
            is_test (bool): Flag to indicate if this is a test set (no labels).
        """
        self.df = df.reset_index(drop=True)
        self.transforms = transforms
        self.id_encoder = id_encoder
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full path
        # metadata file_path is relative to INPUT_DIR (e.g., "train/xxxx.jpg")
        image_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            # Fallback for missing images (though metadata check should prevent this)
            # Create a black image of expected size
            image = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transformations
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]
        else:
            # Minimal transform if none provided
            t = A.Compose(
                [
                    A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                    A.Normalize(),
                    ToTensorV2(),
                ]
            )
            image = t(image=image)["image"]

        # Handle Label
        if self.is_test:
            # For test set, return image and the original filename (for submission mapping)
            return image, row["Image"]
        else:
            label_str = row["Id"]
            label_idx = self.id_encoder.transform(label_str)
            return image, torch.tensor(label_idx, dtype=torch.long)


def get_transforms(phase="train"):
    """
    Returns the Albumentations transformation pipeline.

    Args:
        phase (str): 'train' or 'val'/'test'.
    """
    # Base transforms (Resize + Normalize + Tensor)
    # Normalization uses ImageNet defaults
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if phase == "train":
        return A.Compose(
            [
                A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                # Geometric Augmentations
                A.ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5
                ),
                A.HorizontalFlip(p=0.5),
                # Photometric Augmentations (Brightness/Contrast only)
                # Explicitly excluding Hue/Saturation as per requirements
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.5
                ),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    else:
        # Validation / Test: Deterministic resizing
        return A.Compose(
            [
                A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )


def get_loaders(debug=False, load_cached_data=True):
    """
    Creates DataLoaders for training and validation.

    Args:
        debug (bool): If True, subsamples the data for quick debugging.
        load_cached_data (bool): Passed to get_id_encoder to control caching.

    Returns:
        train_loader, val_loader
    """
    # 1. Load Metadata
    # We read the CSVs directly as they are already processed/split metadata
    if not os.path.exists(Config.TRAIN_CSV) or not os.path.exists(Config.VAL_CSV):
        raise FileNotFoundError(
            f"Metadata files not found. Ensure {Config.METADATA_DIR} contains train.csv and val.csv"
        )

    df_train = pd.read_csv(Config.TRAIN_CSV)
    df_val = pd.read_csv(Config.VAL_CSV)

    # 2. Debug Mode
    if debug:
        df_train = df_train.sample(
            n=min(len(df_train), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
        ).reset_index(drop=True)
        df_val = df_val.sample(
            n=min(len(df_val), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
        ).reset_index(drop=True)

    # 3. Initialize Encoder
    id_encoder = get_id_encoder(load_cached_data=load_cached_data)

    # 4. Create Datasets
    train_dataset = WhaleDataset(
        df_train,
        transforms=get_transforms("train"),
        id_encoder=id_encoder,
        is_test=False,
    )

    val_dataset = WhaleDataset(
        df_val, transforms=get_transforms("val"), id_encoder=id_encoder, is_test=False
    )

    # 5. Create Loaders
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

    return train_loader, val_loader
