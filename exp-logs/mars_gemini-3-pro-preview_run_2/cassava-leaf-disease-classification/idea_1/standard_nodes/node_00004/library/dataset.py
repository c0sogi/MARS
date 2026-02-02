import os
import cv2
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config


def get_transforms(data_type, cfg):
    """
    Returns the albumentations transforms for the given data split.
    Cite solution_lesson_node_00001: Implementing aggressive data augmentation to reduce overfitting.
    """
    if data_type == "train":
        return A.Compose(
            [
                A.RandomResizedCrop(size=(cfg.IMG_SIZE, cfg.IMG_SIZE)),
                A.Transpose(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(p=0.5),
                A.HueSaturationValue(
                    hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5
                ),
                A.CoarseDropout(p=0.5),
                A.Normalize(mean=cfg.MEAN, std=cfg.STD),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(height=cfg.IMG_SIZE, width=cfg.IMG_SIZE),
                A.Normalize(mean=cfg.MEAN, std=cfg.STD),
                ToTensorV2(),
            ]
        )


class CassavaDataset(Dataset):
    """
    Custom Dataset for Cassava Leaf Disease Classification.
    """

    def __init__(self, df, input_dir, transform=None):
        """
        Args:
            df (pd.DataFrame): DataFrame containing image paths and labels.
            input_dir (str): Root directory for input data.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.df = df
        self.input_dir = input_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full file path
        file_path = os.path.join(self.input_dir, row["file_path"])

        # Load image using OpenCV
        image = cv2.imread(file_path)

        if image is None:
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transformations
        # Albumentations expects numpy array, not PIL
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Get label
        label = torch.tensor(row["label"], dtype=torch.long)

        return image, label


def get_dataloaders(cfg):
    """
    Creates and returns DataLoaders for train, validation, and test sets.

    Args:
        cfg (Config): Configuration object instance.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load Metadata CSVs
    train_df = pd.read_csv(cfg.TRAIN_CSV)
    val_df = pd.read_csv(cfg.VAL_CSV)
    test_df = pd.read_csv(cfg.TEST_CSV)

    # Handle Debug Mode
    if cfg.DEBUG:
        train_df = train_df.head(cfg.DEBUG_SAMPLE_SIZE)
        val_df = val_df.head(cfg.DEBUG_SAMPLE_SIZE)
        test_df = test_df.head(cfg.DEBUG_SAMPLE_SIZE)

    # Define Transforms
    train_transform = get_transforms("train", cfg)
    val_transform = get_transforms("val", cfg)
    test_transform = get_transforms("test", cfg)

    # Instantiate Datasets
    train_dataset = CassavaDataset(
        df=train_df, input_dir=cfg.INPUT_DIR, transform=train_transform
    )

    val_dataset = CassavaDataset(
        df=val_df, input_dir=cfg.INPUT_DIR, transform=val_transform
    )

    test_dataset = CassavaDataset(
        df=test_df, input_dir=cfg.INPUT_DIR, transform=test_transform
    )

    # Instantiate DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Drop incomplete batch to maintain batch statistics
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
