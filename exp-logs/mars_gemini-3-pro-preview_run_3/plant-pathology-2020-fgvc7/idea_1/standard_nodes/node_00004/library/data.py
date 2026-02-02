import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config


class AppleLeafDataset(Dataset):
    """
    PyTorch Dataset for Apple Disease Detection.
    Reads images from disk and applies transformations.
    """

    def __init__(
        self, df: pd.DataFrame, transforms: A.Compose = None, mode: str = "train"
    ):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata (image paths, labels).
            transforms (A.Compose, optional): Albumentations transformations.
            mode (str): 'train', 'val', or 'test'. Determines return values.
        """
        self.df = df
        self.transforms = transforms
        self.mode = mode
        self.classes = Config.CLASSES
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        # Construct full image path
        # Metadata file_path is relative, e.g., "images/Train_0.jpg"
        img_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # Load image using OpenCV
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {img_path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transformations
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        if self.mode in ["train", "val"]:
            # Get target label
            # 'stratify_label' contains the class name (e.g., 'rust')
            label_name = row["stratify_label"]
            label_idx = self.class_to_idx[label_name]
            return image, torch.tensor(label_idx, dtype=torch.long)
        else:
            # Test mode: return image and image_id for submission
            return image, row["image_id"]


def get_transforms(split: str = "train") -> A.Compose:
    """
    Returns the image transformation pipeline.

    Args:
        split (str): 'train', 'val', or 'test'.
    """
    if split == "train":
        return A.Compose(
            [
                A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5
                ),
                A.RandomBrightnessContrast(p=0.5),
                A.CoarseDropout(
                    max_holes=8,
                    max_height=int(Config.IMG_SIZE * 0.1),
                    max_width=int(Config.IMG_SIZE * 0.1),
                    min_holes=1,
                    fill_value=0,
                    p=0.5,
                ),
                A.Normalize(mean=Config.IMG_MEAN, std=Config.IMG_STD),
                ToTensorV2(),
            ]
        )
    else:
        # Validation and Test transforms (no augmentation)
        return A.Compose(
            [
                A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                A.Normalize(mean=Config.IMG_MEAN, std=Config.IMG_STD),
                ToTensorV2(),
            ]
        )


def calculate_class_weights(df: pd.DataFrame) -> torch.Tensor:
    """
    Computes inverse frequency weights for class imbalance handling.
    Formula: n_samples / (n_classes * n_samples_j)

    Args:
        df (pd.DataFrame): Training dataframe with 'stratify_label'.

    Returns:
        torch.Tensor: Weights for each class in order of Config.CLASSES.
    """
    class_counts = df["stratify_label"].value_counts().to_dict()
    n_samples = len(df)
    n_classes = len(Config.CLASSES)

    weights = []
    for cls in Config.CLASSES:
        count = class_counts.get(cls, 0)
        if count > 0:
            w = n_samples / (n_classes * count)
        else:
            w = 1.0
        weights.append(w)

    return torch.tensor(weights, dtype=torch.float)


def get_dataloaders(debug: bool = False):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        debug (bool): If True, uses a small subset of data for debugging.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # Debug mode: subset data
    if debug:
        train_df = train_df.sample(
            n=min(50, len(train_df)), random_state=Config.SEED
        ).reset_index(drop=True)
        val_df = val_df.sample(
            n=min(20, len(val_df)), random_state=Config.SEED
        ).reset_index(drop=True)
        test_df = test_df.sample(
            n=min(20, len(test_df)), random_state=Config.SEED
        ).reset_index(drop=True)

    # Define transforms
    train_transforms = get_transforms("train")
    val_transforms = get_transforms("val")

    # Create Datasets
    train_dataset = AppleLeafDataset(
        train_df, transforms=train_transforms, mode="train"
    )
    val_dataset = AppleLeafDataset(val_df, transforms=val_transforms, mode="val")
    test_dataset = AppleLeafDataset(test_df, transforms=val_transforms, mode="test")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
