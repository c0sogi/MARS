import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from library.config import Config


def get_transforms(data: str = "train"):
    """
    Returns the Albumentations transformation pipeline.

    Args:
        data (str): 'train' for training augmentations, 'valid' or 'test' for inference.
    """
    if data == "train":
        return A.Compose(
            [
                A.Resize(Config.img_size, Config.img_size),
                A.HorizontalFlip(p=0.5),
                # CoarseDropout for regularization (albumentations >= 2.0 syntax)
                A.CoarseDropout(
                    num_holes_range=(1, 8),
                    hole_height_range=(16, 64),
                    hole_width_range=(16, 64),
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
    elif data in ["valid", "test"]:
        return A.Compose(
            [
                A.Resize(Config.img_size, Config.img_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    else:
        raise ValueError(f"Unknown data mode: {data}")


class AppleDataset(Dataset):
    """
    PyTorch Dataset for Apple Disease Detection.
    Handles image loading, transformation, and multi-label target encoding.
    """

    def __init__(self, df: pd.DataFrame, transforms=None, debug: bool = False):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe containing 'file_path' and 'labels'.
            transforms (albumentations.Compose): Transformations to apply.
            debug (bool): If True, subsets the data for quick debugging.
        """
        self.df = df
        if debug:
            self.df = self.df.sample(
                n=min(100, len(self.df)), random_state=Config.seed
            ).reset_index(drop=True)

        self.transforms = transforms

        # Create a mapping from label name to index
        self.label_map = {label: i for i, label in enumerate(Config.class_labels)}
        self.num_classes = len(Config.class_labels)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]

        # Construct full image path
        # Metadata file_path is relative to input_dir (e.g., "train_images/xyz.jpg")
        img_path = os.path.join(Config.input_dir, row["file_path"])

        # Load image
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {img_path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        # Process labels
        # Initialize target vector of zeros
        target = torch.zeros(self.num_classes, dtype=torch.float32)

        # 'labels' column is a space-delimited string (e.g., "scab frog_eye_leaf_spot")
        # For test set, labels might be placeholder "healthy", but we still process it
        # (it will just mark "healthy" as 1, which is fine for inference structure)
        if pd.notna(row["labels"]) and row["labels"] != "":
            labels_list = row["labels"].split()
            for label in labels_list:
                if label in self.label_map:
                    idx = self.label_map[label]
                    target[idx] = 1.0

        return image, target


def load_metadata(mode: str = "train") -> pd.DataFrame:
    """
    Loads the metadata CSV file for the specified mode.

    Args:
        mode (str): One of 'train', 'val', 'test'.

    Returns:
        pd.DataFrame: The loaded metadata.
    """
    if mode == "train":
        path = Config.train_meta_path
    elif mode == "val":
        path = Config.val_meta_path
    elif mode == "test":
        path = Config.test_meta_path
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found: {path}")

    return pd.read_csv(path)
