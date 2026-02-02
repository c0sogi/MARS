import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import get_label_encoder


def get_transforms(phase: str, image_size: int = Config.IMAGE_SIZE):
    """
    Returns the image transformation pipeline for the specified phase.

    Args:
        phase (str): 'train', 'val', or 'test'.
        image_size (int): Target image size (default 224).

    Returns:
        A.Compose: Albumentations transform pipeline.
    """
    # Standard ImageNet normalization statistics
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if phase == "train":
        return A.Compose(
            [
                # Random resized crop is standard for ResNet training
                A.RandomResizedCrop(
                    height=image_size, width=image_size, scale=(0.2, 1.0)
                ),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    else:
        # For validation and test, we use standard ResNet preprocessing:
        # Resize such that smaller side is 256, then CenterCrop to 224.
        # 256 is approx 1.143 * 224.
        resize_dim = int(image_size * (256 / 224))

        return A.Compose(
            [
                A.SmallestMaxSize(max_size=resize_dim),
                A.CenterCrop(height=image_size, width=image_size),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )


class HotelDataset(Dataset):
    """
    PyTorch Dataset for Hotel Identification.
    Reads images via OpenCV and applies Albumentations transforms.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        phase: str = "train",
        transform: A.Compose = None,
        label_encoder=None,
        data_root: str = Config.INPUT_DIR,
        max_size: int = None,
    ):
        """
        Args:
            df (pd.DataFrame): Metadata DataFrame containing 'image', 'file_path', and 'hotel_id' (for train/val).
            phase (str): 'train', 'val', or 'test'.
            transform (A.Compose): Albumentations transforms.
            label_encoder (HotelIdLabelEncoder, optional): Pre-fitted label encoder.
                                                           If None, attempts to load/fit based on phase.
            data_root (str): Root directory containing image folders.
            max_size (int, optional): Limit dataset size for debugging purposes.
        """
        self.phase = phase
        self.transform = transform
        self.data_root = data_root

        # Apply debugging limit if specified
        if max_size is not None and max_size < len(df):
            self.df = df.sample(n=max_size, random_state=Config.SEED).reset_index(
                drop=True
            )
        else:
            self.df = df.reset_index(drop=True)

        self.file_paths = self.df["file_path"].values
        self.image_ids = self.df["image"].values

        # Handle Label Encoding
        self.labels = None
        self.label_encoder = label_encoder

        if self.phase in ["train", "val"]:
            # Ensure we have a label encoder
            if self.label_encoder is None:
                if self.phase == "train":
                    # For training, we can fit the encoder on the provided dataframe
                    # (assuming it represents the training set)
                    self.label_encoder = get_label_encoder(
                        metadata_df=self.df, load_cached_data=True
                    )
                else:
                    # For validation, we expect the encoder to be cached (fit during training prep)
                    self.label_encoder = get_label_encoder(load_cached_data=True)

            # Transform hotel_ids to integer targets
            self.labels = self.label_encoder.transform(self.df["hotel_id"].values)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Construct full file path
        file_path = self.file_paths[idx]
        full_path = os.path.join(self.data_root, file_path)

        # Read image using OpenCV
        image = cv2.imread(full_path)

        # Handle potential read failures (though dataset validation should prevent this)
        if image is None:
            # Return a blank image to prevent crashing, or could raise Error
            image = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8)
        else:
            # Convert BGR (OpenCV default) to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply Augmentations / Transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Fallback to simple tensor conversion if no transform provided
            image = ToTensorV2()(image=image)["image"]

        # Construct result dictionary
        result = {"image": image, "image_id": self.image_ids[idx]}

        # Add target for training/validation
        if self.phase in ["train", "val"]:
            result["target"] = torch.tensor(self.labels[idx], dtype=torch.long)

        return result
