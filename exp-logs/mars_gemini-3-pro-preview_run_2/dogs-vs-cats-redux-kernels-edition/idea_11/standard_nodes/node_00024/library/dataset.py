import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from library.config import Config


def get_transforms(data="train"):
    """
    Returns the albumentations transformation pipeline.

    Args:
        data (str): 'train' for training augmentations, 'valid' for deterministic resizing.
    """
    if data == "train":
        return A.Compose(
            [
                A.RandomResizedCrop(
                    size=(Config.IMG_SIZE, Config.IMG_SIZE),
                    scale=(0.8, 1.0),
                    p=1.0,
                ),
                A.HorizontalFlip(p=0.5),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                    max_pixel_value=255.0,
                    p=1.0,
                ),
                ToTensorV2(),
            ]
        )
    elif data == "valid":
        return A.Compose(
            [
                A.Resize(height=Config.IMG_SIZE, width=Config.IMG_SIZE),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                    max_pixel_value=255.0,
                    p=1.0,
                ),
                ToTensorV2(),
            ]
        )
    else:
        raise ValueError(f"Unknown data mode: {data}")


class CatDogDataset(Dataset):
    """
    PyTorch Dataset for the Dog vs Cat classification task.
    Handles image loading via OpenCV and supports both hard and soft labels.
    """

    def __init__(self, df, transforms=None, input_dir=Config.INPUT_DIR):
        """
        Args:
            df (pd.DataFrame): DataFrame containing 'filepath' and optionally 'label' or 'id'.
            transforms (albumentations.Compose): Transformations to apply to the image.
            input_dir (str): Root directory for image paths.
        """
        self.df = df
        self.transforms = transforms
        self.input_dir = input_dir

        # Determine mode based on columns
        self.has_label = "label" in df.columns
        self.has_id = "id" in df.columns

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full file path
        # The metadata contains relative paths like "train/cat.0.jpg"
        file_path = os.path.join(self.input_dir, row["filepath"])

        # Read image using OpenCV
        image = cv2.imread(file_path)

        # Handle potential read errors (though metadata validation should catch missing files)
        if image is None:
            # Create a blank image or raise error.
            # For robustness in training, we'll create a black image but print a warning.
            # In a strict pipeline, raising an error is better.
            raise FileNotFoundError(f"Could not read image at {file_path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]
        else:
            # Fallback to simple tensor conversion if no transforms provided
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        # Return data based on available columns
        if self.has_label:
            # Return image and label
            # Supports both int (0/1) and float (soft labels)
            label = row["label"]
            # Ensure label is a float tensor for BCEWithLogitsLoss
            return image, torch.tensor(label, dtype=torch.float32)

        elif self.has_id:
            # Return image and id (for test submission mapping)
            img_id = row["id"]
            return image, int(img_id)

        else:
            # Fallback for inference without IDs
            return image
