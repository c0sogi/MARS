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
    Returns the Albumentations transformation pipeline for training, validation, or testing.

    Args:
        data (str): The data split ('train', 'valid', or 'test').

    Returns:
        A.Compose: The composition of transforms.
    """
    if data == "train":
        return A.Compose(
            [
                A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.1,
                    rotate_limit=15,
                    p=0.5,
                    border_mode=cv2.BORDER_REFLECT,
                ),
                A.Normalize(mean=Config.MEAN, std=Config.STD),
                ToTensorV2(),
            ]
        )
    elif data == "valid" or data == "test":
        return A.Compose(
            [
                A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                A.Normalize(mean=Config.MEAN, std=Config.STD),
                ToTensorV2(),
            ]
        )
    else:
        raise ValueError(f"Invalid data mode: {data}")


class CassavaDataset(Dataset):
    """
    Custom Dataset for loading Cassava images and labels.
    """

    def __init__(self, df, transforms=None, output_label=True):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata (image_id, label, file_path).
            transforms (albumentations.Compose, optional): Transforms to apply.
            output_label (bool): Whether to return the label (True for train/val).
        """
        self.df = df
        self.transforms = transforms
        self.output_label = output_label

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Retrieve metadata for the current index
        row = self.df.iloc[index]

        # Construct the full file path
        # Metadata contains relative paths (e.g., 'train_images/image.jpg')
        img_path = os.path.join(Config.INPUT_ROOT, row["file_path"])

        # Load image using OpenCV
        img = cv2.imread(img_path)

        # Safety check for invalid images
        if img is None:
            # Return a black image of correct size if load fails to prevent crash
            img = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
        else:
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Apply augmentations/transformations
        if self.transforms:
            img = self.transforms(image=img)["image"]

        # Return image and label if required
        if self.output_label:
            label = torch.tensor(row["label"], dtype=torch.long)
            return img, label
        else:
            return img
