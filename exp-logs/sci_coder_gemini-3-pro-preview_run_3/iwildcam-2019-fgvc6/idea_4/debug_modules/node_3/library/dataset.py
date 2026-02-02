import os
import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from library.config import Config


def get_transforms(data: str = "train"):
    """
    Creates and returns the Albumentations transformation pipeline for a specific data split.

    Args:
        data (str): The data split to get transforms for. Options: 'train', 'valid', 'test'.

    Returns:
        A.Compose: The composed albumentations transformation pipeline.
    """
    if data == "train":
        return A.Compose(
            [
                A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                # Heavy geometric augmentations
                A.ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5
                ),
                # Photometric augmentations
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.5
                ),
                # Distributed occlusion to simulate vegetation
                # Using multiple small holes rather than one large block
                A.CoarseDropout(
                    max_holes=12,
                    min_holes=4,
                    max_height=Config.IMG_SIZE // 8,
                    max_width=Config.IMG_SIZE // 8,
                    fill_value=0,  # Black pixels
                    p=0.5,
                ),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    elif data in ["valid", "test"]:
        return A.Compose(
            [
                A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        raise ValueError(
            f"Unknown data split: {data}. Expected 'train', 'valid', or 'test'."
        )


class AnimalDataset(Dataset):
    """
    PyTorch Dataset for the Animal Species Classification task.
    Handles image loading via OpenCV and applies Albumentations transforms.
    """

    def __init__(self, df, transforms=None, mode="train"):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata (must include 'file_path').
                               For 'train'/'valid' modes, must also include 'Category'.
            transforms (A.Compose, optional): Albumentations augmentation pipeline.
            mode (str): Current mode ('train', 'valid', 'test'). Defaults to 'train'.
        """
        self.df = df
        self.transforms = transforms
        self.mode = mode

        # Pre-construct full file paths to optimize __getitem__
        # Metadata paths are relative to the input root
        self.file_paths = [
            os.path.join(Config.INPUT_ROOT, fp) for fp in df["file_path"].values
        ]

        # Load labels if not in test mode
        if self.mode != "test":
            if "Category" not in df.columns:
                raise ValueError(
                    f"DataFrame must contain 'Category' column for mode {self.mode}"
                )
            self.labels = df["Category"].values

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.df)

    def __getitem__(self, idx):
        """
        Retrieves the sample at the given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (image, label) if mode is 'train' or 'valid'.
            torch.Tensor: image if mode is 'test'.
        """
        image_path = self.file_paths[idx]

        # Load image using OpenCV
        image = cv2.imread(image_path)

        # Handle potential read errors (though metadata should be verified)
        if image is None:
            # Return a black image of correct size to prevent crashing
            image = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
        else:
            # Convert BGR (OpenCV default) to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply augmentations
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]
        else:
            # Fallback basic transform if none provided
            fallback = A.Compose(
                [
                    A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                    A.Normalize(),
                    ToTensorV2(),
                ]
            )
            image = fallback(image=image)["image"]

        # Return based on mode
        if self.mode == "test":
            return image
        else:
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return image, label
