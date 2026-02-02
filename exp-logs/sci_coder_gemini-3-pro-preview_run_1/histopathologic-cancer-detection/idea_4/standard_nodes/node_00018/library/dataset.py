import os
import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from library.config import Config


def get_transforms(split):
    """
    Constructs the data augmentation and transformation pipeline.

    Args:
        split (str): The dataset split ('train', 'val', or 'test').

    Returns:
        A.Compose: The Albumentations composition of transforms.
    """
    transforms = []

    # 1. Hard Attention: Center Crop
    # Extract the central 48x48 region as defined in the idea and config.
    # This aligns the field of view with the tumor annotation logic.
    transforms.append(
        A.CenterCrop(height=Config.CENTER_CROP_SIZE, width=Config.CENTER_CROP_SIZE)
    )

    if split == "train":
        # 2. Geometric Augmentations
        # Randomly flip and rotate to improve invariance.
        if Config.AUG_HORIZONTAL_FLIP:
            transforms.append(A.HorizontalFlip(p=0.5))

        if Config.AUG_VERTICAL_FLIP:
            transforms.append(A.VerticalFlip(p=0.5))

        if Config.AUG_ROTATE_90:
            transforms.append(A.RandomRotate90(p=0.5))

        # 3. Color Augmentations
        # Strictly mild brightness and contrast. Hue and Saturation are excluded
        # to preserve H&E stain characteristics.
        transforms.append(
            A.RandomBrightnessContrast(
                brightness_limit=Config.AUG_BRIGHTNESS_LIMIT,
                contrast_limit=Config.AUG_CONTRAST_LIMIT,
                p=0.5,
            )
        )

    # 4. Normalization and Tensor Conversion
    # Normalize using ImageNet mean/std and convert to PyTorch Tensor.
    transforms.append(A.Normalize(mean=Config.MEAN, std=Config.STD))
    transforms.append(ToTensorV2())

    return A.Compose(transforms)


class TumorDataset(Dataset):
    """
    PyTorch Dataset for Digital Pathology Images.
    Handles image loading, hard attention cropping, and augmentation.
    """

    def __init__(self, df, split="train", transform=None):
        """
        Args:
            df (pd.DataFrame): DataFrame containing 'id', 'file_path', and 'label' (for train/val).
            split (str): 'train', 'val', or 'test'.
            transform (A.Compose, optional): Albumentations transforms. If None, generated based on split.
        """
        self.df = df
        self.split = split

        # Use provided transform or generate default based on split
        self.transform = transform if transform is not None else get_transforms(split)

        # Pre-construct full file paths to avoid overhead during iteration
        # Config.INPUT_DIR is "./input", file_path is relative "train/id.tif"
        self.file_paths = [
            os.path.join(Config.INPUT_DIR, fp) for fp in df["file_path"].tolist()
        ]

        # Store labels if they exist (train/val), else None (test)
        if "label" in df.columns:
            self.labels = df["label"].astype(float).tolist()
        else:
            self.labels = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.file_paths[idx]

        # Load image using OpenCV
        image = cv2.imread(path)

        # Handle potential loading errors (though metadata verification passed)
        if image is None:
            # Return a blank image to prevent crashing
            image = np.zeros(
                (Config.ORIGINAL_IMAGE_SIZE, Config.ORIGINAL_IMAGE_SIZE, 3),
                dtype=np.uint8,
            )
        else:
            # Convert BGR (OpenCV default) to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply the transformation pipeline
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Return image and label
        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return image, label
        else:
            # For test set, return a dummy label (-1) to maintain signature consistency
            return image, torch.tensor(-1.0, dtype=torch.float32)
