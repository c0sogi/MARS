import os
import cv2
import torch
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from library.config import Config


def get_transforms(img_size=Config.IMG_SIZE):
    """
    Returns the standard ImageNet preprocessing pipeline using Albumentations.

    Args:
        img_size (int): The target height and width for resizing.

    Returns:
        A.Compose: The transformation pipeline.
    """
    return A.Compose(
        [
            A.Resize(height=img_size, width=img_size),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0,
                p=1.0,
            ),
            ToTensorV2(),
        ]
    )


class PawpularityDataset(Dataset):
    """
    Custom Dataset for the Pawpularity prediction task.
    Handles loading images, applying transforms, and extracting metadata features.
    """

    def __init__(
        self,
        metadata_path,
        image_root=Config.INPUT_ROOT,
        transform=None,
        test_mode=False,
    ):
        """
        Args:
            metadata_path (str): Path to the metadata CSV file (train.csv, val.csv, or test.csv).
            image_root (str): Root directory containing the input images.
            transform (callable, optional): Albumentations transform pipeline.
            test_mode (bool): If True, returns (image, metadata, Id). If False, returns (image, metadata, target).
        """
        self.df = pd.read_csv(metadata_path)
        self.image_root = image_root
        self.transform = transform
        self.test_mode = test_mode

        # The 12 binary metadata features provided in the dataset
        self.meta_cols = [
            "Focus",
            "Eyes",
            "Face",
            "Near",
            "Action",
            "Accessory",
            "Group",
            "Collage",
            "Human",
            "Occlusion",
            "Info",
            "Blur",
        ]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Retrieve the dataframe row
        row = self.df.iloc[idx]

        # Construct the full image path
        # The metadata 'file_path' is relative (e.g., 'train/{id}.jpg')
        img_path = os.path.join(self.image_root, row["file_path"])

        # Load image using OpenCV
        image = cv2.imread(img_path)

        # Safety check for missing or corrupt images
        if image is None:
            # Return a blank image to prevent crashing, though dataset should be clean
            image = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
        else:
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Basic fallback: normalize and convert to tensor
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        # Extract binary metadata features
        meta = row[self.meta_cols].values.astype(np.float32)
        meta = torch.tensor(meta, dtype=torch.float32)

        if self.test_mode:
            # In test mode, return Id for submission file generation
            # Target is not returned (it might not exist or is irrelevant)
            return image, meta, row["Id"]
        else:
            # In training/validation mode, return the target score
            target = row["Pawpularity"]
            return image, meta, torch.tensor(target, dtype=torch.float32)
