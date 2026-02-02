import os
import cv2
import torch
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from library.config import Config


def get_transforms(mode="train", image_size=224):
    """
    Returns the albumentations transformation pipeline.

    Args:
        mode (str): 'train', 'valid', or 'test'.
        image_size (int): Target image size (width and height).

    Returns:
        A.Compose: Composed transformations.
    """
    # Standard ImageNet normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if mode == "train":
        return A.Compose(
            [
                A.Resize(height=image_size, width=image_size),
                # In a baseline, we stick to minimal augmentation.
                # Additional augs like HorizontalFlip could be added here if needed.
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    else:
        # Validation and Test: Resize and Normalize only
        return A.Compose(
            [
                A.Resize(height=image_size, width=image_size),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )


class ArtworkDataset(Dataset):
    """
    PyTorch Dataset for the Artwork Attribute Labeling task.
    """

    def __init__(
        self, metadata_path, input_dir, transform=None, mode="train", num_classes=3474
    ):
        """
        Args:
            metadata_path (str): Path to the CSV file containing metadata.
            input_dir (str): Root directory containing the image folders.
            transform (A.Compose): Albumentations transforms.
            mode (str): 'train', 'valid', or 'test'.
            num_classes (int): Total number of attribute classes.
        """
        self.df = pd.read_csv(metadata_path)
        self.input_dir = input_dir
        self.transform = transform
        self.mode = mode
        self.num_classes = num_classes

        # Pre-process labels for training and validation
        if self.mode != "test":
            self.labels = []
            for x in self.df["attribute_ids"]:
                if pd.isna(x) or str(x).strip() == "":
                    self.labels.append([])
                else:
                    # Convert "0 1 2" -> [0, 1, 2]
                    self.labels.append([int(i) for i in str(x).split()])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full image path
        # row['file_path'] is relative, e.g., "train/abc.png"
        img_path = os.path.join(self.input_dir, row["file_path"])

        # Load image using OpenCV
        image = cv2.imread(img_path)

        # Handle cases where image might be corrupt or missing (though metadata should be clean)
        if image is None:
            # Return a black image as fallback to prevent crashing
            image = np.zeros((Config.image_size, Config.image_size, 3), dtype=np.uint8)
        else:
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Return data based on mode
        if self.mode == "test":
            # For test, we need the ID to map predictions later
            return image, row["id"]
        else:
            # For train/val, we return the multi-hot encoded target
            label_indices = self.labels[idx]
            target = torch.zeros(self.num_classes, dtype=torch.float32)
            count = 0.0
            if label_indices:
                target[label_indices] = 1.0
                count = float(len(label_indices))

            return image, target, torch.tensor(count, dtype=torch.float32)
