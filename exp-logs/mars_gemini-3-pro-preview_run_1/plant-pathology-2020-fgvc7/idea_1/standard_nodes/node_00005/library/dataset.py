import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.utils import seed_everything

# Define the target classes in a fixed order
TARGET_COLS = ["healthy", "multiple_diseases", "rust", "scab"]


def get_transforms(split: str, image_size: int = 256):
    """
    Returns the data transformations for the specified split using Albumentations.
    Cite solution_lesson_node_00003: Synergistic Scaling of Model Depth and Data Augmentation.

    Args:
        split (str): One of 'train', 'val', or 'test'.
        image_size (int): The target spatial dimension (height and width) for resizing.

    Returns:
        albumentations.Compose: Composed transformations.
    """
    # ImageNet normalization statistics
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if split == "train":
        return A.Compose(
            [
                A.Resize(height=image_size, width=image_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=30, p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    else:
        # Validation and Test: Deterministic resizing and normalization
        return A.Compose(
            [
                A.Resize(height=image_size, width=image_size),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )


class AppleDataset(Dataset):
    def __init__(
        self, metadata_path, transform=None, input_dir="./input", mode="train"
    ):
        """
        Custom Dataset for Apple Disease Detection.

        Args:
            metadata_path (str): Path to the metadata CSV file (e.g., ./metadata/train_metadata.csv).
            transform (callable, optional): Optional transform to be applied on a sample.
            input_dir (str): Root directory containing the images (defaults to ./input).
            mode (str): Operation mode - 'train', 'val', or 'test'.
                        If 'train' or 'val', returns (image, label_index).
                        If 'test', returns (image, image_id).
        """
        self.metadata_path = metadata_path
        self.transform = transform
        self.input_dir = input_dir
        self.mode = mode
        self.target_cols = TARGET_COLS

        # Load metadata
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

        self.df = pd.read_csv(metadata_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full image path
        # row['file_path'] is relative to input_dir (e.g., "images/Train_0.jpg")
        img_path = os.path.join(self.input_dir, row["file_path"])

        # Load image and convert to RGB
        try:
            image = np.array(Image.open(img_path).convert("RGB"))
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image or handle error appropriately; here we raise
            raise e

        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Return data based on mode
        if self.mode in ["train", "val"]:
            # Extract target labels
            # We assume the columns exist and contain one-hot or probability distribution
            # We return the index of the max value for CrossEntropyLoss
            labels = row[self.target_cols].values.astype(np.float32)
            label_idx = np.argmax(labels)

            return image, torch.tensor(label_idx, dtype=torch.long)
        else:
            # Test mode: return image and ID for submission generation
            image_id = row["image_id"]
            return image, image_id
