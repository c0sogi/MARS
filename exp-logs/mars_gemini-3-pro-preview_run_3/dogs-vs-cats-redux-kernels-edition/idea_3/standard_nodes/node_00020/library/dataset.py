import os
import cv2
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from library.config import Config


def get_transforms(phase: str):
    """
    Returns the data transformation pipeline for the specified phase.

    Args:
        phase (str): One of 'train', 'val', or 'test'.

    Returns:
        torchvision.transforms.Compose: The transform pipeline.
    """
    # Standard ImageNet normalization statistics
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if phase == "train":
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                # Context-Preserving Augmentation:
                # Scale (0.8, 1.0) prevents aggressive cropping of the subject
                transforms.RandomResizedCrop(
                    Config.IMG_SIZE,
                    scale=(0.8, 1.0),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    else:
        # Validation and Test: Resize to target size without cropping
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(
                    (Config.IMG_SIZE, Config.IMG_SIZE),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )


class DogCatDataset(Dataset):
    """
    PyTorch Dataset for the Dog vs Cat classification task.
    """

    def __init__(self, split: str, transform=None):
        """
        Args:
            split (str): One of 'train', 'val', or 'test'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.split = split
        self.transform = transform
        self.data = self._load_metadata(split)

        # Handle Debugging Mode
        if Config.DEBUG:
            # Slice the dataset to a smaller size for rapid iteration
            self.data = self.data.iloc[: Config.DEBUG_SAMPLES].reset_index(drop=True)
            print(f"DEBUG MODE: {split} dataset reduced to {len(self.data)} samples.")

    def _load_metadata(self, split: str) -> pd.DataFrame:
        """
        Loads the metadata CSV for the given split.

        Args:
            split (str): The dataset split ('train', 'val', 'test').

        Returns:
            pd.DataFrame: The loaded metadata.
        """
        if split == "train":
            csv_path = Config.TRAIN_CSV
        elif split == "val":
            csv_path = Config.VAL_CSV
        elif split == "test":
            csv_path = Config.TEST_CSV
        else:
            raise ValueError(f"Unknown split: {split}")

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Metadata file not found: {csv_path}")

        return pd.read_csv(csv_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index

        Returns:
            tuple: (image, target) where target is label for train/val and id for test.
        """
        row = self.data.iloc[idx]

        # Construct full file path
        # Metadata contains relative paths (e.g., 'train/cat.0.jpg')
        img_path = os.path.join(Config.INPUT_DIR, row["filepath"])

        # Load image using OpenCV
        image = cv2.imread(img_path)

        if image is None:
            # Fallback for missing/corrupt images (should be caught by metadata validation)
            # Return a blank image to prevent crashing, or raise error
            # Here we create a black image of default size
            image = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
        else:
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Return appropriate target based on split
        if self.split == "test":
            # For test, we need the ID for submission
            target = row["id"]
            return image, target
        else:
            # For train/val, we return the label
            target = row["label"]
            return image, torch.tensor(target, dtype=torch.float32)
