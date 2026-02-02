import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.tokenizer import Tokenizer


def get_transforms(phase: str):
    """
    Returns the image transformations for the specified phase.

    Args:
        phase (str): 'train', 'valid', or 'test'.

    Returns:
        A.Compose: Albumentations composition of transforms.
    """
    # Standard ImageNet normalization statistics
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    # ResNet-TCN requires fixed size input
    return A.Compose(
        [
            A.Resize(height=Config.IMAGE_SIZE, width=Config.IMAGE_SIZE),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )


class InChiDataset(Dataset):
    """
    PyTorch Dataset for InChI chemical structure images.
    """

    def __init__(
        self, csv_path: str, tokenizer: Tokenizer, transform=None, is_test: bool = False
    ):
        """
        Args:
            csv_path (str): Path to the metadata CSV file.
            tokenizer (Tokenizer): Instance of the Tokenizer class.
            transform (A.Compose, optional): Albumentations transforms.
            is_test (bool): Flag to indicate if this is the test set (no labels).
        """
        self.tokenizer = tokenizer
        self.transform = transform
        self.is_test = is_test

        # Load metadata
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Metadata file not found at {csv_path}")

        self.df = pd.read_csv(csv_path)

        # Debugging subset logic
        if Config.DEBUG:
            print(
                f"DEBUG MODE: Sampling {Config.DEBUG_SUBSET_SIZE} rows from {os.path.basename(csv_path)}"
            )
            # Ensure we don't sample more than available
            sample_n = min(len(self.df), Config.DEBUG_SUBSET_SIZE)
            self.df = self.df.sample(n=sample_n, random_state=Config.SEED).reset_index(
                drop=True
            )

        # Pre-calculate file paths relative to input directory
        # The metadata contains 'file_path' relative to './input/'
        self.file_paths = self.df["file_path"].values

        if not self.is_test:
            self.labels = self.df["InChI"].values
        else:
            self.labels = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Construct full image path
        img_path = os.path.join(Config.INPUT_DIR, self.file_paths[idx])

        # Read image
        image = cv2.imread(img_path)
        if image is None:
            # Fallback for missing images (though validation showed none)
            # Create a black image of correct dimensions
            image = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Fallback to basic transform if none provided
            basic_transform = get_transforms("test")
            augmented = basic_transform(image=image)
            image = augmented["image"]

        # Handle Label
        if self.is_test:
            # For test set, return image and image_id (useful for submission)
            image_id = self.df.iloc[idx]["image_id"]
            return image, image_id
        else:
            label_text = self.labels[idx]
            # Convert text to tensor sequence
            label_seq = self.tokenizer.text_to_sequence(label_text)
            return image, label_seq
