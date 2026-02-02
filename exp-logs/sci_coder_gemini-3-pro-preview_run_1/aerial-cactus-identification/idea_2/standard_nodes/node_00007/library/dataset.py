import os
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from library.config import Config

# Dataset statistics derived from analysis
# Mean: R=128.37, G=115.25, B=119.40 -> Normalized by 255
DATASET_MEAN = [0.5034, 0.4520, 0.4682]
# Std: R=38.60, G=35.68, B=39.15 -> Normalized by 255
DATASET_STD = [0.1514, 0.1399, 0.1535]


def get_transforms(split="train"):
    """
    Returns the data transformation pipeline for a specific split.

    Args:
        split (str): The data split ('train', 'val', 'test').

    Returns:
        torchvision.transforms.Compose: Composed transformations.
    """
    if split == "train":
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=DATASET_MEAN, std=DATASET_STD),
            ]
        )
    else:
        # Validation and Test
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(mean=DATASET_MEAN, std=DATASET_STD),
            ]
        )


class CactusDataset(Dataset):
    """
    Custom Dataset for loading Cactus images from metadata.
    """

    def __init__(self, metadata_path, transform=None, debug=False):
        """
        Args:
            metadata_path (str): Path to the metadata CSV file.
            transform (callable, optional): Optional transform to be applied on a sample.
            debug (bool): If True, uses a small subset of the data for debugging.
        """
        self.metadata = pd.read_csv(metadata_path)
        self.transform = transform
        self.input_dir = Config.INPUT_DIR

        if debug:
            self.metadata = self.metadata.head(Config.DEBUG_SUBSET_SIZE)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # Get row data
        row = self.metadata.iloc[idx]

        # Construct file path
        # Metadata file_path is relative (e.g., "train/xxx.jpg")
        img_path = os.path.join(self.input_dir, row["file_path"])

        # Load image
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image at {img_path}")

        # Convert BGR (OpenCV default) to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        # Get label
        label = row["has_cactus"]

        # Return image and label (as float tensor for BCELoss)
        return image, torch.tensor(label, dtype=torch.float32)
