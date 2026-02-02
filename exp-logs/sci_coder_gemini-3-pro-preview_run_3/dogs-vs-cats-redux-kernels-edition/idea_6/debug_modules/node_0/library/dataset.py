import os
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from library.config import Config


def get_transforms(phase: str):
    """
    Constructs the data augmentation and normalization pipeline.

    Args:
        phase (str): The phase of execution ('train', 'val', 'test').

    Returns:
        torchvision.transforms.Compose: The transform pipeline.
    """
    # Standard ImageNet normalization statistics
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if phase == "train":
        return transforms.Compose(
            [
                # Context-Preserving Augmentation:
                # RandomResizedCrop with restricted scale to avoid cropping out the subject
                transforms.RandomResizedCrop(
                    size=(Config.IMG_SIZE, Config.IMG_SIZE), scale=Config.CROP_SCALE
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                # Mild ColorJitter to add regularization without destroying semantics
                transforms.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    else:
        # Validation and Test:
        # Deterministic resize to the target input size
        return transforms.Compose(
            [
                transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )


class CatDogDataset(Dataset):
    """
    PyTorch Dataset for the Dog vs Cat classification task.
    Reads metadata from CSVs and loads images on-the-fly.
    """

    def __init__(self, csv_path: str, phase: str = "train", debug: bool = False):
        """
        Args:
            csv_path (str): Path to the metadata CSV file (train.csv, val.csv, or test.csv).
            phase (str): 'train', 'val', or 'test'. Controls transforms and return values.
            debug (bool): If True, limits the dataset to a small subset for debugging.
        """
        self.phase = phase
        self.csv_path = csv_path

        # Load metadata
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Metadata file not found at {csv_path}")

        self.df = pd.read_csv(csv_path)

        # Apply debug sampling if requested
        if debug:
            self.df = self.df.iloc[: Config.DEBUG_SAMPLE_SIZE].reset_index(drop=True)

        # Initialize transforms
        self.transforms = get_transforms(phase)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct absolute file path
        # row['filepath'] is relative to Config.INPUT_DIR (e.g., "train/cat.0.jpg")
        img_path = os.path.join(Config.INPUT_DIR, row["filepath"])

        # Load image using OpenCV
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Failed to load image at {img_path}")

        # Convert BGR (OpenCV default) to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image for compatibility with torchvision transforms
        image = Image.fromarray(image)

        # Apply data augmentations/preprocessing
        if self.transforms:
            image = self.transforms(image)

        # Return data based on phase
        if self.phase in ["train", "val"]:
            # For training/validation, return image and label
            # Ensure label is float32 for BCEWithLogitsLoss
            label = torch.tensor(row["label"], dtype=torch.float32)
            return image, label
        else:
            # For testing, return image and id (for submission mapping)
            img_id = row["id"]
            return image, img_id
