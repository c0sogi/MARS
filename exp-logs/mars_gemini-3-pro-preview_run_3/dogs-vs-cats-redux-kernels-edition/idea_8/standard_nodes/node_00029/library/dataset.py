import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from library.config import Config


def get_transforms(img_size, is_train=True):
    """
    Returns the data transformation pipeline based on the image size and mode.

    Args:
        img_size (int): The target image resolution (e.g., 224 or 256).
        is_train (bool): Whether to apply training augmentations.

    Returns:
        torchvision.transforms.Compose: The composed transform.
    """
    # Standard ImageNet normalization statistics
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if is_train:
        return transforms.Compose(
            [
                # Context-Preserving Augmentation: RandomResizedCrop with scale (0.8, 1.0)
                # Uses BICUBIC interpolation as per the strategy
                transforms.RandomResizedCrop(
                    (img_size, img_size),
                    scale=(0.8, 1.0),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(),
                # ColorJitter with intensity >= 0.2
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    else:
        return transforms.Compose(
            [
                # Deterministic resizing for validation/testing
                # Strictly use BICUBIC to align with training and pre-training recipes
                transforms.Resize(
                    (img_size, img_size), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )


class DogCatDataset(Dataset):
    """
    PyTorch Dataset for the Dog vs Cat classification task.
    Reads metadata from CSV files and loads images on-the-fly.
    """

    def __init__(self, split, img_size, transform=None, limit=None):
        """
        Args:
            split (str): One of 'train', 'val', 'test'.
            img_size (int): Target image size (used if transform is not provided,
                            though usually transform is passed explicitly).
            transform (callable, optional): Optional transform to be applied on a sample.
            limit (int, optional): If provided, limits the dataset size for debugging.
        """
        self.split = split
        self.transform = transform

        # Determine which metadata file to load
        if split == "train":
            csv_path = Config.TRAIN_CSV
        elif split == "val":
            csv_path = Config.VAL_CSV
        elif split == "test":
            csv_path = Config.TEST_CSV
        else:
            raise ValueError(
                f"Invalid split: {split}. Must be 'train', 'val', or 'test'."
            )

        # Load metadata
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Metadata file not found at {csv_path}")

        self.metadata = pd.read_csv(csv_path)

        # Debugging: Limit dataset size if requested
        if limit is not None:
            self.metadata = self.metadata.iloc[:limit]

        # Pre-construct full paths to avoid overhead in __getitem__
        # The metadata contains relative paths from the input directory
        self.metadata["full_path"] = self.metadata["filepath"].apply(
            lambda x: os.path.join(Config.INPUT_DIR, x)
        )

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        img_path = row["full_path"]

        # Load image and convert to RGB (handles grayscale images)
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # Fallback or error handling could go here, but for this task we assume valid data
            # based on the metadata generation script checks.
            raise IOError(f"Error loading image {img_path}: {e}")

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        # Return data based on split
        if self.split == "test":
            # For test set, we need the ID for submission
            # Ensure ID is an integer
            img_id = int(row["id"])
            return image, img_id
        else:
            # For train/val, we return the label
            # Label: 0 for cat, 1 for dog
            label = int(row["label"])
            # Return label as float for BCEWithLogitsLoss (expects float target)
            return image, torch.tensor(label, dtype=torch.float32)
