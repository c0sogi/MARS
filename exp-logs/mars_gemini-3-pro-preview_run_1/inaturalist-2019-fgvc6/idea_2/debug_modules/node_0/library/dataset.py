import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from library.utils import seed_everything

# Constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
INPUT_ROOT = "./input"


def get_transforms(stage: str, image_size: int = 256):
    """
    Returns the image transformations for the specified stage.

    Args:
        stage (str): 'train', 'val', or 'test'.
        image_size (int): Target input size for the model (default 256).
    """
    if stage == "train":
        return transforms.Compose(
            [
                # Scale 0.4-1.0 forces model to look at local details and global shape
                transforms.RandomResizedCrop(image_size, scale=(0.4, 1.0)),
                transforms.RandomHorizontalFlip(),
                # Mild jitter, strictly avoiding hue rotation as color is discriminative
                transforms.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.1, hue=0
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
    elif stage in ["val", "test"]:
        # Resize shortest edge to approx 1.14x target size (e.g., 292 for 256)
        # to strictly preserve aspect ratio before center cropping.
        resize_dim = int(image_size * (292 / 256))

        return transforms.Compose(
            [
                transforms.Resize(resize_dim),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
    else:
        raise ValueError(f"Unknown stage: {stage}")


class INatDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        root_dir: str = INPUT_ROOT,
        transform=None,
        is_test: bool = False,
    ):
        """
        Custom Dataset for iNaturalist 2019.

        Args:
            df (pd.DataFrame): DataFrame containing metadata (paths, labels/ids).
            root_dir (str): Root directory where images are stored.
            transform (callable, optional): Transform to be applied on a sample.
            is_test (bool): If True, returns (image, image_id). If False, returns (image, category_id).
        """
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        self.is_test = is_test

        # Validation of dataframe columns
        if self.is_test:
            if "image_id" not in df.columns or "file_path" not in df.columns:
                raise ValueError(
                    "Test dataframe must contain 'image_id' and 'file_path' columns."
                )
        else:
            if "category_id" not in df.columns or "file_path" not in df.columns:
                raise ValueError(
                    "Train/Val dataframe must contain 'category_id' and 'file_path' columns."
                )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = os.path.join(self.root_dir, row["file_path"])

        # Open image and ensure RGB (handles grayscale or alpha channels)
        try:
            image = Image.open(file_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image at {file_path}: {e}")
            # Return a black image in case of corruption to prevent crash,
            # though in this competition data is assumed clean.
            image = Image.new("RGB", (256, 256), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        if self.is_test:
            # Return image and ID for submission mapping
            return image, row["image_id"]
        else:
            # Return image and target class index
            return image, row["category_id"]
