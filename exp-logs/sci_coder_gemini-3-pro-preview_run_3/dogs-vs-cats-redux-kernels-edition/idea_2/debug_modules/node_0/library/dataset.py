import os
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from library.config import Config
from library.utils import get_logger

logger = get_logger("dataset")


class CatDogDataset(Dataset):
    """
    PyTorch Dataset for Dog vs Cat classification.
    """

    def __init__(
        self, df: pd.DataFrame, image_dir: str, transform=None, mode: str = "train"
    ):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata (filepath, label/id).
            image_dir (str): Root directory for images.
            transform (callable, optional): Optional transform to be applied on a sample.
            mode (str): 'train', 'val', or 'test'. Determines what is returned.
        """
        self.df = df
        self.image_dir = image_dir
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full file path
        # Metadata filepath is relative to input dir, e.g., "train/cat.0.jpg"
        img_path = os.path.join(self.image_dir, row["filepath"])

        # Load image using OpenCV
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {img_path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image for torchvision transforms
        image = Image.fromarray(image)

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Return data based on mode
        if self.mode == "test":
            # For test, return image and ID
            return image, row["id"]
        else:
            # For train/val, return image and label
            # Config.NUM_CLASSES = 2, so we return a LongTensor for CrossEntropyLoss
            label = torch.tensor(row["label"], dtype=torch.long)
            return image, label


def get_transforms(mode: str = "train"):
    """
    Returns the torchvision transforms for the specified mode.

    Args:
        mode (str): 'train', 'val', or 'test'.
    """
    # ImageNet normalization statistics
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if mode == "train":
        return transforms.Compose(
            [
                # Context-Preserving Augmentation: RandomResizedCrop
                # This crops a random portion of the image and resizes it to IMG_SIZE
                transforms.RandomResizedCrop(
                    size=Config.IMG_SIZE, scale=Config.AUG_CROP_SCALE
                ),
                transforms.RandomHorizontalFlip(p=Config.AUG_HFLIP_PROB),
                transforms.ColorJitter(
                    brightness=Config.AUG_COLOR_JITTER,
                    contrast=Config.AUG_COLOR_JITTER,
                    saturation=Config.AUG_COLOR_JITTER,
                    hue=0.0,  # Hue is typically kept stable for natural images unless specified
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    else:
        # Validation/Test: Deterministic resize to target size
        # We resize to (IMG_SIZE, IMG_SIZE) to ensure consistent input dimensions
        return transforms.Compose(
            [
                transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )


def get_dataloaders(
    train_csv=Config.TRAIN_CSV,
    val_csv=Config.VAL_CSV,
    test_csv=Config.TEST_CSV,
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
):
    """
    Creates and returns DataLoaders for train, validation, and test sets.

    Returns:
        train_loader, val_loader, test_loader
    """
    logger.info("Loading metadata...")

    # Load DataFrames
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    logger.info(f"Train samples: {len(train_df)}")
    logger.info(f"Val samples: {len(val_df)}")
    logger.info(f"Test samples: {len(test_df)}")

    # Initialize Datasets
    train_dataset = CatDogDataset(
        df=train_df,
        image_dir=Config.INPUT_DIR,
        transform=get_transforms(mode="train"),
        mode="train",
    )

    val_dataset = CatDogDataset(
        df=val_df,
        image_dir=Config.INPUT_DIR,
        transform=get_transforms(mode="val"),
        mode="val",
    )

    test_dataset = CatDogDataset(
        df=test_df,
        image_dir=Config.INPUT_DIR,
        transform=get_transforms(mode="test"),
        mode="test",
    )

    # Initialize DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch to maintain consistent stats
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
