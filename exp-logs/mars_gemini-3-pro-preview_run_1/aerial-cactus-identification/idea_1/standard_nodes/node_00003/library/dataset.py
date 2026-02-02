import os
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from library.utils import Config


class CactusDataset(Dataset):
    """
    Custom Dataset for loading Cactus images.
    Reads images from disk based on metadata CSVs.
    """

    def __init__(self, metadata_path, root_dir, transform=None, debug_size=None):
        """
        Args:
            metadata_path (str): Path to the metadata CSV file.
            root_dir (str): Root directory containing the images (usually 'input').
            transform (callable, optional): Optional transform to be applied on a sample.
            debug_size (int, optional): If provided, limits the dataset size for debugging.
        """
        self.df = pd.read_csv(metadata_path)
        if debug_size is not None:
            self.df = self.df.iloc[:debug_size]

        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get row data
        row = self.df.iloc[idx]

        # Construct full file path
        # Metadata 'file_path' is relative to input dir (e.g., "train/id.jpg")
        img_path = os.path.join(self.root_dir, row["file_path"])

        # Load image
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {img_path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        # Get label
        # Ensure label is a float tensor for BCEWithLogitsLoss
        label = torch.tensor(row["has_cactus"], dtype=torch.float32)

        return image, label


def get_dataloaders(config: Config):
    """
    Creates and returns DataLoaders for train, validation, and test sets.

    Args:
        config (Config): Configuration object containing paths and hyperparameters.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Normalization statistics from Data Analysis
    # RGB Mean: [128.37, 115.25, 119.40] / 255
    # RGB Std:  [38.60, 35.68, 39.15] / 255
    mean = [0.5034, 0.4520, 0.4682]
    std = [0.1514, 0.1399, 0.1535]

    # Training Transforms: Augmentation + Normalization
    # Cite solution_lesson_node_00002: Adding RandomVerticalFlip for geometric invariance
    train_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    # Validation/Test Transforms: Normalization only
    val_test_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    # Instantiate Datasets
    # We pass config.INPUT_DIR because metadata file_paths are like "train/xxx.jpg"
    train_dataset = CactusDataset(
        metadata_path=config.TRAIN_METADATA_PATH,
        root_dir=config.INPUT_DIR,
        transform=train_transform,
        debug_size=config.DEBUG_SUBSET_SIZE,
    )

    val_dataset = CactusDataset(
        metadata_path=config.VAL_METADATA_PATH,
        root_dir=config.INPUT_DIR,
        transform=val_test_transform,
        debug_size=config.DEBUG_SUBSET_SIZE,
    )

    test_dataset = CactusDataset(
        metadata_path=config.TEST_METADATA_PATH,
        root_dir=config.INPUT_DIR,
        transform=val_test_transform,
        debug_size=config.DEBUG_SUBSET_SIZE,
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if config.DEVICE == "cuda" else False,
        drop_last=True,  # Drop last incomplete batch for training stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if config.DEVICE == "cuda" else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if config.DEVICE == "cuda" else False,
    )

    return train_loader, val_loader, test_loader
