import os
import cv2
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from library.config import Config
from library.utils import set_seed


class AnimalDataset(Dataset):
    """
    Dataset class for loading animal images from disk.
    Designed to work with the Cached Linear Probing strategy by providing
    efficient image loading and preprocessing.
    """

    def __init__(
        self, metadata_path, root_dir, transform=None, is_test=False, sample_size=None
    ):
        """
        Args:
            metadata_path (str): Path to the metadata CSV file.
            root_dir (str): Root directory containing the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            is_test (bool): If True, returns (image, id). If False, returns (image, label).
            sample_size (int, optional): If provided, limits the dataset to this many samples for debugging.
        """
        self.df = pd.read_csv(metadata_path)

        # Debugging option to limit dataset size
        if sample_size is not None:
            self.df = self.df.iloc[:sample_size]

        self.root_dir = root_dir
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct file path
        # metadata contains relative path in 'file_path' column (e.g., 'train_images/abc.jpg')
        rel_path = row["file_path"]
        full_path = os.path.join(self.root_dir, rel_path)

        # Load image using OpenCV (faster than PIL for resizing)
        img = cv2.imread(full_path)

        if img is None:
            # Fallback for potentially missing/corrupt images
            # Create a black image of correct size
            img = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8)
        else:
            # Convert BGR (OpenCV default) to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize image efficiently using OpenCV
        img = cv2.resize(img, (Config.IMAGE_SIZE, Config.IMAGE_SIZE))

        # Apply transforms (typically ToTensor and Normalize)
        if self.transform:
            # transforms.ToTensor() handles numpy [H, W, C] -> tensor [C, H, W] scaled to [0, 1]
            img = self.transform(img)

        if self.is_test:
            # Return image and Id for submission mapping
            return img, str(row["Id"])
        else:
            # Return image and Category label
            label = int(row["Category"])
            return img, torch.tensor(label, dtype=torch.long)


def get_transforms(split="train"):
    """
    Returns the transformations for the given split.

    For the Linear Probing strategy:
    - We use a frozen backbone, so we strictly use standard ImageNet normalization.
    - We avoid geometric augmentations (flips, crops) to ensure deterministic feature extraction.
    - Resizing is handled in __getitem__ for efficiency.
    """
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=Config.IMAGENET_MEAN, std=Config.IMAGENET_STD),
        ]
    )


def create_dataloaders(
    batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, sample_size=None
):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        batch_size (int): Batch size for loading.
        num_workers (int): Number of subprocesses for data loading.
        sample_size (int, optional): If provided, limits the dataset size for debugging.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Ensure reproducibility
    set_seed(Config.SEED)

    transform = get_transforms()

    # Initialize Datasets
    train_dataset = AnimalDataset(
        metadata_path=Config.TRAIN_METADATA_PATH,
        root_dir=Config.INPUT_DIR,
        transform=transform,
        is_test=False,
        sample_size=sample_size,
    )

    val_dataset = AnimalDataset(
        metadata_path=Config.VAL_METADATA_PATH,
        root_dir=Config.INPUT_DIR,
        transform=transform,
        is_test=False,
        sample_size=sample_size,
    )

    test_dataset = AnimalDataset(
        metadata_path=Config.TEST_METADATA_PATH,
        root_dir=Config.INPUT_DIR,
        transform=transform,
        is_test=True,
        sample_size=sample_size,
    )

    # Initialize DataLoaders
    # IMPORTANT: shuffle=False is used for ALL loaders in this strategy.
    # This ensures that the extracted features (saved to .npy) correspond 1-to-1
    # with the rows in the metadata CSVs (targets), simplifying the training logic.

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
