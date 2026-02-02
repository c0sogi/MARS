import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from library.config import Config


class CactusDataset(Dataset):
    """
    Custom Dataset for loading Cactus images from numpy arrays.
    """

    def __init__(self, images, labels, transform=None):
        """
        Args:
            images (np.ndarray): Array of images with shape (N, H, W, C).
            labels (np.ndarray): Array of labels with shape (N,).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Retrieve image and label
        image = self.images[idx]
        label = self.labels[idx]

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        # Convert label to tensor (float32 for BCEWithLogitsLoss)
        label = torch.tensor(label, dtype=torch.float32)

        return image, label


def get_transforms(split="train"):
    """
    Returns the transformations for the given split.

    Args:
        split (str): 'train', 'val', or 'test'.

    Returns:
        torchvision.transforms.Compose: Composed transformations.
    """
    if split == "train":
        return transforms.Compose(
            [
                transforms.ToTensor(),  # Converts HWC [0,255] to CHW [0.0, 1.0]
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ]
        )
    else:
        # For val and test, just convert to tensor
        return transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )


def _load_and_cache_data(
    metadata_path, cache_prefix, load_cached_data=True, debug=False
):
    """
    Loads data from metadata CSV and images, implementing a caching mechanism.

    Args:
        metadata_path (str): Path to the metadata CSV file.
        cache_prefix (str): Prefix for cache files (e.g., 'train', 'val').
        load_cached_data (bool): Whether to attempt loading from cache.
        debug (bool): If True, returns a subset of data for debugging.

    Returns:
        tuple: (images, labels) as numpy arrays.
    """
    # Ensure working directory exists for caching
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    images_cache_path = os.path.join(Config.WORKING_DIR, f"{cache_prefix}_images.npy")
    labels_cache_path = os.path.join(Config.WORKING_DIR, f"{cache_prefix}_labels.npy")

    # 1. Try to load from cache
    if (
        load_cached_data
        and os.path.exists(images_cache_path)
        and os.path.exists(labels_cache_path)
    ):
        images = np.load(images_cache_path)
        labels = np.load(labels_cache_path)
    else:
        # 2. Compute/Process from scratch
        df = pd.read_csv(metadata_path)

        img_list = []
        lbl_list = []

        for _, row in df.iterrows():
            # Metadata file_path is relative (e.g., "train/xxx.jpg")
            # We must join with INPUT_DIR to get the absolute path
            full_path = os.path.join(Config.INPUT_DIR, row["file_path"])

            # Read image using cv2 (reads as BGR)
            img = cv2.imread(full_path)
            if img is None:
                continue

            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img_list.append(img)
            lbl_list.append(row["has_cactus"])

        images = np.array(img_list)
        labels = np.array(lbl_list)

        # Save to cache for future runs
        np.save(images_cache_path, images)
        np.save(labels_cache_path, labels)

    # Handle debug mode by slicing the dataset
    if debug:
        images = images[: Config.DEBUG_SUBSET_SIZE]
        labels = labels[: Config.DEBUG_SUBSET_SIZE]

    return images, labels


def get_dataloaders(
    batch_size=None, num_workers=None, debug=False, load_cached_data=True
):
    """
    Creates and returns DataLoaders for train, val, and test sets.

    Args:
        batch_size (int, optional): Batch size. Defaults to Config.BATCH_SIZE.
        num_workers (int, optional): Number of workers. Defaults to Config.NUM_WORKERS.
        debug (bool, optional): Whether to run in debug mode. Defaults to False.
        load_cached_data (bool, optional): Whether to use cached data. Defaults to True.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    if batch_size is None:
        batch_size = Config.BATCH_SIZE
    if num_workers is None:
        num_workers = Config.NUM_WORKERS

    # Load Data (with caching)
    train_images, train_labels = _load_and_cache_data(
        Config.TRAIN_METADATA_PATH, "train", load_cached_data, debug
    )
    val_images, val_labels = _load_and_cache_data(
        Config.VAL_METADATA_PATH, "val", load_cached_data, debug
    )
    test_images, test_labels = _load_and_cache_data(
        Config.TEST_METADATA_PATH, "test", load_cached_data, debug
    )

    # Create Dataset Instances
    train_dataset = CactusDataset(
        train_images, train_labels, transform=get_transforms("train")
    )
    val_dataset = CactusDataset(val_images, val_labels, transform=get_transforms("val"))
    test_dataset = CactusDataset(
        test_images, test_labels, transform=get_transforms("test")
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
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
