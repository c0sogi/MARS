import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from library.utils import seed_everything

# Constants
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
CACHE_DIR = "./working/idea_29"


class CactusDataset(Dataset):
    """
    Custom Dataset for Cactus Identification.
    """

    def __init__(self, images, labels=None, transform=None):
        """
        Args:
            images (np.ndarray): Array of images (N, H, W, C).
            labels (np.ndarray, optional): Array of labels (N,).
            transform (callable, optional): Transform to apply.
        """
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Images are stored as uint8 (H, W, C)
        image = self.images[idx]

        if self.transform:
            image = self.transform(image)

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return image, label
        else:
            return image


def get_transforms(split="train"):
    """
    Returns transforms for a given split.

    Args:
        split (str): 'train', 'val', or 'test'.
    """
    if split == "train":
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),  # Converts [0, 255] -> [0.0, 1.0]
            ]
        )
    else:
        # val or test
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
            ]
        )


def _load_metadata(split):
    """Loads metadata CSV for a split."""
    path = os.path.join(METADATA_DIR, f"{split}_metadata.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found: {path}")
    return pd.read_csv(path)


def _process_data_from_source(split):
    """Reads images and labels from source files based on metadata."""
    df = _load_metadata(split)

    images = []
    labels = []
    ids = []

    # Iterate through metadata
    for _, row in df.iterrows():
        rel_path = row["file_path"]
        full_path = os.path.join(INPUT_DIR, rel_path)

        # Load image
        img = cv2.imread(full_path)
        if img is None:
            continue

        # BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        images.append(img)
        ids.append(row["id"])

        if "has_cactus" in row:
            labels.append(row["has_cactus"])

    images = np.array(images, dtype=np.uint8)
    labels = np.array(labels, dtype=np.float32) if labels else None
    ids = np.array(ids)

    return images, labels, ids


def load_data(split, load_cached_data=True):
    """
    Loads data for a split, using caching mechanism.

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): If True, attempts to load from cache.

    Returns:
        tuple: (images, labels, ids)
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    images_path = os.path.join(CACHE_DIR, f"{split}_images.npy")
    labels_path = os.path.join(CACHE_DIR, f"{split}_labels.npy")
    ids_path = os.path.join(CACHE_DIR, f"{split}_ids.npy")

    # Attempt to load from cache
    if load_cached_data:
        if os.path.exists(images_path) and os.path.exists(ids_path):
            # Check labels existence (test might not have valid labels, but we save None/placeholder)
            if split == "test" or os.path.exists(labels_path):
                try:
                    images = np.load(images_path)
                    ids = np.load(ids_path, allow_pickle=True)
                    if os.path.exists(labels_path):
                        labels = np.load(labels_path)
                    else:
                        labels = None
                    return images, labels, ids
                except Exception:
                    # If load fails, fall through to process from scratch
                    pass

    # Process from scratch
    images, labels, ids = _process_data_from_source(split)

    # Save to cache
    np.save(images_path, images)
    np.save(ids_path, ids)
    if labels is not None:
        np.save(labels_path, labels)
    elif os.path.exists(labels_path):
        # Clean up old labels if they exist but shouldn't
        try:
            os.remove(labels_path)
        except OSError:
            pass

    return images, labels, ids


def get_dataloaders(batch_size=32, num_workers=2, load_cached_data=True):
    """
    Constructs DataLoaders for train, val, and test splits.

    Args:
        batch_size (int): Batch size.
        num_workers (int): Number of worker subprocesses.
        load_cached_data (bool): Whether to use cached data.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load data
    train_imgs, train_lbls, _ = load_data("train", load_cached_data)
    val_imgs, val_lbls, _ = load_data("val", load_cached_data)
    test_imgs, _, _ = load_data("test", load_cached_data)

    # Create Datasets
    train_dataset = CactusDataset(
        train_imgs, train_lbls, transform=get_transforms("train")
    )

    val_dataset = CactusDataset(val_imgs, val_lbls, transform=get_transforms("val"))

    test_dataset = CactusDataset(
        test_imgs, labels=None, transform=get_transforms("test")
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
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


def get_test_ids(load_cached_data=True):
    """
    Returns the IDs for the test set, ensuring alignment with the test_loader.
    """
    _, _, ids = load_data("test", load_cached_data)
    return ids
