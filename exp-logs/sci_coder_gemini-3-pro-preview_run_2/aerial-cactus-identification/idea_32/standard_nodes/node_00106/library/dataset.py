import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from library.utils import cache_data

# Constants
INPUT_DIR = "./input"
CACHE_DIR = "./working/idea_32/"
METADATA_DIR = "./metadata"


class CactusDataset(Dataset):
    """
    PyTorch Dataset for Cactus images.
    """

    def __init__(self, data_dict, transform=None, mode="train"):
        """
        Args:
            data_dict (dict): Dictionary containing 'images', 'labels', and 'ids'.
            transform (callable, optional): Optional transform to be applied on a sample.
            mode (str): 'train', 'val', or 'test'. Determines what __getitem__ returns.
        """
        self.images = data_dict["images"]
        self.labels = data_dict["labels"]
        self.ids = data_dict["ids"]
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Image is (H, W, C) numpy array
        img = self.images[idx]

        # Apply transformations
        if self.transform:
            img = self.transform(img)

        if self.mode == "test":
            # For test, return image and ID for submission generation
            return img, self.ids[idx]
        else:
            # For train/val, return image and label
            label = self.labels[idx]
            return img, torch.tensor(label, dtype=torch.float32)


def _load_raw_data(metadata_path):
    """
    Reads metadata and loads images from disk.

    Args:
        metadata_path (str): Path to the metadata CSV file.

    Returns:
        dict: Dictionary with keys 'images', 'labels', 'ids'.
    """
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)

    images = []
    labels = []
    ids = []

    for _, row in df.iterrows():
        # Construct full path: ./input/train/xxxx.jpg
        full_path = os.path.join(INPUT_DIR, row["file_path"])

        # Load image in native resolution (32x32)
        img = cv2.imread(full_path)
        if img is None:
            # Skip missing files if any (though metadata check passed)
            continue

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        images.append(img)
        labels.append(row["has_cactus"])
        ids.append(row["id"])

    return {
        "images": np.array(images, dtype=np.uint8),
        "labels": np.array(labels, dtype=np.float32),
        "ids": np.array(ids),
    }


def get_transforms(split):
    """
    Returns the appropriate transformations for the given split.

    Args:
        split (str): 'train', 'val', or 'test'.

    Returns:
        torchvision.transforms.Compose: Composed transforms.
    """
    # Base transform: Convert to Tensor (scales [0, 255] -> [0.0, 1.0])
    transform_list = [transforms.ToTensor()]

    if split == "train":
        # Light augmentation for training
        transform_list.extend(
            [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()]
        )

    return transforms.Compose(transform_list)


def get_dataloaders(batch_size=64, num_workers=4, load_cached_data=True):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        batch_size (int): Batch size for the dataloaders.
        num_workers (int): Number of worker threads for data loading.
        load_cached_data (bool): Whether to try loading data from cache.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Ensure cache directory exists
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Define metadata paths
    train_meta = os.path.join(METADATA_DIR, "train_metadata.csv")
    val_meta = os.path.join(METADATA_DIR, "val_metadata.csv")
    test_meta = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Load data using caching mechanism
    train_data = cache_data(
        func=_load_raw_data,
        cache_dir=CACHE_DIR,
        cache_file="train_data.npy",
        load_cached_data=load_cached_data,
        metadata_path=train_meta,
    )

    val_data = cache_data(
        func=_load_raw_data,
        cache_dir=CACHE_DIR,
        cache_file="val_data.npy",
        load_cached_data=load_cached_data,
        metadata_path=val_meta,
    )

    test_data = cache_data(
        func=_load_raw_data,
        cache_dir=CACHE_DIR,
        cache_file="test_data.npy",
        load_cached_data=load_cached_data,
        metadata_path=test_meta,
    )

    # Create Datasets
    train_dataset = CactusDataset(
        train_data, transform=get_transforms("train"), mode="train"
    )

    val_dataset = CactusDataset(val_data, transform=get_transforms("val"), mode="val")

    test_dataset = CactusDataset(
        test_data, transform=get_transforms("test"), mode="test"
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
