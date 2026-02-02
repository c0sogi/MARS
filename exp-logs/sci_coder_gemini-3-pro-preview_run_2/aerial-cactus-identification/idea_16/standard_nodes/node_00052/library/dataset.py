import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from library.utils import seed_everything

# Configuration Constants
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
CACHE_DIR = "./working/idea_16"


class CactusDataset(Dataset):
    """
    Custom Dataset for Cactus Identification.
    """

    def __init__(self, images, labels=None, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Image is (H, W, C) uint8
        img = self.images[idx]

        # Apply transformations
        # ToTensor converts (H,W,C) [0,255] -> (C,H,W) [0.0,1.0]
        if self.transform:
            img = self.transform(img)

        if self.labels is not None:
            # Label is binary 0 or 1
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return img, label
        else:
            return img


def get_transforms(split: str):
    """
    Returns transformations based on the data split.
    Native resolution 32x32 is preserved.
    Normalization to [0, 1] is handled by ToTensor().
    """
    if split == "train":
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ]
        )
    else:
        # Validation and Test
        return transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )


def load_data(metadata_file, cache_prefix, load_cached_data=True):
    """
    Loads data from metadata CSVs. Implements caching using .npy files.
    """
    # Ensure cache directory exists
    os.makedirs(CACHE_DIR, exist_ok=True)

    img_cache_path = os.path.join(CACHE_DIR, f"{cache_prefix}_images.npy")
    lbl_cache_path = os.path.join(CACHE_DIR, f"{cache_prefix}_labels.npy")
    id_cache_path = os.path.join(CACHE_DIR, f"{cache_prefix}_ids.npy")

    # 1. Try to load from cache
    if load_cached_data:
        if os.path.exists(img_cache_path) and os.path.exists(id_cache_path):
            try:
                images = np.load(img_cache_path)
                ids = np.load(id_cache_path)
                labels = None
                if os.path.exists(lbl_cache_path):
                    labels = np.load(lbl_cache_path)

                # Simple validation to ensure integrity
                if len(images) == len(ids):
                    return images, labels, ids
            except Exception:
                # If load fails, fall through to processing
                pass

    # 2. Process from scratch
    metadata_path = os.path.join(METADATA_DIR, metadata_file)
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)

    images = []
    ids = []
    labels = []

    for _, row in df.iterrows():
        rel_path = row["file_path"]
        img_id = row["id"]

        # Construct full path
        full_path = os.path.join(INPUT_DIR, rel_path)

        # Read image
        img = cv2.imread(full_path)
        if img is None:
            continue

        # BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        images.append(img)
        ids.append(img_id)

        if "has_cactus" in row:
            labels.append(row["has_cactus"])

    # Convert to numpy arrays
    images = np.array(images, dtype=np.uint8)
    ids = np.array(ids)

    # Save to cache
    np.save(img_cache_path, images)
    np.save(id_cache_path, ids)

    if labels:
        labels = np.array(labels, dtype=np.float32)
        np.save(lbl_cache_path, labels)
    else:
        labels = None

    return images, labels, ids


def get_dataloaders(batch_size=32, num_workers=2, load_cached_data=True):
    """
    Creates DataLoaders for train, validation, and test sets.
    """
    seed_everything(42)

    # Load Data
    train_imgs, train_lbls, _ = load_data(
        "train_metadata.csv", "train", load_cached_data
    )
    val_imgs, val_lbls, _ = load_data("val_metadata.csv", "val", load_cached_data)
    test_imgs, _, test_ids = load_data("test_metadata.csv", "test", load_cached_data)

    # Create Datasets
    train_dataset = CactusDataset(
        train_imgs, train_lbls, transform=get_transforms("train")
    )
    val_dataset = CactusDataset(val_imgs, val_lbls, transform=get_transforms("val"))
    # For test, we don't pass labels to the dataset
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

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "test_ids": test_ids,
    }
