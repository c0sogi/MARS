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
    Custom Dataset for Cactus Identification.
    Loads images from numpy arrays (cached in memory).
    """

    def __init__(self, images, labels=None, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Image is (H, W, C)
        img = self.images[idx]

        # Apply transforms
        # ToTensor converts HWC [0, 255] to CHW [0.0, 1.0]
        if self.transform:
            img = self.transform(img)

        # Return image and label
        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return img, label
        else:
            # Return dummy label for test set consistency
            return img, torch.tensor(0.0, dtype=torch.float32)


def get_transforms(split="train"):
    """
    Returns the data transformations for the given split.
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


def _load_and_cache_data(metadata_path, prefix, load_cached_data=True):
    """
    Loads data from metadata CSV and images, with caching to .npy files.
    Strictly follows the caching logic requirement.
    """
    # Define cache paths
    cache_dir = Config.WORKING_DIR
    ids_path = os.path.join(cache_dir, f"{prefix}_ids.npy")
    images_path = os.path.join(cache_dir, f"{prefix}_images.npy")
    labels_path = os.path.join(cache_dir, f"{prefix}_labels.npy")

    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)

    # 1. Try loading from cache
    if load_cached_data:
        # We check for IDs and Images. Labels might be optional (e.g. test set).
        if os.path.exists(ids_path) and os.path.exists(images_path):
            try:
                ids = np.load(ids_path, allow_pickle=True)
                images = np.load(images_path)
                if os.path.exists(labels_path):
                    labels = np.load(labels_path)
                else:
                    labels = None
                return ids, images, labels
            except Exception:
                # If loading fails (corrupt file), proceed to process from scratch
                pass

    # 2. Process from scratch
    df = pd.read_csv(metadata_path)

    ids_list = []
    images_list = []
    labels_list = []

    for _, row in df.iterrows():
        # Construct full path (relative to INPUT_DIR)
        full_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # Read image
        img = cv2.imread(full_path)
        if img is None:
            continue

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ids_list.append(row["id"])
        images_list.append(img)

        if "has_cactus" in row:
            labels_list.append(row["has_cactus"])

    # Convert to numpy arrays
    ids = np.array(ids_list)
    images = np.array(images_list)
    labels = np.array(labels_list) if labels_list else None

    # Save to cache
    np.save(ids_path, ids)
    np.save(images_path, images)
    if labels is not None:
        np.save(labels_path, labels)

    # 3. Return data
    return ids, images, labels


def get_dataloaders(load_cached_data=True):
    """
    Creates and returns DataLoaders for train, val, and test sets.
    Also returns test_ids for submission.
    """
    # Load data
    train_ids, train_imgs, train_lbls = _load_and_cache_data(
        Config.TRAIN_METADATA_PATH, "train", load_cached_data
    )
    val_ids, val_imgs, val_lbls = _load_and_cache_data(
        Config.VAL_METADATA_PATH, "val", load_cached_data
    )
    test_ids, test_imgs, _ = _load_and_cache_data(
        Config.TEST_METADATA_PATH, "test", load_cached_data
    )

    # Create Datasets
    train_dataset = CactusDataset(
        train_imgs, train_lbls, transform=get_transforms("train")
    )
    val_dataset = CactusDataset(val_imgs, val_lbls, transform=get_transforms("val"))
    # Note: test_dataset has no labels, will return dummy labels
    test_dataset = CactusDataset(test_imgs, None, transform=get_transforms("test"))

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    return train_loader, val_loader, test_loader, test_ids
