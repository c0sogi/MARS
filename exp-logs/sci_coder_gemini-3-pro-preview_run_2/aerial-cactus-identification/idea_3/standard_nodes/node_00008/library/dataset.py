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
    Custom Dataset for Cactus Classification.
    """

    def __init__(self, images, labels=None, transform=None):
        """
        Args:
            images (np.ndarray): Array of images with shape (N, H, W, C).
            labels (np.ndarray, optional): Array of labels with shape (N,).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Images are stored as (H, W, C) uint8
        image = self.images[idx]

        # Apply transforms
        # ToTensor converts (H, W, C) [0, 255] -> (C, H, W) [0.0, 1.0]
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        if self.labels is not None:
            label = self.labels[idx]
            # Return float tensor for BCEWithLogitsLoss
            return image, torch.tensor(label, dtype=torch.float32)
        else:
            # For test set, return a placeholder label
            return image, torch.tensor(-1.0, dtype=torch.float32)


def get_transforms(split="train"):
    """
    Returns the transformations for the given split.

    Args:
        split (str): 'train', 'val', or 'test'.

    Returns:
        torchvision.transforms.Compose: The composition of transforms.
    """
    if split == "train":
        # Light augmentation as per requirements
        return transforms.Compose(
            [
                transforms.ToTensor(),  # Converts numpy (H, W, C) to tensor (C, H, W) in [0, 1]
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ]
        )
    else:
        # Validation and Test: Only normalization/conversion
        return transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )


def _load_and_cache_data(metadata_path, cache_name, load_cached_data):
    """
    Internal helper to load data from disk or cache.

    Args:
        metadata_path (str): Path to the metadata CSV.
        cache_name (str): Prefix for the cache files (e.g., 'train', 'val', 'test').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (images_np, labels_np)
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    images_cache_path = os.path.join(Config.WORKING_DIR, f"{cache_name}_images.npy")
    labels_cache_path = os.path.join(Config.WORKING_DIR, f"{cache_name}_labels.npy")

    # 1. Try to load from cache
    if load_cached_data:
        if os.path.exists(images_cache_path) and os.path.exists(labels_cache_path):
            images = np.load(images_cache_path)
            labels = np.load(labels_cache_path)
            return images, labels

    # 2. Process from scratch
    df = pd.read_csv(metadata_path)
    image_list = []
    label_list = []

    for _, row in df.iterrows():
        # Construct full path
        # Metadata file_path is relative to input dir (e.g., "train/xxx.jpg")
        full_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # Load image
        img = cv2.imread(full_path)
        if img is None:
            # Fallback for safety, though data should exist
            img = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8)
        else:
            # BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        image_list.append(img)
        label_list.append(row["has_cactus"])

    images = np.array(image_list, dtype=np.uint8)
    labels = np.array(label_list, dtype=np.float32)

    # Save to cache
    np.save(images_cache_path, images)
    np.save(labels_cache_path, labels)

    return images, labels


def get_dataloaders(
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    load_cached_data=True,
    debug=Config.DEBUG,
):
    """
    Creates and returns DataLoaders for train, val, and test sets.

    Args:
        batch_size (int): Batch size.
        num_workers (int): Number of workers.
        load_cached_data (bool): Whether to use cached numpy files.
        debug (bool): If True, subsets the data for quick debugging.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """

    # Load Data
    train_imgs, train_lbls = _load_and_cache_data(
        Config.TRAIN_METADATA_PATH, "train", load_cached_data
    )
    val_imgs, val_lbls = _load_and_cache_data(
        Config.VAL_METADATA_PATH, "val", load_cached_data
    )
    test_imgs, test_lbls = _load_and_cache_data(
        Config.TEST_METADATA_PATH, "test", load_cached_data
    )

    # Debug Mode
    if debug:
        subset_size = Config.DEBUG_SAMPLE_SIZE
        train_imgs = train_imgs[:subset_size]
        train_lbls = train_lbls[:subset_size]
        val_imgs = val_imgs[:subset_size]
        val_lbls = val_lbls[:subset_size]
        test_imgs = test_imgs[:subset_size]
        test_lbls = test_lbls[:subset_size]

    # Create Datasets
    train_dataset = CactusDataset(
        train_imgs, train_lbls, transform=get_transforms("train")
    )

    val_dataset = CactusDataset(val_imgs, val_lbls, transform=get_transforms("val"))

    test_dataset = CactusDataset(
        test_imgs, test_lbls, transform=get_transforms("test")  # Placeholders
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True,  # Helps with batch norm stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # Must be False to maintain ID order
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return train_loader, val_loader, test_loader
