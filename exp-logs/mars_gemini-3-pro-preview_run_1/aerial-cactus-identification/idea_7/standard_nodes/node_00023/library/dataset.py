import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from library.config import Config


class CactusDataset(Dataset):
    """
    PyTorch Dataset for the Cactus Identification task.
    Wraps in-memory numpy arrays for high-throughput access.
    """

    def __init__(self, images, targets=None, transform=None, is_test=False, ids=None):
        """
        Args:
            images (np.ndarray): Array of shape (N, C, H, W) containing float32 images.
            targets (np.ndarray, optional): Array of shape (N,) containing labels.
            transform (callable, optional): Transformations to apply to the images.
            is_test (bool): If True, returns (image, id) instead of (image, label).
            ids (np.ndarray, optional): Array of shape (N,) containing image IDs (filenames).
        """
        self.images = images
        self.targets = targets
        self.transform = transform
        self.is_test = is_test
        self.ids = ids

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Convert numpy array to torch tensor
        # Images are already (C, H, W) float32 in [0, 1]
        img = torch.from_numpy(self.images[idx])

        if self.transform:
            img = self.transform(img)

        if self.is_test:
            # For test set, return image and ID
            img_id = self.ids[idx]
            return img, img_id
        else:
            # For train/val set, return image and label
            label = torch.tensor(self.targets[idx], dtype=torch.float32)
            return img, label


def load_data_to_memory(
    metadata_path,
    cache_imgs_path,
    cache_labels_path=None,
    cache_ids_path=None,
    load_cached_data=True,
):
    """
    Loads images and labels/ids into memory, using disk caching to speed up subsequent runs.

    Args:
        metadata_path (str): Path to the metadata CSV.
        cache_imgs_path (str): Path to save/load cached image array.
        cache_labels_path (str, optional): Path to save/load cached label array.
        cache_ids_path (str, optional): Path to save/load cached ID array (for test set).
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (images, labels) or (images, ids) depending on arguments.
    """
    # Ensure working directory exists
    os.makedirs(os.path.dirname(cache_imgs_path), exist_ok=True)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_imgs_path):
        print(f"Loading cached data from {cache_imgs_path}...")
        images = np.load(cache_imgs_path)

        second_item = None
        if cache_labels_path and os.path.exists(cache_labels_path):
            second_item = np.load(cache_labels_path)
        elif cache_ids_path and os.path.exists(cache_ids_path):
            second_item = np.load(cache_ids_path)

        if second_item is not None:
            return images, second_item
        else:
            print("Cache incomplete, reloading from source...")

    # 2. Process from source
    print(f"Processing data from {metadata_path}...")
    df = pd.read_csv(metadata_path)

    img_list = []
    label_list = []
    id_list = []

    # Pre-construct full paths
    # Metadata file_path is relative to input dir (e.g., "train/xxx.jpg")
    file_paths = (
        df["file_path"].apply(lambda x: os.path.join(Config.INPUT_DIR, x)).values
    )
    ids = df["id"].values

    has_labels = "has_cactus" in df.columns and cache_labels_path is not None
    if has_labels:
        labels = df["has_cactus"].values

    count = 0
    for idx, path in enumerate(file_paths):
        # Read image
        img = cv2.imread(path)
        if img is None:
            print(f"Warning: Could not read image {path}")
            continue

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1] float32
        img = img.astype(np.float32) / 255.0

        # Transpose to (C, H, W)
        img = img.transpose(2, 0, 1)

        img_list.append(img)
        id_list.append(ids[idx])

        if has_labels:
            label_list.append(labels[idx])

        count += 1
        if count % 5000 == 0:
            print(f"Processed {count} images...")

    # Stack into numpy arrays
    images_np = np.stack(img_list)

    # Save to cache
    np.save(cache_imgs_path, images_np)
    print(f"Saved images cache to {cache_imgs_path}")

    if has_labels:
        labels_np = np.array(label_list, dtype=np.float32)
        if cache_labels_path:
            np.save(cache_labels_path, labels_np)
        return images_np, labels_np
    else:
        ids_np = np.array(id_list)
        if cache_ids_path:
            np.save(cache_ids_path, ids_np)
        return images_np, ids_np


def get_dataloaders(
    batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, load_cached=True
):
    """
    Creates DataLoaders for training and validation sets.

    Args:
        batch_size (int): Batch size.
        num_workers (int): Number of worker threads.
        load_cached (bool): Whether to use cached data.

    Returns:
        tuple: (train_loader, val_loader)
    """
    # Load Training Data
    train_imgs, train_labels = load_data_to_memory(
        Config.TRAIN_METADATA,
        Config.CACHE_TRAIN_IMGS,
        cache_labels_path=Config.CACHE_TRAIN_LABELS,
        load_cached_data=load_cached,
    )

    # Load Validation Data
    val_imgs, val_labels = load_data_to_memory(
        Config.VAL_METADATA,
        Config.CACHE_VAL_IMGS,
        cache_labels_path=Config.CACHE_VAL_LABELS,
        load_cached_data=load_cached,
    )

    # Define Transforms
    # Images are tensors (C, H, W). Geometric augmentations work on tensors.
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
        ]
    )

    # No transforms for validation (already normalized and transposed)
    val_transform = None

    # Create Datasets
    train_dataset = CactusDataset(train_imgs, train_labels, transform=train_transform)
    val_dataset = CactusDataset(val_imgs, val_labels, transform=val_transform)

    # Create Loaders
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

    return train_loader, val_loader


def get_test_dataloader(
    batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, load_cached=True
):
    """
    Creates a DataLoader for the test set.

    Args:
        batch_size (int): Batch size.
        num_workers (int): Number of worker threads.
        load_cached (bool): Whether to use cached data.

    Returns:
        DataLoader: Test data loader yielding (image, id).
    """
    test_imgs, test_ids = load_data_to_memory(
        Config.TEST_METADATA,
        Config.CACHE_TEST_IMGS,
        cache_ids_path=Config.CACHE_TEST_IDS,
        load_cached_data=load_cached,
    )

    test_dataset = CactusDataset(
        test_imgs, targets=None, transform=None, is_test=True, ids=test_ids
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return test_loader
