import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

import library.config as config
import library.utils as utils


class CactusDataset(Dataset):
    """
    PyTorch Dataset for the Cactus classification task.
    """

    def __init__(self, images, labels=None, ids=None, transform=None):
        """
        Args:
            images (np.ndarray): Array of images with shape (N, H, W, C).
            labels (np.ndarray, optional): Array of binary labels.
            ids (np.ndarray, optional): Array of image IDs (filenames).
            transform (callable, optional): Transform to be applied on a sample.
        """
        self.images = images
        self.labels = labels
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image (H, W, C)
        img = self.images[idx]

        # Apply transforms
        if self.transform:
            img = self.transform(img)

        # Return based on available data
        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            if self.ids is not None:
                return img, label, self.ids[idx]
            return img, label
        elif self.ids is not None:
            return img, self.ids[idx]
        else:
            return img


def get_transforms(split: str):
    """
    Returns the transformations for the specific split.

    Args:
        split (str): 'train', 'val', or 'test'.
    """
    if split == "train":
        # Light augmentation: H-Flip and V-Flip only, then normalize to [0, 1] via ToTensor
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ToTensor(),  # Converts [0, 255] -> [0.0, 1.0]
            ]
        )
    else:
        # Validation/Test: Normalize to [0, 1] via ToTensor
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
            ]
        )


def load_and_cache_data(
    metadata_path, cache_prefix, load_cached_data=True, debug=False
):
    """
    Loads data from disk, caches it as .npy files, or loads from cache if available.

    Args:
        metadata_path (str): Path to the metadata CSV.
        cache_prefix (str): Prefix for the cached files (e.g., 'train', 'val').
        load_cached_data (bool): Whether to attempt loading from cache.
        debug (bool): If True, load only a small subset.

    Returns:
        tuple: (images, labels, ids) or (images, None, ids)
    """
    # Ensure working directory exists
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    images_cache_path = os.path.join(config.WORKING_DIR, f"{cache_prefix}_images.npy")
    labels_cache_path = os.path.join(config.WORKING_DIR, f"{cache_prefix}_labels.npy")
    ids_cache_path = os.path.join(config.WORKING_DIR, f"{cache_prefix}_ids.npy")

    # 1. Try to load from cache
    if (
        load_cached_data
        and os.path.exists(images_cache_path)
        and os.path.exists(ids_cache_path)
    ):
        # Check labels cache existence only if we expect labels (train/val)
        # However, for simplicity, we check if labels cache exists.
        # If it's test set, labels cache might not exist or be irrelevant.
        # We'll rely on the file existence.

        print(f"Loading {cache_prefix} data from cache...")
        images = np.load(images_cache_path)
        ids = np.load(ids_cache_path, allow_pickle=True)  # IDs are strings

        labels = None
        if os.path.exists(labels_cache_path):
            labels = np.load(labels_cache_path)

        if debug:
            print(
                f"Debug mode: trimming cached {cache_prefix} data to {config.DEBUG_SAMPLE_SIZE} samples."
            )
            images = images[: config.DEBUG_SAMPLE_SIZE]
            ids = ids[: config.DEBUG_SAMPLE_SIZE]
            if labels is not None:
                labels = labels[: config.DEBUG_SAMPLE_SIZE]

        return images, labels, ids

    # 2. Process from scratch
    print(
        f"Processing {cache_prefix} data from scratch (Cache miss or force reload)..."
    )

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)

    if debug:
        print(f"Debug mode: sampling {config.DEBUG_SAMPLE_SIZE} rows from metadata.")
        df = df.head(config.DEBUG_SAMPLE_SIZE)

    image_list = []
    label_list = []
    id_list = []

    # Pre-allocate if possible? Images are small, list append is fine for ~15k items.
    # Iterating rows
    for _, row in df.iterrows():
        # Construct full path
        # metadata contains relative path 'train/xxxx.jpg' or 'test/xxxx.jpg'
        rel_path = row["file_path"]
        full_path = os.path.join(config.INPUT_DIR, rel_path)

        # Load image
        img = cv2.imread(full_path)
        if img is None:
            print(f"Warning: Could not read image {full_path}. Skipping.")
            continue

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image_list.append(img)

        # Store ID
        id_list.append(row["id"])

        # Store Label if present
        if "has_cactus" in row:
            label_list.append(row["has_cactus"])

    # Convert to numpy arrays
    images = np.array(image_list, dtype=np.uint8)
    ids = np.array(id_list)

    # Save to cache
    np.save(images_cache_path, images)
    np.save(ids_cache_path, ids)

    labels = None
    if label_list:
        labels = np.array(label_list, dtype=np.float32)
        np.save(labels_cache_path, labels)

    print(f"Saved {cache_prefix} data to cache at {config.WORKING_DIR}")

    return images, labels, ids


def get_datasets(load_cached_data=True):
    """
    Prepares and returns the training, validation, and test datasets.

    Args:
        load_cached_data (bool): Whether to use cached numpy files.

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    debug_mode = config.DEBUG

    # Load Train
    train_imgs, train_lbls, train_ids = load_and_cache_data(
        config.TRAIN_METADATA_PATH,
        "train",
        load_cached_data=load_cached_data,
        debug=debug_mode,
    )

    # Load Val
    val_imgs, val_lbls, val_ids = load_and_cache_data(
        config.VAL_METADATA_PATH,
        "val",
        load_cached_data=load_cached_data,
        debug=debug_mode,
    )

    # Load Test
    test_imgs, _, test_ids = load_and_cache_data(
        config.TEST_METADATA_PATH,
        "test",
        load_cached_data=load_cached_data,
        debug=debug_mode,
    )

    # Create Datasets
    train_dataset = CactusDataset(
        train_imgs, train_lbls, ids=train_ids, transform=get_transforms("train")
    )

    val_dataset = CactusDataset(
        val_imgs, val_lbls, ids=val_ids, transform=get_transforms("val")
    )

    test_dataset = CactusDataset(
        test_imgs,
        labels=None,  # Test set has no ground truth for prediction
        ids=test_ids,
        transform=get_transforms("test"),
    )

    return train_dataset, val_dataset, test_dataset


def get_dataloaders(batch_size=None, num_workers=None, load_cached_data=True):
    """
    Helper to get DataLoaders directly.
    """
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    if num_workers is None:
        num_workers = config.NUM_WORKERS

    train_ds, val_ds, test_ds = get_datasets(load_cached_data=load_cached_data)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
