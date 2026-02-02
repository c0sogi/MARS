import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from library.config import (
    INPUT_DIR,
    WORKING_DIR,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    IMAGE_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    DEBUG,
)


def get_transforms(phase: str):
    """
    Returns the data transformations for the specified phase.

    Args:
        phase (str): 'train', 'val', or 'test'.

    Returns:
        torchvision.transforms.Compose: The composition of transforms.
    """
    if phase == "train":
        return transforms.Compose(
            [
                transforms.ToTensor(),  # Converts HWC [0,255] to CHW [0.0, 1.0]
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.ToTensor(),  # Converts HWC [0,255] to CHW [0.0, 1.0]
            ]
        )


def load_and_cache_data(
    metadata_path, cache_prefix, load_cached_data=True, is_test=False
):
    """
    Loads data from metadata CSV, reads images, and caches them as .npy files.

    Args:
        metadata_path (str): Path to the metadata CSV file.
        cache_prefix (str): Prefix for the cached files (e.g., 'train', 'val').
        load_cached_data (bool): Whether to attempt loading from cache.
        is_test (bool): If True, labels might be placeholders.

    Returns:
        tuple: (images_array, labels_array, ids_array)
    """
    os.makedirs(WORKING_DIR, exist_ok=True)

    images_cache_path = os.path.join(WORKING_DIR, f"{cache_prefix}_images.npy")
    labels_cache_path = os.path.join(WORKING_DIR, f"{cache_prefix}_labels.npy")
    ids_cache_path = os.path.join(WORKING_DIR, f"{cache_prefix}_ids.npy")

    # 1. Attempt to load from cache
    if load_cached_data:
        # Check if essential files exist
        if os.path.exists(images_cache_path) and os.path.exists(ids_cache_path):
            # For test sets, labels might be placeholders, but we check existence if not strictly test
            if is_test or os.path.exists(labels_cache_path):
                try:
                    images = np.load(images_cache_path)
                    ids = np.load(ids_cache_path)
                    if os.path.exists(labels_cache_path):
                        labels = np.load(labels_cache_path)
                    else:
                        labels = None
                    return images, labels, ids
                except Exception:
                    # If load fails (corrupt file), fall through to re-processing
                    pass

    # 2. Process from scratch
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)

    if DEBUG:
        df = df.head(100)  # Limit data for debugging

    img_list = []
    label_list = []
    id_list = []

    for idx, row in df.iterrows():
        # Metadata contains relative path, e.g., "train/xxx.jpg"
        full_path = os.path.join(INPUT_DIR, row["file_path"])

        # Read image
        img = cv2.imread(full_path)
        if img is None:
            continue

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Ensure 32x32 dimensions
        if img.shape[0] != IMAGE_SIZE or img.shape[1] != IMAGE_SIZE:
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

        img_list.append(img)
        id_list.append(row["id"])

        # Handle labels
        if "has_cactus" in row:
            label_list.append(row["has_cactus"])
        else:
            label_list.append(0.5)  # Placeholder

    # Convert to numpy arrays
    images = np.array(img_list, dtype=np.uint8)
    ids = np.array(id_list)
    labels = np.array(label_list, dtype=np.float32)

    # 3. Save to cache
    np.save(images_cache_path, images)
    np.save(ids_cache_path, ids)
    np.save(labels_cache_path, labels)

    return images, labels, ids


class CactusDataset(Dataset):
    def __init__(self, images, labels, ids, transform=None):
        self.images = images
        self.labels = labels
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Image is HWC, uint8
        img = self.images[idx]
        label = self.labels[idx]
        img_id = self.ids[idx]

        if self.transform:
            img = self.transform(img)

        # Return image, label tensor, and id
        return img, torch.tensor(label, dtype=torch.float32), img_id


def get_dataloaders(load_cached_data=True):
    """
    Prepares datasets and returns DataLoaders for train, val, and test.

    Args:
        load_cached_data (bool): Whether to use cached .npy files.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """

    # 1. Prepare Data Arrays
    train_imgs, train_lbls, train_ids = load_and_cache_data(
        TRAIN_METADATA_PATH, "train", load_cached_data=load_cached_data
    )
    val_imgs, val_lbls, val_ids = load_and_cache_data(
        VAL_METADATA_PATH, "val", load_cached_data=load_cached_data
    )
    test_imgs, test_lbls, test_ids = load_and_cache_data(
        TEST_METADATA_PATH, "test", load_cached_data=load_cached_data, is_test=True
    )

    # 2. Create Dataset Instances
    train_dataset = CactusDataset(
        train_imgs, train_lbls, train_ids, transform=get_transforms("train")
    )
    val_dataset = CactusDataset(
        val_imgs, val_lbls, val_ids, transform=get_transforms("val")
    )
    test_dataset = CactusDataset(
        test_imgs, test_lbls, test_ids, transform=get_transforms("test")
    )

    # 3. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
