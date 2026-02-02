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
    Custom Dataset for loading Cactus images.
    Stores images in memory as numpy arrays for efficiency.
    """

    def __init__(self, images, labels, ids, transform=None):
        self.images = images
        self.labels = labels
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Retrieve image (H, W, C) uint8
        image = self.images[idx]

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        # Retrieve label
        label = self.labels[idx]

        # Return image and label (float32 for BCE loss)
        return image, torch.tensor(label, dtype=torch.float32)


def _load_and_cache_data(metadata_path, cache_prefix, load_cached_data=True):
    """
    Loads data from metadata CSV and images from disk.
    Implements caching using .npy files.
    """
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    img_cache_path = os.path.join(cache_dir, f"{cache_prefix}_images.npy")
    lbl_cache_path = os.path.join(cache_dir, f"{cache_prefix}_labels.npy")
    ids_cache_path = os.path.join(cache_dir, f"{cache_prefix}_ids.npy")

    # 1. Try to load from cache
    if load_cached_data:
        if (
            os.path.exists(img_cache_path)
            and os.path.exists(lbl_cache_path)
            and os.path.exists(ids_cache_path)
        ):
            try:
                images = np.load(img_cache_path)
                labels = np.load(lbl_cache_path)
                ids = np.load(ids_cache_path)
                return images, labels, ids
            except Exception:
                # If loading fails, fall through to processing from scratch
                pass

    # 2. Process from scratch
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)

    images_list = []
    labels_list = []
    ids_list = []

    # Pre-allocate if possible or just append (dataset is small enough for append)
    for _, row in df.iterrows():
        # Construct full path: input_dir + relative_path_from_metadata
        full_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # Read image using OpenCV
        img = cv2.imread(full_path)
        if img is None:
            continue

        # Convert BGR (OpenCV default) to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        images_list.append(img)
        labels_list.append(row["has_cactus"])
        ids_list.append(row["id"])

    # Convert to numpy arrays
    images = np.array(images_list, dtype=np.uint8)
    labels = np.array(labels_list, dtype=np.float32)
    ids = np.array(ids_list)

    # 3. Save to cache
    np.save(img_cache_path, images)
    np.save(lbl_cache_path, labels)
    np.save(ids_cache_path, ids)

    return images, labels, ids


def get_transforms(mode="train"):
    """
    Returns torchvision transforms based on the mode.
    """
    if mode == "train":
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ToTensor(),  # Converts [0, 255] -> [0.0, 1.0]
            ]
        )
    else:
        # Validation and Test
        return transforms.Compose(
            [
                transforms.ToTensor(),  # Converts [0, 255] -> [0.0, 1.0]
            ]
        )


def get_datasets(load_cached_data=True):
    """
    Constructs and returns the training, validation, and test datasets.
    """
    # Load data arrays
    train_imgs, train_lbls, train_ids = _load_and_cache_data(
        Config.TRAIN_METADATA_PATH, "train", load_cached_data
    )
    val_imgs, val_lbls, val_ids = _load_and_cache_data(
        Config.VAL_METADATA_PATH, "val", load_cached_data
    )
    test_imgs, test_lbls, test_ids = _load_and_cache_data(
        Config.TEST_METADATA_PATH, "test", load_cached_data
    )

    # Handle Debug Mode
    if Config.DEBUG:
        limit = Config.DEBUG_SAMPLE_SIZE
        train_imgs, train_lbls, train_ids = (
            train_imgs[:limit],
            train_lbls[:limit],
            train_ids[:limit],
        )
        val_imgs, val_lbls, val_ids = (
            val_imgs[:limit],
            val_lbls[:limit],
            val_ids[:limit],
        )
        test_imgs, test_lbls, test_ids = (
            test_imgs[:limit],
            test_lbls[:limit],
            test_ids[:limit],
        )

    # Instantiate Datasets
    train_ds = CactusDataset(
        train_imgs, train_lbls, train_ids, transform=get_transforms("train")
    )
    val_ds = CactusDataset(val_imgs, val_lbls, val_ids, transform=get_transforms("val"))
    test_ds = CactusDataset(
        test_imgs, test_lbls, test_ids, transform=get_transforms("test")
    )

    return train_ds, val_ds, test_ds


def get_dataloaders(load_cached_data=True):
    """
    Constructs and returns DataLoaders for train, val, and test.
    """
    train_ds, val_ds, test_ds = get_datasets(load_cached_data)

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
