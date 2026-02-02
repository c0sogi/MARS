import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from library.config import Config

# Ensure the working directory exists for caching
os.makedirs(Config.WORKING_DIR, exist_ok=True)


class CactusDataset(Dataset):
    """
    Custom Dataset for Cactus Classification.
    Holds data in memory as numpy arrays for efficiency.
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
        # Retrieve image (H, W, C) float32 [0, 1]
        image = self.images[idx]

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        # Return (image, label) for training/val, or (image, id) for test
        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return image, label
        else:
            img_id = self.ids[idx]
            return image, img_id


def get_transforms(phase="train"):
    """
    Returns the transformation pipeline for the specified phase.

    Args:
        phase (str): 'train', 'val', or 'test'.

    Returns:
        torchvision.transforms.Compose: Composed transforms.
    """
    if phase == "train":
        # Strictly RandomHorizontalFlip and RandomVerticalFlip as requested
        # ToTensor converts HWC numpy array to CHW tensor
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
            ]
        )
    else:
        # Just convert to tensor for validation/inference
        return transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )


def _load_and_cache_data(metadata_path, cache_prefix, load_cached_data=True):
    """
    Loads data from metadata/disk, with caching to .npy files to speed up subsequent runs.

    Args:
        metadata_path (str): Path to the metadata CSV file.
        cache_prefix (str): Prefix for the cached files (e.g., 'train', 'val').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (images, labels, ids)
    """
    cache_dir = Config.WORKING_DIR
    images_path = os.path.join(cache_dir, f"{cache_prefix}_images.npy")
    labels_path = os.path.join(cache_dir, f"{cache_prefix}_labels.npy")
    ids_path = os.path.join(cache_dir, f"{cache_prefix}_ids.npy")

    # 1. Attempt to load from cache
    if load_cached_data:
        # Check if essential files exist
        if os.path.exists(images_path) and os.path.exists(ids_path):
            # Determine if we expect labels (test set usually doesn't need them loaded this way)
            expect_labels = "test" not in cache_prefix
            if not expect_labels or os.path.exists(labels_path):
                images = np.load(images_path)
                ids = np.load(ids_path, allow_pickle=True)
                labels = np.load(labels_path) if expect_labels else None
                return images, labels, ids

    # 2. Process data from scratch if cache miss or reload requested
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)

    img_list = []
    label_list = []
    id_list = []

    for _, row in df.iterrows():
        # Construct full path
        rel_path = row["file_path"]
        full_path = os.path.join(Config.IMAGE_ROOT, rel_path)

        # Load image with OpenCV
        img = cv2.imread(full_path)
        if img is None:
            raise FileNotFoundError(f"Failed to load image: {full_path}")

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1] float32
        img = img.astype(np.float32) / 255.0

        img_list.append(img)
        id_list.append(row["id"])

        # Collect label if present
        if "has_cactus" in row:
            label_list.append(row["has_cactus"])

    # Convert to numpy arrays
    images = np.array(img_list)
    ids = np.array(id_list)
    labels = np.array(label_list) if label_list else None

    # 3. Save to cache
    np.save(images_path, images)
    np.save(ids_path, ids)
    if labels is not None:
        np.save(labels_path, labels)

    return images, labels, ids


def get_loaders(load_cached_data=True):
    """
    Generates DataLoaders for train, validation, and test sets.

    Args:
        load_cached_data (bool): Whether to use cached .npy files if available.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load data arrays
    train_imgs, train_lbls, _ = _load_and_cache_data(
        Config.TRAIN_META_PATH, "train", load_cached_data
    )
    val_imgs, val_lbls, _ = _load_and_cache_data(
        Config.VAL_META_PATH, "val", load_cached_data
    )
    test_imgs, _, test_ids = _load_and_cache_data(
        Config.TEST_META_PATH, "test", load_cached_data
    )

    # Apply debug subsampling if configured
    if Config.DEBUG_SAMPLE_SIZE is not None:
        train_imgs = train_imgs[: Config.DEBUG_SAMPLE_SIZE]
        train_lbls = train_lbls[: Config.DEBUG_SAMPLE_SIZE]
        val_imgs = val_imgs[: Config.DEBUG_SAMPLE_SIZE]
        val_lbls = val_lbls[: Config.DEBUG_SAMPLE_SIZE]
        test_imgs = test_imgs[: Config.DEBUG_SAMPLE_SIZE]
        test_ids = test_ids[: Config.DEBUG_SAMPLE_SIZE]

    # Instantiate Datasets
    train_dataset = CactusDataset(
        train_imgs, train_lbls, transform=get_transforms("train")
    )

    val_dataset = CactusDataset(val_imgs, val_lbls, transform=get_transforms("val"))

    test_dataset = CactusDataset(
        test_imgs, ids=test_ids, transform=get_transforms("test")
    )

    # Instantiate DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
