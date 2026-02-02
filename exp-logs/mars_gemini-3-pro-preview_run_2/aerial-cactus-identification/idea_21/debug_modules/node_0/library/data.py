import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library import utils


def get_transforms(split):
    """
    Returns the Albumentations transformations for the given split.

    Args:
        split (str): 'train', 'val', or 'test'.

    Returns:
        A.Compose: Composed transformations.
    """
    if split == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=Config.AUG_HFLIP_PROB),
                A.VerticalFlip(p=Config.AUG_VFLIP_PROB),
                # Normalize to [0, 1] by dividing by 255.0. No mean subtraction.
                A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255.0),
                ToTensorV2(),
            ]
        )
    else:
        # Validation and Test: Only Normalize and ToTensor
        return A.Compose(
            [
                A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255.0),
                ToTensorV2(),
            ]
        )


def load_data_from_disk(split, load_cached_data=True):
    """
    Loads image data and labels from disk, utilizing caching to .npy files.

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (images_np, labels_np, ids_np)
    """
    # Ensure working directory exists for cache
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Define cache paths
    images_cache_path = os.path.join(Config.WORKING_DIR, f"{split}_images.npy")
    labels_cache_path = os.path.join(Config.WORKING_DIR, f"{split}_labels.npy")
    ids_cache_path = os.path.join(Config.WORKING_DIR, f"{split}_ids.npy")

    # Attempt to load from cache
    if load_cached_data:
        if (
            os.path.exists(images_cache_path)
            and os.path.exists(labels_cache_path)
            and os.path.exists(ids_cache_path)
        ):
            try:
                images = np.load(images_cache_path)
                labels = np.load(labels_cache_path)
                ids = np.load(ids_cache_path, allow_pickle=True)  # IDs are strings
                return images, labels, ids
            except Exception:
                # If loading fails, fall back to processing from scratch
                pass

    # Load metadata
    df = utils.load_metadata(split)

    # Handle Debugging Subset
    if Config.DEBUG_SUBSET_SIZE is not None:
        df = df.head(Config.DEBUG_SUBSET_SIZE)

    img_list = []
    label_list = []
    id_list = []

    for _, row in df.iterrows():
        # Construct full path: INPUT_DIR + relative path from metadata
        full_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # Read image
        img = cv2.imread(full_path)
        if img is None:
            raise FileNotFoundError(f"Image not found at {full_path}")

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_list.append(img)
        # For test set, this is a placeholder (0.5), but we load it anyway for consistency
        label_list.append(row["has_cactus"])
        id_list.append(row["id"])

    # Convert to numpy arrays
    images = np.array(img_list, dtype=np.uint8)
    labels = np.array(label_list, dtype=np.float32)
    ids = np.array(id_list)

    # Save to cache
    np.save(images_cache_path, images)
    np.save(labels_cache_path, labels)
    np.save(ids_cache_path, ids)

    return images, labels, ids


class CactusDataset(Dataset):
    """
    PyTorch Dataset for Cactus Identification.
    """

    def __init__(self, images, labels, transform=None):
        """
        Args:
            images (np.ndarray): Array of images (N, H, W, C).
            labels (np.ndarray): Array of labels (N,).
            transform (callable, optional): Albumentations transform pipeline.
        """
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            # Albumentations expects 'image' kwarg
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Return image and label (as tensor)
        return image, torch.tensor(label, dtype=torch.float32)


def get_dataloader(
    split, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS
):
    """
    Creates and returns a DataLoader for the specified split.

    Args:
        split (str): 'train', 'val', or 'test'.
        batch_size (int): Batch size.
        shuffle (bool): Whether to shuffle the data.
        num_workers (int): Number of worker processes.

    Returns:
        tuple: (DataLoader, ids_array)
    """
    # Load data (cached if available)
    images, labels, ids = load_data_from_disk(split, load_cached_data=True)

    # Get transforms
    transform = get_transforms(split)

    # Create Dataset
    dataset = CactusDataset(images, labels, transform=transform)

    # Create DataLoader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    return loader, ids
