import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from library.utils import set_seed

# Constants
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
CACHE_DIR = "./working/idea_8"


class CactusDataset(Dataset):
    """
    Custom Dataset for Cactus Identification.
    """

    def __init__(self, images: np.ndarray, labels: np.ndarray = None, transform=None):
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
        # Image is (H, W, C)
        image = self.images[idx]

        if self.transform:
            image = self.transform(image)

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return image, label

        # Return a dummy label for test set if needed, or just the image
        # Returning dummy label 0.0 to maintain consistency in loop signatures
        return image, torch.tensor(0.0)


def get_transforms(split: str):
    """
    Returns the appropriate transforms for the given split.

    Args:
        split (str): 'train', 'val', or 'test'.
    """
    if split == "train":
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ToTensor(),  # Converts to [0, 1]
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),  # Converts to [0, 1]
            ]
        )


def _load_raw_data(metadata_path: str):
    """
    Loads images and labels based on metadata CSV.
    """
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)

    images = []
    labels = []
    ids = []

    for _, row in df.iterrows():
        rel_path = row["file_path"]
        full_path = os.path.join(INPUT_DIR, rel_path)

        # Load image
        img = cv2.imread(full_path)
        if img is None:
            raise ValueError(f"Failed to load image: {full_path}")

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)

        ids.append(row["id"])

        # Load label if available
        if "has_cactus" in row:
            labels.append(row["has_cactus"])

    images_np = np.array(images, dtype=np.uint8)
    ids_np = np.array(ids)
    labels_np = np.array(labels, dtype=np.float32) if labels else None

    return images_np, labels_np, ids_np


def get_data_arrays(split: str, load_cached_data: bool = True):
    """
    Retrieves data arrays, using cache if available and requested.

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to use cached .npy files.

    Returns:
        tuple: (images, labels, ids)
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    img_cache_path = os.path.join(CACHE_DIR, f"{split}_images.npy")
    lbl_cache_path = os.path.join(CACHE_DIR, f"{split}_labels.npy")
    id_cache_path = os.path.join(CACHE_DIR, f"{split}_ids.npy")

    # Check cache
    if load_cached_data:
        has_img = os.path.exists(img_cache_path)
        has_id = os.path.exists(id_cache_path)

        if has_img and has_id:
            try:
                images = np.load(img_cache_path)
                ids = np.load(id_cache_path)
                labels = None
                if os.path.exists(lbl_cache_path):
                    labels = np.load(lbl_cache_path)
                return images, labels, ids
            except Exception as e:
                print(f"Error loading cache for {split}: {e}. Recomputing...")

    # Process from scratch
    meta_path = os.path.join(METADATA_DIR, f"{split}_metadata.csv")
    images, labels, ids = _load_raw_data(meta_path)

    # Save to cache
    np.save(img_cache_path, images)
    np.save(id_cache_path, ids)
    if labels is not None:
        np.save(lbl_cache_path, labels)

    return images, labels, ids


def get_dataloaders(
    batch_size: int = 32,
    num_workers: int = 2,
    load_cached_data: bool = True,
    seed: int = 42,
):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        batch_size (int): Batch size.
        num_workers (int): Number of worker threads.
        load_cached_data (bool): Whether to use cached data.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: (train_loader, val_loader, test_loader, test_ids)
    """
    set_seed(seed)

    # Load Data Arrays
    train_imgs, train_lbls, _ = get_data_arrays("train", load_cached_data)
    val_imgs, val_lbls, _ = get_data_arrays("val", load_cached_data)
    test_imgs, _, test_ids = get_data_arrays("test", load_cached_data)

    # Create Datasets
    train_dataset = CactusDataset(
        train_imgs, train_lbls, transform=get_transforms("train")
    )

    val_dataset = CactusDataset(val_imgs, val_lbls, transform=get_transforms("val"))

    # For test, we ignore the labels (which are placeholders)
    test_dataset = CactusDataset(
        test_imgs, labels=None, transform=get_transforms("test")
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
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
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return train_loader, val_loader, test_loader, test_ids
