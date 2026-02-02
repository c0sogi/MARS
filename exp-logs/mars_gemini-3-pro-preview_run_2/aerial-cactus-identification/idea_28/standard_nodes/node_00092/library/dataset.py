import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# --- Configuration ---
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_28"


def get_transforms(split="train"):
    """
    Returns the transformations for the given split.
    Strictly adheres to 'light augmentation': Random Horizontal and Vertical Flips only.

    Args:
        split (str): 'train', 'val', or 'test'.

    Returns:
        torchvision.transforms.Compose: The transform pipeline.
    """
    if split == "train":
        return transforms.Compose(
            [
                transforms.ToTensor(),  # Converts HWC [0, 255] to CHW [0.0, 1.0]
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
            ]
        )
    else:
        # For val and test
        return transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )


class CactusDataset(Dataset):
    def __init__(self, images, labels=None, transform=None):
        """
        Custom Dataset for Cactus Images.

        Args:
            images (np.ndarray): Array of images with shape (N, H, W, C).
            labels (np.ndarray, optional): Array of labels with shape (N,).
            transform (callable, optional): Transform to be applied on a sample.
        """
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Retrieve image (HWC, RGB)
        img = self.images[idx]

        # Apply transformations
        if self.transform:
            img = self.transform(img)
        else:
            # Fallback to basic tensor conversion if no transform is provided
            img = transforms.ToTensor()(img)

        # Return image and label (if available)
        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return img, label

        return img


def load_data(load_cached_data=True):
    """
    Loads data from disk or cache.
    Implements caching mechanism using .npy files in ./working/idea_28/

    Args:
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        tuple: ((train_imgs, train_labels), (val_imgs, val_labels), (test_imgs, test_ids))
    """
    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Define cache file paths
    cache_files = {
        "train_imgs": os.path.join(WORKING_DIR, "train_imgs.npy"),
        "train_labels": os.path.join(WORKING_DIR, "train_labels.npy"),
        "val_imgs": os.path.join(WORKING_DIR, "val_imgs.npy"),
        "val_labels": os.path.join(WORKING_DIR, "val_labels.npy"),
        "test_imgs": os.path.join(WORKING_DIR, "test_imgs.npy"),
        "test_ids": os.path.join(WORKING_DIR, "test_ids.npy"),
    }

    # Check if all cache files exist
    cache_exists = all(os.path.exists(p) for p in cache_files.values())

    # 1. Try loading from cache
    if load_cached_data and cache_exists:
        print("Loading data from cache...")
        train_imgs = np.load(cache_files["train_imgs"])
        train_labels = np.load(cache_files["train_labels"])
        val_imgs = np.load(cache_files["val_imgs"])
        val_labels = np.load(cache_files["val_labels"])
        test_imgs = np.load(cache_files["test_imgs"])
        test_ids = np.load(cache_files["test_ids"])
        return (train_imgs, train_labels), (val_imgs, val_labels), (test_imgs, test_ids)

    # 2. Process data from scratch
    print("Cache missing or reload requested. Processing data from scratch...")

    # Load Metadata CSVs
    train_meta_path = os.path.join(METADATA_DIR, "train_metadata.csv")
    val_meta_path = os.path.join(METADATA_DIR, "val_metadata.csv")
    test_meta_path = os.path.join(METADATA_DIR, "test_metadata.csv")

    def _process_subset(meta_path, has_labels=True):
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        df = pd.read_csv(meta_path)
        imgs = []
        ids = []
        labels = []

        for _, row in df.iterrows():
            # file_path is relative to INPUT_DIR (e.g., "train/xxx.jpg")
            full_path = os.path.join(INPUT_DIR, row["file_path"])

            # Read image using OpenCV
            img = cv2.imread(full_path)
            if img is None:
                continue

            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgs.append(img)
            ids.append(row["id"])

            if has_labels:
                labels.append(row["has_cactus"])

        # Convert to numpy arrays
        # Keep images as uint8 to save space until transform time
        imgs_np = np.array(imgs, dtype=np.uint8)
        ids_np = np.array(ids)

        if has_labels:
            labels_np = np.array(labels, dtype=np.float32)
            return imgs_np, ids_np, labels_np
        else:
            return imgs_np, ids_np, None

    # Process each split
    print("Processing Training Set...")
    train_imgs, _, train_labels = _process_subset(train_meta_path, has_labels=True)

    print("Processing Validation Set...")
    val_imgs, _, val_labels = _process_subset(val_meta_path, has_labels=True)

    print("Processing Test Set...")
    test_imgs, test_ids, _ = _process_subset(test_meta_path, has_labels=False)

    # 3. Save to cache
    print("Saving processed data to cache...")
    np.save(cache_files["train_imgs"], train_imgs)
    np.save(cache_files["train_labels"], train_labels)
    np.save(cache_files["val_imgs"], val_imgs)
    np.save(cache_files["val_labels"], val_labels)
    np.save(cache_files["test_imgs"], test_imgs)
    np.save(cache_files["test_ids"], test_ids)

    return (train_imgs, train_labels), (val_imgs, val_labels), (test_imgs, test_ids)
