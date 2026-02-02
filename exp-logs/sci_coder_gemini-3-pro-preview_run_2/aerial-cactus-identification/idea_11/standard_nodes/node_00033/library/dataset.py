import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from library.config import Config


def get_transforms(split="train"):
    """
    Returns the transformations for the given split.

    Args:
        split (str): One of 'train', 'val', 'test'.

    Returns:
        torchvision.transforms.Compose: Composed transformations.
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
    """
    Custom Dataset for Cactus Classification.
    Loads images from disk, caches them as numpy arrays for speed, and applies transforms.
    """

    def __init__(
        self, metadata_path, split, transform=None, load_cached_data=True, debug=False
    ):
        """
        Args:
            metadata_path (str): Path to the metadata CSV file.
            split (str): 'train', 'val', or 'test'. Used for cache naming.
            transform (callable, optional): Optional transform to be applied on a sample.
            load_cached_data (bool): Whether to try loading from cache.
            debug (bool): If True, limits the dataset size for debugging.
        """
        self.metadata_path = metadata_path
        self.split = split
        self.transform = transform
        self.debug = debug

        # Define cache paths
        # Using Config.WORK_DIR ensures we write to the allowed working directory
        cache_dir = Config.WORK_DIR
        os.makedirs(cache_dir, exist_ok=True)

        suffix = "_debug" if debug else ""
        self.images_cache = os.path.join(cache_dir, f"{split}_images{suffix}.npy")
        self.labels_cache = os.path.join(cache_dir, f"{split}_labels{suffix}.npy")
        self.ids_cache = os.path.join(cache_dir, f"{split}_ids{suffix}.npy")

        # Load data
        self.images, self.labels, self.ids = self._load_data(load_cached_data)

    def _load_data(self, load_cached_data):
        """
        Internal method to load data from cache or raw files.
        """
        # 1. Try loading from cache
        if (
            load_cached_data
            and os.path.exists(self.images_cache)
            and os.path.exists(self.labels_cache)
            and os.path.exists(self.ids_cache)
        ):
            try:
                images = np.load(self.images_cache)
                labels = np.load(self.labels_cache)
                ids = np.load(self.ids_cache)
                return images, labels, ids
            except Exception:
                # If load fails, fall back to processing
                pass

        # 2. Process from scratch
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        df = pd.read_csv(self.metadata_path)

        if self.debug:
            df = df.head(Config.DEBUG_SUBSET_SIZE)

        images_list = []
        labels_list = []
        ids_list = []

        # Pre-calculate full paths to avoid doing it in the loop repeatedly
        # Metadata file_path is relative to input dir
        for _, row in df.iterrows():
            rel_path = row["file_path"]
            full_path = os.path.join(Config.INPUT_DIR, rel_path)

            # Read image using OpenCV
            img = cv2.imread(full_path)

            if img is None:
                # Skip if image cannot be read
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
        try:
            np.save(self.images_cache, images)
            np.save(self.labels_cache, labels)
            np.save(self.ids_cache, ids)
        except Exception as e:
            print(f"Warning: Failed to save cache: {e}")

        return images, labels, ids

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Get image and label
        img = self.images[idx]
        label = self.labels[idx]
        img_id = self.ids[idx]

        # Apply transforms
        # img is HWC uint8, transforms.ToTensor() converts to CHW float32 [0, 1]
        if self.transform:
            img = self.transform(img)

        # Ensure label is a tensor
        label = torch.tensor(label, dtype=torch.float32)

        return img, label, img_id
