import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from library.config import Config


def get_transforms(phase: str):
    """
    Returns the image transformation pipeline based on the phase.

    Args:
        phase (str): 'train', 'val', or 'test'.

    Returns:
        torchvision.transforms.Compose: The transform pipeline.
    """
    if phase == "train":
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                # ToTensor converts [0, 255] uint8 to [0.0, 1.0] float
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )


class CactusDataset(Dataset):
    """
    Custom Dataset for Cactus Classification.
    Loads images and labels, supporting caching for efficiency.
    """

    def __init__(
        self,
        metadata_path,
        phase,
        transform=None,
        max_samples=None,
        load_cached_data=True,
    ):
        """
        Args:
            metadata_path (str): Path to the metadata CSV file.
            phase (str): 'train', 'val', or 'test'.
            transform (callable, optional): Optional transform to be applied on a sample.
            max_samples (int, optional): Limit the number of samples for debugging.
            load_cached_data (bool): Whether to try loading from cache.
        """
        self.metadata_path = metadata_path
        self.phase = phase
        self.transform = transform
        self.max_samples = max_samples

        # Ensure working directory exists for caching
        self.cache_dir = Config.WORKING_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

        # Define cache file paths
        self.images_path = os.path.join(self.cache_dir, f"{phase}_images.npy")
        self.labels_path = os.path.join(self.cache_dir, f"{phase}_labels.npy")
        self.ids_path = os.path.join(self.cache_dir, f"{phase}_ids.npy")

        self.images = None
        self.labels = None
        self.ids = None

        # Load data (either from cache or source)
        self._load_data(load_cached_data)

        # Apply max_samples limit if requested (for debugging)
        if self.max_samples is not None:
            self.images = self.images[: self.max_samples]
            self.labels = self.labels[: self.max_samples]
            self.ids = self.ids[: self.max_samples]

    def _load_data(self, load_cached_data):
        """
        Internal method to load data from cache or compute from scratch.
        """
        cache_exists = (
            os.path.exists(self.images_path)
            and os.path.exists(self.labels_path)
            and os.path.exists(self.ids_path)
        )

        if load_cached_data and cache_exists:
            try:
                # Load from npy files
                self.images = np.load(self.images_path)
                self.labels = np.load(self.labels_path)
                # Load IDs and decode from bytes to string
                self.ids = np.load(self.ids_path).astype(str)
                return
            except Exception:
                # If loading fails, fall back to processing from scratch
                pass

        # Process from scratch if cache missing or load failed
        self._process_and_cache()

    def _process_and_cache(self):
        """
        Reads images from disk based on metadata, processes them, and saves to cache.
        """
        df = pd.read_csv(self.metadata_path)

        img_list = []
        label_list = []
        id_list = []

        for _, row in df.iterrows():
            rel_path = row["file_path"]
            full_path = os.path.join(Config.INPUT_DIR, rel_path)

            # Load image using OpenCV
            img = cv2.imread(full_path)
            if img is None:
                continue

            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img_list.append(img)
            label_list.append(row["has_cactus"])
            id_list.append(row["id"])

        # Convert to numpy arrays
        self.images = np.array(img_list, dtype=np.uint8)
        self.labels = np.array(label_list, dtype=np.float32)
        # Store IDs as bytes (S type) to avoid pickle usage in np.save
        self.ids = np.array(id_list, dtype="S")

        # Save to cache
        np.save(self.images_path, self.images)
        np.save(self.labels_path, self.labels)
        np.save(self.ids_path, self.ids)

        # Convert IDs back to string for internal usage
        self.ids = self.ids.astype(str)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        img_id = self.ids[idx]

        # Apply transformations
        if self.transform:
            img = self.transform(img)

        return img, label, img_id
