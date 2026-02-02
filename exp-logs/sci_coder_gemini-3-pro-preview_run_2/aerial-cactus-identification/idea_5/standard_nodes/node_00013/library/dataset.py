import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from library.config import Config


def get_train_transforms():
    """
    Returns transforms for training:
    - ToTensor (converts HWC uint8 [0, 255] to CHW float [0.0, 1.0])
    - Random Horizontal Flip
    - Random Vertical Flip
    """
    return T.Compose(
        [
            T.ToTensor(),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
        ]
    )


def get_valid_transforms():
    """
    Returns transforms for validation/testing:
    - ToTensor (converts HWC uint8 [0, 255] to CHW float [0.0, 1.0])
    """
    return T.Compose(
        [
            T.ToTensor(),
        ]
    )


# Alias for test transforms
get_test_transforms = get_valid_transforms


class CactusDataset(Dataset):
    def __init__(
        self, metadata_path, transform=None, mode="train", load_cached_data=True
    ):
        """
        Args:
            metadata_path (str): Path to the metadata CSV file.
            transform (callable, optional): Optional transform to be applied on a sample.
            mode (str): 'train', 'val', or 'test'. Used for cache naming.
            load_cached_data (bool): Whether to try loading from cache.
        """
        self.metadata_path = metadata_path
        self.transform = transform
        self.mode = mode

        # Define cache paths in the working directory
        # Using .npy for fast loading of processed arrays
        self.cache_dir = Config.WORKING_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

        self.images_cache_path = os.path.join(self.cache_dir, f"{mode}_images.npy")
        self.labels_cache_path = os.path.join(self.cache_dir, f"{mode}_labels.npy")
        self.ids_cache_path = os.path.join(self.cache_dir, f"{mode}_ids.npy")

        # Load the data (either from cache or from scratch)
        self._load_data(load_cached_data)

        # Apply Debugging limit if configured
        if Config.DEBUG:
            limit = min(len(self.images), Config.DEBUG_SAMPLES)
            self.images = self.images[:limit]
            self.labels = self.labels[:limit]
            self.ids = self.ids[:limit]

    def _load_data(self, load_cached_data):
        """
        Handles the logic of loading data from cache or processing from raw files.
        """
        # 1. Try to load from cache
        if load_cached_data:
            if (
                os.path.exists(self.images_cache_path)
                and os.path.exists(self.labels_cache_path)
                and os.path.exists(self.ids_cache_path)
            ):
                try:
                    self.images = np.load(self.images_cache_path)
                    self.labels = np.load(self.labels_cache_path)
                    self.ids = np.load(self.ids_cache_path)
                    return
                except Exception:
                    # If loading fails, fall through to compute from scratch
                    pass

        # 2. Compute from scratch
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        df = pd.read_csv(self.metadata_path)

        img_list = []
        label_list = []
        id_list = []

        # Iterate through metadata and load images
        for _, row in df.iterrows():
            # Construct full path: input_dir + relative_path_from_metadata
            full_path = os.path.join(Config.INPUT_DIR, row["file_path"])

            # Load image using OpenCV
            img = cv2.imread(full_path)
            if img is None:
                # Skip missing images (though metadata validation ensures they exist)
                continue

            # Convert BGR (OpenCV default) to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img_list.append(img)
            label_list.append(row["has_cactus"])
            id_list.append(row["id"])

        # Convert to numpy arrays
        # Images: uint8 (0-255), shape (N, 32, 32, 3)
        self.images = np.array(img_list, dtype=np.uint8)
        # Labels: float32 for BCEWithLogitsLoss
        self.labels = np.array(label_list, dtype=np.float32)
        # IDs: string array
        self.ids = np.array(id_list)

        # 3. Save to cache for future runs
        np.save(self.images_cache_path, self.images)
        np.save(self.labels_cache_path, self.labels)
        np.save(self.ids_cache_path, self.ids)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Retrieve image and label
        img = self.images[idx]
        label = self.labels[idx]

        # Apply transforms (e.g., ToTensor, Augmentations)
        if self.transform:
            img = self.transform(img)

        # Return tensor image and tensor label
        return img, torch.tensor(label, dtype=torch.float32)
