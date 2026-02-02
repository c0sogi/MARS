import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config

# Set seeds for reproducibility where applicable
np.random.seed(Config.SEED)
torch.manual_seed(Config.SEED)


def get_transforms(split="train"):
    """
    Returns the Albumentations transform pipeline for the specified split.

    Args:
        split (str): One of 'train', 'val', 'test'.

    Returns:
        A.Compose: The transform pipeline.
    """
    # Statistics derived from Data Analysis (converted to 0-1 scale)
    # Mean: R=128.37, G=115.25, B=119.40 -> /255
    mean = [0.50339, 0.45197, 0.46825]
    # Std: R=38.60, G=35.68, B=39.15 -> /255
    std = [0.15138, 0.13993, 0.15355]

    transforms_list = []

    if split == "train":
        # Geometric augmentations for training
        transforms_list.extend(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
            ]
        )

    # Normalization and Tensor conversion for all splits
    transforms_list.extend(
        [A.Normalize(mean=mean, std=std, max_pixel_value=1.0), ToTensorV2()]
    )

    return A.Compose(transforms_list)


class CactusDataset(Dataset):
    """
    A PyTorch Dataset wrapper for in-memory image tensors.
    """

    def __init__(self, images, labels=None, ids=None, transform=None):
        """
        Args:
            images (np.ndarray): Array of shape (N, H, W, C) with float32 values in [0, 1].
            labels (np.ndarray, optional): Array of shape (N,) with targets.
            ids (np.ndarray, optional): Array of shape (N,) with image IDs.
            transform (A.Compose, optional): Albumentations transform pipeline.
        """
        self.images = images
        self.labels = labels
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Retrieve image
        image = self.images[idx]

        # Apply transforms
        if self.transform:
            # Albumentations expects image in (H, W, C)
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Fallback to simple tensor conversion if no transform provided
            image = torch.from_numpy(image.transpose(2, 0, 1))

        if self.labels is not None:
            label = self.labels[idx]
            # Return float tensor for label to support Mixup/BCEWithLogits
            return image, torch.tensor(label, dtype=torch.float32)
        elif self.ids is not None:
            return image, self.ids[idx]
        else:
            return image


def load_data_to_memory(
    metadata_path,
    cache_imgs_path,
    cache_labels_path=None,
    cache_ids_path=None,
    load_cached_data=True,
    is_test=False,
):
    """
    Loads images and labels/ids into memory, using disk caching to speed up subsequent runs.

    Args:
        metadata_path (str): Path to the metadata CSV file.
        cache_imgs_path (str): Path to save/load cached images .npy file.
        cache_labels_path (str, optional): Path to save/load cached labels .npy file.
        cache_ids_path (str, optional): Path to save/load cached IDs .npy file (for test set).
        load_cached_data (bool): Whether to attempt loading from cache.
        is_test (bool): Whether loading test data (expects IDs instead of labels).

    Returns:
        tuple: (images, targets) where targets is labels (if not is_test) or ids (if is_test).
    """
    # Ensure working directory exists
    os.makedirs(os.path.dirname(cache_imgs_path), exist_ok=True)

    # 1. Try to load from cache
    if load_cached_data:
        imgs_exist = os.path.exists(cache_imgs_path)
        targets_exist = False

        if is_test and cache_ids_path:
            targets_exist = os.path.exists(cache_ids_path)
        elif not is_test and cache_labels_path:
            targets_exist = os.path.exists(cache_labels_path)

        if imgs_exist and targets_exist:
            print(f"Loading cached data from {os.path.dirname(cache_imgs_path)}...")
            images = np.load(cache_imgs_path)
            if is_test:
                ids = np.load(cache_ids_path, allow_pickle=True)
                return images, ids
            else:
                labels = np.load(cache_labels_path)
                return images, labels

    # 2. Process from scratch
    print(f"Processing data from {metadata_path}...")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)

    # Pre-allocate arrays
    num_samples = len(df)
    # Images are 32x32 RGB
    images = np.zeros(
        (num_samples, Config.IMAGE_SIZE, Config.IMAGE_SIZE, Config.IN_CHANNELS),
        dtype=np.float32,
    )

    if not is_test:
        labels = np.zeros(num_samples, dtype=np.float32)
    else:
        ids = []

    # Iterate and load
    for idx, row in df.iterrows():
        # Construct full path: INPUT_DIR + relative path from metadata
        rel_path = row["file_path"]
        full_path = os.path.join(Config.INPUT_DIR, rel_path)

        # Read image
        img = cv2.imread(full_path)
        if img is None:
            # Fallback for robustness, though metadata validation ensures files exist
            print(f"Warning: Could not read image {full_path}. Using black image.")
            img = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8)
        else:
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0

        images[idx] = img

        if not is_test:
            labels[idx] = row["has_cactus"]
        else:
            ids.append(row["id"])

    # 3. Save to cache
    print(f"Saving processed data to cache...")
    np.save(cache_imgs_path, images)

    if is_test:
        # Save IDs
        ids_arr = np.array(ids)
        np.save(cache_ids_path, ids_arr)
        return images, ids_arr
    else:
        # Save Labels
        np.save(cache_labels_path, labels)
        return images, labels
