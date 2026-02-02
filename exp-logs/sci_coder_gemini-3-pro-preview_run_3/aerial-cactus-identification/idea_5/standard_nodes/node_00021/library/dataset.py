import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from library.config import Config


def get_transforms(phase: str):
    """
    Returns the data augmentation and normalization pipeline.

    Args:
        phase (str): 'train', 'val', or 'test'.

    Returns:
        torchvision.transforms.Compose: The transform pipeline.
    """
    if phase == "train":
        return transforms.Compose(
            [
                # Geometric (Lossless Only)
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                # Photometric
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                # Normalization to [0, 1]
                transforms.ToTensor(),
            ]
        )
    else:
        # Validation and Test: Only normalize
        return transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )


def load_data(metadata_path, input_dir, cache_prefix, load_cached_data=True):
    """
    Loads data from metadata CSV and images, with caching to .npy files.

    Args:
        metadata_path (str): Path to the metadata CSV file.
        input_dir (str): Root directory containing image files.
        cache_prefix (str): Prefix for the cached filenames (e.g., 'train', 'val').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (images_np, labels_np, ids_np)
    """
    # Ensure working directory exists
    os.makedirs(Config.WORK_DIR, exist_ok=True)

    # Define cache paths
    cache_imgs_path = os.path.join(Config.WORK_DIR, f"{cache_prefix}_imgs.npy")
    cache_lbls_path = os.path.join(Config.WORK_DIR, f"{cache_prefix}_lbls.npy")
    cache_ids_path = os.path.join(Config.WORK_DIR, f"{cache_prefix}_ids.npy")

    # Attempt to load from cache
    if load_cached_data:
        if (
            os.path.exists(cache_imgs_path)
            and os.path.exists(cache_lbls_path)
            and os.path.exists(cache_ids_path)
        ):

            # Load from npy
            imgs = np.load(cache_imgs_path)
            lbls = np.load(cache_lbls_path)
            ids = np.load(cache_ids_path)

            # Handle Debug mode on cached data
            if Config.DEBUG:
                imgs = imgs[: Config.DEBUG_SAMPLE_SIZE]
                lbls = lbls[: Config.DEBUG_SAMPLE_SIZE]
                ids = ids[: Config.DEBUG_SAMPLE_SIZE]

            return imgs, lbls, ids

    # If cache miss or forced reload, process from scratch
    df = pd.read_csv(metadata_path)

    if Config.DEBUG:
        df = df.head(Config.DEBUG_SAMPLE_SIZE)

    img_list = []
    lbl_list = []
    id_list = []

    for _, row in df.iterrows():
        rel_path = row["file_path"]
        full_path = os.path.join(input_dir, rel_path)
        img_id = row["id"]
        label = row["has_cactus"]

        # Read image
        img = cv2.imread(full_path)
        if img is None:
            # Skip missing images if any (though metadata check passed)
            continue

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_list.append(img)
        lbl_list.append(label)
        id_list.append(img_id)

    # Convert to numpy arrays
    imgs_np = np.array(img_list, dtype=np.uint8)
    lbls_np = np.array(lbl_list, dtype=np.float32)  # Float for BCE loss
    ids_np = np.array(id_list)

    # Save to cache
    np.save(cache_imgs_path, imgs_np)
    np.save(cache_lbls_path, lbls_np)
    np.save(cache_ids_path, ids_np)

    return imgs_np, lbls_np, ids_np


class CactusDataset(Dataset):
    """
    PyTorch Dataset for Cactus Identification.
    """

    def __init__(self, images, labels, ids, transform=None):
        """
        Args:
            images (np.ndarray): Array of images (N, H, W, C).
            labels (np.ndarray): Array of labels (N,).
            ids (np.ndarray): Array of image IDs (N,).
            transform (callable, optional): Transform to apply to images.
        """
        self.images = images
        self.labels = labels
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Get image and convert to PIL for torchvision transforms
        img_arr = self.images[idx]
        img = Image.fromarray(img_arr)

        # Get label and id
        label = self.labels[idx]
        img_id = self.ids[idx]

        # Apply transforms
        if self.transform:
            img = self.transform(img)

        # Return tensors
        # Label needs to be a tensor for loss calculation
        target = torch.tensor(label, dtype=torch.float32)

        return img, target, img_id
