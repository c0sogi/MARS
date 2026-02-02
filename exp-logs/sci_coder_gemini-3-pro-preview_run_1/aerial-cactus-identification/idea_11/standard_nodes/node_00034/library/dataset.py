import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from library.config import Config


def mixup_data(x, y, alpha=1.0, device="cpu"):
    """
    Returns mixed inputs, pairs of targets, and lambda.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Calculates the mixup loss.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def get_transforms(mode="train", image_size=32):
    """
    Returns the Albumentations transforms for the given mode.
    """
    if mode == "train":
        return A.Compose([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), ToTensorV2()])
    else:
        return A.Compose([ToTensorV2()])


def load_data_to_memory(
    metadata_path,
    cache_img_path,
    cache_filesize_path,
    cache_label_path=None,
    cache_id_path=None,
    load_cached_data=True,
):
    """
    Loads data from metadata/images, caches it to disk as npy, and returns numpy arrays.

    Args:
        metadata_path (str): Path to the metadata CSV.
        cache_img_path (str): Path to save/load image cache.
        cache_filesize_path (str): Path to save/load filesize cache.
        cache_label_path (str, optional): Path to save/load label cache.
        cache_id_path (str, optional): Path to save/load ID cache.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (imgs, labels, filesizes, ids)
    """
    # Check if we can load from cache
    cache_exists = os.path.exists(cache_img_path) and os.path.exists(
        cache_filesize_path
    )
    if cache_label_path:
        cache_exists = cache_exists and os.path.exists(cache_label_path)
    if cache_id_path:
        cache_exists = cache_exists and os.path.exists(cache_id_path)

    if load_cached_data and cache_exists:
        imgs = np.load(cache_img_path)
        filesizes = np.load(cache_filesize_path)

        labels = (
            np.load(cache_label_path)
            if cache_label_path
            else np.zeros(len(imgs), dtype=np.float32)
        )
        ids = np.load(cache_id_path, allow_pickle=True) if cache_id_path else None

        return imgs, labels, filesizes, ids

    # Process from scratch
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)
    n_samples = len(df)

    # Pre-allocate arrays
    # Images: NHWC format, float32, 0-1 range
    imgs = np.zeros(
        (n_samples, Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.float32
    )
    filesizes = np.zeros(n_samples, dtype=np.float32)
    labels = np.zeros(n_samples, dtype=np.float32)
    ids = df["id"].values if cache_id_path else None

    for i, row in df.iterrows():
        rel_path = row["file_path"]
        full_path = os.path.join(Config.INPUT_DIR, rel_path)

        # Load and process image
        img = cv2.imread(full_path)
        if img is None:
            # Should not happen based on metadata validation, but handle gracefully
            img = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Normalize to 0-1 float32
        imgs[i] = img.astype(np.float32) / 255.0

        # Extract File Size
        if os.path.exists(full_path):
            filesizes[i] = os.path.getsize(full_path)
        else:
            filesizes[i] = 0.0

        # Extract Label (if available in metadata, else 0.5 placeholder)
        if "has_cactus" in row:
            labels[i] = row["has_cactus"]

    # Save to cache
    os.makedirs(os.path.dirname(cache_img_path), exist_ok=True)
    np.save(cache_img_path, imgs)
    np.save(cache_filesize_path, filesizes)

    if cache_label_path:
        np.save(cache_label_path, labels)

    if cache_id_path:
        np.save(cache_id_path, ids)

    return imgs, labels, filesizes, ids


class CactusDataset(Dataset):
    def __init__(
        self,
        images,
        filesizes,
        labels=None,
        transform=None,
        filesize_mean=0.0,
        filesize_std=1.0,
    ):
        """
        Args:
            images (np.ndarray): Array of images (N, H, W, C).
            filesizes (np.ndarray): Array of raw file sizes.
            labels (np.ndarray, optional): Array of labels.
            transform (A.Compose, optional): Albumentations transforms.
            filesize_mean (float): Mean of file sizes (from training set) for normalization.
            filesize_std (float): Std of file sizes (from training set) for normalization.
        """
        self.images = images
        self.filesizes = filesizes
        self.labels = labels
        self.transform = transform
        self.filesize_mean = filesize_mean
        self.filesize_std = filesize_std if filesize_std != 0 else 1.0

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Retrieve image (H, W, C)
        image = self.images[idx]

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]  # Returns Tensor (C, H, W)
        else:
            # Fallback conversion if no transform
            image = torch.from_numpy(image.transpose(2, 0, 1))

        # Normalize Metadata (Z-score)
        raw_size = self.filesizes[idx]
        norm_size = (raw_size - self.filesize_mean) / self.filesize_std
        # Ensure it's a float tensor
        norm_size = torch.tensor(norm_size, dtype=torch.float32)

        # Retrieve Label
        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
        else:
            label = torch.tensor(0.0, dtype=torch.float32)

        # Return nested tuple: ((visual_input, meta_input), target)
        return (image, norm_size), label
