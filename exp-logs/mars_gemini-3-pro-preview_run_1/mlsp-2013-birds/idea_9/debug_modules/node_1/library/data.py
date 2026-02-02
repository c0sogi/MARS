import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from library.config import (
    WORKING_DIR,
    SPECTROGRAM_DIR,
    IMG_HEIGHT,
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV,
    SEED,
    NUM_WORKERS,
    NUM_CLASSES,
)
from library.utils import set_seed


def mixup_data(x, y, alpha=1.0, device="cuda"):
    """
    Applies Mixup augmentation to the input batch.

    Args:
        x (torch.Tensor): Input batch of images.
        y (torch.Tensor): Input batch of labels.
        alpha (float): Mixup interpolation coefficient parameter.
        device (str): Device to perform computations on.

    Returns:
        mixed_x (torch.Tensor): Mixed images.
        y_a (torch.Tensor): Labels for the first permutation.
        y_b (torch.Tensor): Labels for the second permutation.
        lam (float): Interpolation coefficient.
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


class BirdDataset(Dataset):
    """
    PyTorch Dataset for Bird Species Classification.
    Handles 3-channel replicated spectrograms and multi-label targets.
    """

    def __init__(self, images, labels, ids, transform=None):
        """
        Args:
            images (torch.Tensor or np.ndarray): Image data of shape (N, 3, H, W).
                                                 Values should be normalized (0-1) or standardized.
            labels (torch.Tensor or np.ndarray): Label data of shape (N, num_classes).
            ids (np.ndarray): Recording IDs corresponding to the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.images = images
        self.labels = labels
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Ensure data is tensor
        image = self.images[idx]
        label = self.labels[idx]
        rec_id = self.ids[idx]

        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).float()
        if not isinstance(label, torch.Tensor):
            label = torch.from_numpy(label).float()

        if self.transform:
            image = self.transform(image)

        return image, label, rec_id


def load_metadata(split):
    """
    Loads the metadata CSV for a given split.

    Args:
        split (str): One of 'train', 'val', 'test'.

    Returns:
        pd.DataFrame: Loaded metadata.
    """
    if split == "train":
        return pd.read_csv(TRAIN_CSV)
    elif split == "val":
        return pd.read_csv(VAL_CSV)
    elif split == "test":
        return pd.read_csv(TEST_CSV)
    else:
        raise ValueError(f"Unknown split: {split}")


def process_and_cache_data(df, split_name, width, height, load_cached_data=True):
    """
    Loads images from disk, resizes them, replicates channels, and caches the result.
    If cached data exists and load_cached_data is True, loads from cache instead.

    Args:
        df (pd.DataFrame): Metadata dataframe containing 'rec_id', 'file_path', and label columns.
        split_name (str): Unique identifier for the split (e.g., 'train', 'val', 'test').
                          Used for cache filename generation.
        width (int): Target width for resizing.
        height (int): Target height for resizing.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (images, labels, ids)
            images (torch.Tensor): Shape (N, 3, H, W), float32, normalized (ImageNet stats).
            labels (torch.Tensor): Shape (N, num_classes), float32.
            ids (np.ndarray): Shape (N,), int64.
    """
    # Define cache paths
    cache_dir = WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    img_cache_path = os.path.join(
        cache_dir, f"images_{split_name}_{width}x{height}.npy"
    )
    lbl_cache_path = os.path.join(cache_dir, f"labels_{split_name}.npy")
    id_cache_path = os.path.join(cache_dir, f"ids_{split_name}.npy")

    # Attempt to load from cache
    if load_cached_data:
        if (
            os.path.exists(img_cache_path)
            and os.path.exists(lbl_cache_path)
            and os.path.exists(id_cache_path)
        ):
            try:
                # print(f"Loading {split_name} data from cache: {img_cache_path}")
                images = np.load(img_cache_path)
                labels = np.load(lbl_cache_path)
                ids = np.load(id_cache_path)
                return torch.from_numpy(images), torch.from_numpy(labels), ids
            except Exception as e:
                # print(f"Failed to load cache for {split_name}: {e}. Recomputing...")
                pass

    # Process from scratch
    # print(f"Processing {split_name} data from scratch...")

    # Pre-allocate arrays
    num_samples = len(df)

    # Identify label columns
    # Explicitly select columns based on config to avoid artifacts (Cite debug_lesson_1)
    label_cols = [f"species_{i}" for i in range(NUM_CLASSES)]
    num_classes = len(label_cols)

    # Initialize containers
    # We store as float32 for direct usage in PyTorch
    images_arr = np.zeros((num_samples, 3, height, width), dtype=np.float32)
    labels_arr = np.zeros((num_samples, num_classes), dtype=np.float32)
    ids_arr = np.zeros(num_samples, dtype=np.int64)

    # ImageNet Normalization Constants
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

    for idx, row in df.iterrows():
        # Get Rec ID
        rec_id = row["rec_id"]
        ids_arr[idx] = rec_id

        # Get Labels
        if num_classes > 0:
            labels_arr[idx] = row[label_cols].values.astype(np.float32)

        # Load Image
        # file_path in csv is relative to input dir
        rel_path = row["file_path"]
        # The spectrograms are in SPECTROGRAM_DIR with .bmp extension
        # We need to map wav filename to bmp filename
        wav_basename = os.path.basename(rel_path)
        bmp_basename = os.path.splitext(wav_basename)[0] + ".bmp"
        img_path = os.path.join(SPECTROGRAM_DIR, bmp_basename)

        if os.path.exists(img_path):
            # Read as grayscale
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                # Fallback: create black image
                img = np.zeros((height, width), dtype=np.uint8)
            else:
                # Resize
                img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
        else:
            # Fallback
            img = np.zeros((height, width), dtype=np.uint8)

        # Normalize to 0-1
        img = img.astype(np.float32) / 255.0

        # Replicate channels (H, W) -> (3, H, W)
        # First expand dims to (1, H, W)
        img_ch = img[np.newaxis, :, :]
        img_rgb = np.repeat(img_ch, 3, axis=0)

        # Apply ImageNet Normalization
        img_rgb = (img_rgb - mean) / std

        images_arr[idx] = img_rgb

    # Save to cache
    np.save(img_cache_path, images_arr)
    np.save(lbl_cache_path, labels_arr)
    np.save(id_cache_path, ids_arr)

    return torch.from_numpy(images_arr), torch.from_numpy(labels_arr), ids_arr


def get_loader(
    images, labels, ids, batch_size, shuffle=True, drop_last=False, transform=None
):
    """
    Creates a DataLoader from pre-processed tensors.

    Args:
        images (torch.Tensor): Image tensor.
        labels (torch.Tensor): Label tensor.
        ids (np.ndarray): ID array.
        batch_size (int): Batch size.
        shuffle (bool): Whether to shuffle.
        drop_last (bool): Whether to drop the last incomplete batch.
        transform (callable): Optional transform.

    Returns:
        DataLoader: PyTorch DataLoader.
    """
    dataset = BirdDataset(images, labels, ids, transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=drop_last,
    )

    return loader
