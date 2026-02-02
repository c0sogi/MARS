import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import (
    INPUT_DIR,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    CACHE_DIR,
    IMG_SIZE,
    INPUT_CHANNELS,
    SPATIAL_OFFSETS,
    SEED,
    NUM_WORKERS,
    PIN_MEMORY,
)
from library.utils import read_dicom_image, calculate_modality_com, seed_everything

# Ensure reproducibility
seed_everything(SEED)


def get_transforms(phase="train"):
    """
    Returns the Albumentations transformation pipeline.

    Args:
        phase (str): 'train', 'val', or 'test'.

    Returns:
        A.Compose: The composition of transforms.
    """
    if phase == "train":
        return A.Compose(
            [
                # Spatially-Coupled Augmentations
                # Strictly exclude RandomScale and Shift (Translation) to preserve CoM anchoring
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0),
                A.ElasticTransform(
                    alpha=1,
                    sigma=50,
                    alpha_affine=0,  # Disable affine translation/scaling part of elastic transform
                    p=0.3,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                ),
                A.GridDistortion(
                    num_steps=5,
                    distort_limit=0.3,
                    p=0.3,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                ),
                ToTensorV2(),
            ]
        )
    else:
        # Validation/Test: Just convert to tensor
        # Normalization is already applied during pre-processing
        return A.Compose([ToTensorV2()])


def preprocess_image(img_path, target_size=IMG_SIZE):
    """
    Reads, resizes, and normalizes a single DICOM image.

    Args:
        img_path (str): Path to the DICOM file.
        target_size (tuple): Target resolution (H, W).

    Returns:
        np.ndarray: Preprocessed image (H, W) in float32, range [0, 1].
    """
    if img_path is None:
        # Return black image if path is invalid
        return np.zeros(target_size, dtype=np.float32)

    img = read_dicom_image(img_path)

    if img is None:
        return np.zeros(target_size, dtype=np.float32)

    # Resize
    if img.shape != target_size:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)

    # Convert to float32
    img = img.astype(np.float32)

    # Independent Min-Max Scaling
    min_val = img.min()
    max_val = img.max()

    if max_val > min_val:
        img = (img - min_val) / (max_val - min_val)
    else:
        img = np.zeros_like(img)

    return img


def load_expert_data(expert_name, split, metadata_path, load_cached_data=True):
    """
    Loads data for a specific expert and split. Implements caching logic.

    Args:
        expert_name (str): 'lower', 'center', or 'upper'.
        split (str): 'train', 'val', or 'test'.
        metadata_path (str): Path to the metadata CSV.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (images, labels, ids)
               images: np.ndarray (N, H, W, C)
               labels: np.ndarray (N,) or None
               ids: np.ndarray (N,)
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Define cache paths
    cache_prefix = f"{split}_{expert_name}"
    images_cache_path = os.path.join(CACHE_DIR, f"{cache_prefix}_images.npy")
    labels_cache_path = os.path.join(CACHE_DIR, f"{cache_prefix}_labels.npy")
    ids_cache_path = os.path.join(CACHE_DIR, f"{cache_prefix}_ids.npy")

    # 1. Try to load from cache
    if load_cached_data:
        if os.path.exists(images_cache_path) and os.path.exists(ids_cache_path):
            # Check labels existence only if not test
            if split == "test" or os.path.exists(labels_cache_path):
                print(f"Loading cached data for {expert_name} ({split})...")
                images = np.load(images_cache_path)
                ids = np.load(ids_cache_path)
                labels = np.load(labels_cache_path) if split != "test" else None
                return images, labels, ids

    # 2. Process from scratch
    print(f"Processing data for {expert_name} ({split})...")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)
    offset_ratio = SPATIAL_OFFSETS[expert_name]

    img_list = []
    label_list = []
    id_list = []

    # Iterate through subjects
    for _, row in df.iterrows():
        subject_id = row["BraTS21ID"]

        # Collect channels for this subject
        channels = []
        for mod in INPUT_CHANNELS:  # ['FLAIR', 'T1wCE', 'T2w']
            # Metadata paths are relative to INPUT_DIR
            # e.g., row['flair_path'] -> 'train/00000/FLAIR'
            mod_key = mod.lower()
            if mod == "T1wCE":
                mod_key = "t1wce"  # adjustment for column name matching

            rel_path = row[f"{mod_key}_path"]
            full_dir_path = os.path.join(INPUT_DIR, rel_path)

            # Independent Content-Based Anchoring
            # Calculate CoM and get slice path
            slice_path = calculate_modality_com(
                full_dir_path, offset_ratio=offset_ratio
            )

            # Preprocess (Read, Resize, Normalize)
            img_slice = preprocess_image(slice_path)
            channels.append(img_slice)

        # Stack channels: (H, W) list -> (H, W, 3)
        img_vol = np.stack(channels, axis=-1)

        img_list.append(img_vol)
        id_list.append(subject_id)

        if split != "test":
            label_list.append(row["MGMT_value"])

    # Convert to numpy arrays
    images = np.array(img_list, dtype=np.float32)
    ids = np.array(id_list, dtype=np.int64)

    if split != "test":
        labels = np.array(label_list, dtype=np.float32)
    else:
        labels = None

    # Save to cache
    np.save(images_cache_path, images)
    np.save(ids_cache_path, ids)
    if labels is not None:
        np.save(labels_cache_path, labels)

    print(f"Cached data saved to {CACHE_DIR}")
    return images, labels, ids


class SDCDataset(Dataset):
    """
    Spatially-Decomposed Consensus Dataset.
    """

    def __init__(self, images, labels=None, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Image is (H, W, 3) float32 in [0, 1]
        image = self.images[idx]

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Fallback if no transform provided (shouldn't happen with get_transforms)
            image = torch.from_numpy(image.transpose(2, 0, 1))

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return image, label
        else:
            return image


def get_expert_dataloader(
    expert_name, split, batch_size=32, shuffle=True, load_cached_data=True
):
    """
    Creates a DataLoader for a specific expert and split.

    Args:
        expert_name (str): 'lower', 'center', or 'upper'.
        split (str): 'train', 'val', or 'test'.
        batch_size (int): Batch size.
        shuffle (bool): Whether to shuffle data.
        load_cached_data (bool): Whether to use cached .npy files.

    Returns:
        DataLoader: The configured dataloader.
    """
    # Determine metadata path
    if split == "train":
        meta_path = TRAIN_METADATA_PATH
    elif split == "val":
        meta_path = VAL_METADATA_PATH
    elif split == "test":
        meta_path = TEST_METADATA_PATH
    else:
        raise ValueError(f"Invalid split: {split}")

    # Load data (cached or processed)
    images, labels, ids = load_expert_data(
        expert_name, split, meta_path, load_cached_data
    )

    # Get transforms
    transform = get_transforms(phase=split)

    # Create Dataset
    dataset = SDCDataset(images, labels, transform=transform)

    # Create DataLoader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    # Attach IDs to the loader for tracking/submission
    loader.dataset_ids = ids

    return loader
