import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config


def load_and_cache_data(mode, config, load_cached_data=True):
    """
    Loads data from disk or cache.

    Args:
        mode (str): 'train', 'val', or 'test'.
        config (class): Configuration class.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        dict: Dictionary containing 'ids', 'images', 'depths', and optionally 'masks'.
    """
    cache_dir = config.CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # Define cache filenames
    cache_files = {
        "ids": os.path.join(cache_dir, f"{mode}_ids.npy"),
        "images": os.path.join(cache_dir, f"{mode}_images.npy"),
        "depths": os.path.join(cache_dir, f"{mode}_depths.npy"),
        "masks": os.path.join(cache_dir, f"{mode}_masks.npy"),
    }

    # Check if cache exists
    cache_exists = all(
        os.path.exists(cache_files[k]) for k in ["ids", "images", "depths"]
    )
    if mode != "test":
        cache_exists = cache_exists and os.path.exists(cache_files["masks"])

    if load_cached_data and cache_exists:
        # Load from cache
        data = {}
        data["ids"] = np.load(cache_files["ids"], allow_pickle=True)
        data["images"] = np.load(cache_files["images"])
        data["depths"] = np.load(cache_files["depths"])
        if mode != "test":
            data["masks"] = np.load(cache_files["masks"])
        return data

    # If not cached or reload forced, process from scratch
    if mode == "train":
        csv_path = config.TRAIN_CSV
    elif mode == "val":
        csv_path = config.VAL_CSV
    elif mode == "test":
        csv_path = config.TEST_CSV
    else:
        raise ValueError(f"Unknown mode: {mode}")

    df = pd.read_csv(csv_path)

    # Debug mode: subset data
    if config.DEBUG:
        df = df.head(config.DEBUG_SAMPLE_SIZE)

    ids = df["id"].values
    depths = df["z"].values
    image_paths = df["image_path"].values

    # Pre-allocate arrays
    # Images are 101x101 grayscale (uint8)
    n_samples = len(ids)
    images = np.zeros(
        (n_samples, config.ORIG_IMG_SIZE, config.ORIG_IMG_SIZE), dtype=np.uint8
    )

    masks = None
    if mode != "test":
        masks = np.zeros(
            (n_samples, config.ORIG_IMG_SIZE, config.ORIG_IMG_SIZE), dtype=np.uint8
        )
        mask_paths = df["mask_path"].values

    # Load images
    for i in range(n_samples):
        # Load Image
        img_path = os.path.join(config.INPUT_DIR, image_paths[i])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Fallback (should not happen given validation)
            img = np.zeros((config.ORIG_IMG_SIZE, config.ORIG_IMG_SIZE), dtype=np.uint8)
        images[i] = img

        # Load Mask
        if mode != "test":
            msk_path = os.path.join(config.INPUT_DIR, mask_paths[i])
            msk = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
            if msk is None:
                msk = np.zeros(
                    (config.ORIG_IMG_SIZE, config.ORIG_IMG_SIZE), dtype=np.uint8
                )
            # Ensure binary
            msk = (msk > 127).astype(np.uint8)
            masks[i] = msk

    # Save to cache
    np.save(cache_files["ids"], ids)
    np.save(cache_files["images"], images)
    np.save(cache_files["depths"], depths)
    if mode != "test":
        np.save(cache_files["masks"], masks)

    data = {"ids": ids, "images": images, "depths": depths}
    if mode != "test":
        data["masks"] = masks

    return data


def get_transforms(mode, config):
    """
    Returns the Albumentations transform pipeline.
    """
    transforms = []

    # Pad 101x101 -> 128x128 using Reflection Padding
    # This is crucial for U-Net architecture which requires dimensions divisible by 32
    transforms.append(
        A.PadIfNeeded(
            min_height=config.IMG_SIZE,
            min_width=config.IMG_SIZE,
            border_mode=cv2.BORDER_REFLECT_101,
            always_apply=True,
        )
    )

    # Augmentations for training
    if mode == "train":
        if config.TTA_FLIP:
            transforms.append(A.HorizontalFlip(p=0.5))

    return A.Compose(transforms)


class SaltDataset(Dataset):
    """
    Dataset class for Salt Segmentation.
    """

    def __init__(self, mode, config, data, transform=None):
        self.mode = mode
        self.config = config
        self.transform = transform

        self.ids = data["ids"]
        self.images = data["images"]
        self.depths = data["depths"]
        self.masks = data.get("masks", None)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # Retrieve data from memory
        image = self.images[idx]  # uint8, (101, 101)
        depth = self.depths[idx]

        mask = None
        if self.masks is not None:
            mask = self.masks[idx]  # uint8, (101, 101)

        # Apply transforms (Padding, Flipping)
        # Albumentations works with numpy arrays (H, W) or (H, W, C)
        data_dict = {"image": image}
        if mask is not None:
            data_dict["mask"] = mask

        if self.transform:
            augmented = self.transform(**data_dict)
            image = augmented["image"]
            if mask is not None:
                mask = augmented["mask"]

        # Normalize to [0, 1] and convert to Tensor
        # Image: (H, W) -> (1, H, W) float32
        image = image.astype(np.float32) / 255.0
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)

        if mask is not None:
            # Mask: (H, W) -> (1, H, W) float32
            mask = mask.astype(np.float32)  # Already 0 or 1
            mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
            return image, mask, depth, self.ids[idx]
        else:
            return image, depth, self.ids[idx]


def get_dataloaders(config, load_cached_data=True):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        config (class): Configuration class.
        load_cached_data (bool): Whether to use cached numpy arrays.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load Data
    train_data = load_and_cache_data("train", config, load_cached_data)
    val_data = load_and_cache_data("val", config, load_cached_data)
    test_data = load_and_cache_data("test", config, load_cached_data)

    # Create Datasets
    train_dataset = SaltDataset(
        mode="train",
        config=config,
        data=train_data,
        transform=get_transforms("train", config),
    )

    val_dataset = SaltDataset(
        mode="val",
        config=config,
        data=val_data,
        transform=get_transforms("val", config),
    )

    test_dataset = SaltDataset(
        mode="test",
        config=config,
        data=test_data,
        transform=get_transforms("test", config),
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
