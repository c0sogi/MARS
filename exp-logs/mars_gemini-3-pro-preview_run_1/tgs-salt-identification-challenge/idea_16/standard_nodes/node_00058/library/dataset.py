import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import rle_decode


class SaltDataset(Dataset):
    """
    PyTorch Dataset for Salt Segmentation.
    Handles on-the-fly augmentation and tensor conversion.
    """

    def __init__(self, images, masks=None, ids=None, transform=False):
        """
        Args:
            images (np.ndarray): Array of images (N, H, W, C).
            masks (np.ndarray, optional): Array of masks (N, H, W, 1).
            ids (list): List of image IDs.
            transform (bool): Whether to apply data augmentation (Horizontal Flip).
        """
        self.images = images
        self.masks = masks
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load data
        image = self.images[idx]  # (H, W, 2)

        if self.masks is not None:
            mask = self.masks[idx]  # (H, W, 1)
        else:
            # Create dummy mask for test set
            mask = np.zeros((image.shape[0], image.shape[1], 1), dtype=np.float32)

        # Apply Augmentation (Horizontal Flip)
        if self.transform:
            # Random horizontal flip with p=0.5
            if np.random.rand() > 0.5:
                image = np.fliplr(image).copy()
                mask = np.fliplr(mask).copy()

        # Convert to Tensor and Permute (H, W, C) -> (C, H, W)
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
        mask_tensor = torch.from_numpy(mask.transpose(2, 0, 1)).float()

        return image_tensor, mask_tensor, self.ids[idx]


def pad_image(img, target_size=(128, 128)):
    """
    Pads an image to the target size using reflection padding.
    """
    h, w = img.shape[:2]
    target_h, target_w = target_size

    delta_h = target_h - h
    delta_w = target_w - w

    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    # Handle both 2D (H,W) and 3D (H,W,C) images
    if len(img.shape) == 2:
        padded = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_REFLECT_101
        )
        return padded
    else:
        # cv2.copyMakeBorder works on channels automatically for 3 channel,
        # but for custom channels or >3, we might need care.
        # Here we deal with 1-channel image or mask mostly before stacking.
        padded = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_REFLECT_101
        )
        # Ensure last dim is preserved if it was (H, W, 1)
        if padded.ndim == 2 and img.ndim == 3:
            padded = padded[:, :, np.newaxis]
        return padded


def preprocess_and_cache(df, mode, min_depth, max_depth, load_cached_data=True):
    """
    Loads raw data, applies deterministic preprocessing (padding, normalization, depth fusion),
    and caches the result as numpy arrays.
    """
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Define cache filenames
    cache_imgs = os.path.join(Config.CACHE_DIR, f"{mode}_images.npy")
    cache_masks = os.path.join(Config.CACHE_DIR, f"{mode}_masks.npy")
    cache_ids = os.path.join(Config.CACHE_DIR, f"{mode}_ids.npy")

    # Try to load from cache
    if load_cached_data:
        if os.path.exists(cache_imgs) and os.path.exists(cache_ids):
            # For test set, masks might not exist
            if mode == "test" or os.path.exists(cache_masks):
                print(f"Loading {mode} data from cache...")
                images = np.load(cache_imgs)
                ids = np.load(cache_ids)
                masks = np.load(cache_masks) if mode != "test" else None
                return images, masks, ids

    print(f"Processing {mode} data from scratch...")

    img_list = []
    mask_list = []
    id_list = []

    # Pre-calculate padding dimensions
    orig_h, orig_w = Config.ORIG_HEIGHT, Config.ORIG_WIDTH
    target_h, target_w = Config.IMG_HEIGHT, Config.IMG_WIDTH

    for _, row in df.iterrows():
        img_id = row["id"]

        # 1. Load Image
        # Construct full path. Metadata contains relative path like "train/images/xxxx.png"
        img_path = os.path.join(Config.INPUT_DIR, row["image_path"])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        # 2. Normalize Image (0-1)
        img = img.astype(np.float32) / 255.0

        # 3. Pad Image
        img_padded = pad_image(img, (target_h, target_w))

        # 4. Process Depth
        z = row["z"]
        z_norm = (z - min_depth) / (max_depth - min_depth)
        # Create depth channel (dense)
        depth_channel = np.full_like(img_padded, z_norm, dtype=np.float32)

        # 5. Stack Channels (Image + Depth)
        # Result shape: (H, W, 2)
        combined_img = np.dstack([img_padded, depth_channel])
        img_list.append(combined_img)

        # 6. Process Mask (if available)
        if mode != "test":
            rle = row["rle_mask"]
            # Decode RLE
            mask = rle_decode(rle, (orig_h, orig_w))
            mask = mask.astype(np.float32)
            # Pad Mask
            mask_padded = pad_image(mask, (target_h, target_w))
            # Add channel dim: (H, W, 1)
            mask_padded = mask_padded[:, :, np.newaxis]
            mask_list.append(mask_padded)

        id_list.append(img_id)

    # Convert to numpy arrays
    images_arr = np.array(img_list, dtype=np.float32)
    ids_arr = np.array(id_list)

    if mode != "test":
        masks_arr = np.array(mask_list, dtype=np.float32)
    else:
        masks_arr = None

    # Save to cache
    print(f"Saving {mode} data to cache at {Config.CACHE_DIR}...")
    np.save(cache_imgs, images_arr)
    np.save(cache_ids, ids_arr)
    if masks_arr is not None:
        np.save(cache_masks, masks_arr)

    return images_arr, masks_arr, ids_arr


def get_loaders(load_cached_data=True):
    """
    Main function to prepare DataLoaders.

    Args:
        load_cached_data (bool): If True, attempts to load pre-processed numpy arrays from disk.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # 1. Load Metadata
    df_train = pd.read_csv(Config.TRAIN_CSV)
    df_val = pd.read_csv(Config.VAL_CSV)
    df_test = pd.read_csv(Config.TEST_CSV)

    # 2. Determine Global Depth Statistics
    # We use the provided depths.csv to find global min/max for consistent normalization
    df_depths = pd.read_csv(Config.DEPTHS_CSV)
    min_depth = df_depths["z"].min()
    max_depth = df_depths["z"].max()

    # 3. Process Datasets
    # Train
    train_imgs, train_masks, train_ids = preprocess_and_cache(
        df_train, "train", min_depth, max_depth, load_cached_data
    )

    # Val
    val_imgs, val_masks, val_ids = preprocess_and_cache(
        df_val, "val", min_depth, max_depth, load_cached_data
    )

    # Test
    test_imgs, test_masks, test_ids = preprocess_and_cache(
        df_test, "test", min_depth, max_depth, load_cached_data
    )

    # 4. Create Dataset Instances
    train_dataset = SaltDataset(train_imgs, train_masks, train_ids, transform=True)

    val_dataset = SaltDataset(val_imgs, val_masks, val_ids, transform=False)

    test_dataset = SaltDataset(test_imgs, None, test_ids, transform=False)

    # 5. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
