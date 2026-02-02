import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# -------------------------------------------------------------------------
# Constants and Configuration
# -------------------------------------------------------------------------
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
CACHE_DIR = "./working/idea_14"
ORIG_H, ORIG_W = 101, 101
TARGET_H, TARGET_W = 128, 128


def pad_image(img):
    """
    Pads an image from 101x101 to 128x128 using reflection padding.
    Args:
        img (np.ndarray): Input image of shape (101, 101) or (101, 101, C).
    Returns:
        np.ndarray: Padded image of shape (128, 128) or (128, 128, C).
    """
    h, w = img.shape[:2]
    if h == TARGET_H and w == TARGET_W:
        return img

    pad_h = TARGET_H - h
    pad_w = TARGET_W - w

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    # Use Reflection Padding as per strategy
    # BORDER_REFLECT matches the requirement to handle dimensions without artifacts
    padded = cv2.copyMakeBorder(
        img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT
    )
    return padded


def load_and_preprocess_data(mode, load_cached_data=True):
    """
    Loads data from metadata CSVs, preprocesses images (padding/normalization),
    and caches them as .npy files for deterministic and fast loading.

    Args:
        mode (str): 'train', 'val', or 'test'.
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        tuple: (images, masks, depths, ids) as numpy arrays.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Define cache file paths
    cache_images = os.path.join(CACHE_DIR, f"{mode}_images.npy")
    cache_masks = os.path.join(CACHE_DIR, f"{mode}_masks.npy")
    cache_depths = os.path.join(CACHE_DIR, f"{mode}_depths.npy")
    cache_ids = os.path.join(CACHE_DIR, f"{mode}_ids.npy")

    # 1. Try Loading from Cache
    if load_cached_data:
        # Check if essential files exist
        files_exist = (
            os.path.exists(cache_images)
            and os.path.exists(cache_depths)
            and os.path.exists(cache_ids)
        )
        # For train/val, masks must also exist
        if mode != "test" and not os.path.exists(cache_masks):
            files_exist = False

        if files_exist:
            print(f"Loading cached {mode} data from {CACHE_DIR}...")
            images = np.load(cache_images)
            depths = np.load(cache_depths)
            ids = np.load(cache_ids)
            masks = np.load(cache_masks) if mode != "test" else None
            return images, masks, depths, ids

    # 2. Process from Scratch
    print(f"Processing {mode} data from scratch...")

    csv_path = os.path.join(METADATA_DIR, f"{mode}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    images_list = []
    masks_list = []
    depths_list = []
    ids_list = []

    for _, row in df.iterrows():
        img_id = row["id"]
        depth = row["z"]
        img_rel_path = row["image_path"]

        # Load Image
        img_path = os.path.join(INPUT_DIR, img_rel_path)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Fallback or error; assuming data integrity based on metadata check
            raise FileNotFoundError(f"Image not found: {img_path}")

        # Preprocess Image: Pad -> Normalize
        img_padded = pad_image(img)
        img_normalized = img_padded.astype(np.float32) / 255.0
        # Expand dims: (128, 128) -> (128, 128, 1)
        img_normalized = np.expand_dims(img_normalized, axis=-1)

        images_list.append(img_normalized)
        depths_list.append(depth)
        ids_list.append(img_id)

        # Load Mask (if available)
        if mode != "test":
            mask_rel_path = row["mask_path"]
            mask_path = os.path.join(INPUT_DIR, mask_rel_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                # Should not happen given metadata validation
                mask = np.zeros((ORIG_H, ORIG_W), dtype=np.uint8)

            # Preprocess Mask: Pad -> Binarize -> Normalize
            mask_padded = pad_image(mask)
            mask_normalized = (mask_padded > 127).astype(np.float32)
            # Expand dims: (128, 128) -> (128, 128, 1)
            mask_normalized = np.expand_dims(mask_normalized, axis=-1)

            masks_list.append(mask_normalized)

    # Convert to Numpy Arrays
    images_arr = np.array(images_list)
    depths_arr = np.array(depths_list)
    ids_arr = np.array(ids_list)
    masks_arr = np.array(masks_list) if mode != "test" else None

    # Save to Cache
    np.save(cache_images, images_arr)
    np.save(cache_depths, depths_arr)
    np.save(cache_ids, ids_arr)
    if mode != "test":
        np.save(cache_masks, masks_arr)

    return images_arr, masks_arr, depths_arr, ids_arr


class SaltDataset(Dataset):
    def __init__(self, images, masks, depths, ids, mode="train"):
        """
        Args:
            images (np.ndarray): Shape (N, 128, 128, 1)
            masks (np.ndarray): Shape (N, 128, 128, 1) or None
            depths (np.ndarray): Shape (N,)
            ids (np.ndarray): Shape (N,)
            mode (str): 'train', 'val', or 'test'
        """
        self.images = images
        self.masks = masks
        self.depths = depths
        self.ids = ids
        self.mode = mode

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # Retrieve data
        image = self.images[idx]  # (128, 128, 1)
        depth = self.depths[idx]
        img_id = self.ids[idx]

        if self.mode != "test":
            mask = self.masks[idx]  # (128, 128, 1)

            # Augmentation: Horizontal Flip (Train only)
            if self.mode == "train":
                if np.random.rand() > 0.5:
                    # Flip width (axis 1)
                    image = np.flip(image, axis=1)
                    mask = np.flip(mask, axis=1)

            # Convert to Tensor (C, H, W)
            # Use .copy() to handle negative strides from np.flip
            image_tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).float()
            mask_tensor = torch.from_numpy(mask.copy()).permute(2, 0, 1).float()
            depth_tensor = torch.tensor([depth], dtype=torch.float32)

            return image_tensor, mask_tensor, depth_tensor, img_id

        else:
            # Test Mode
            image_tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).float()
            depth_tensor = torch.tensor([depth], dtype=torch.float32)
            # Return zero mask for consistency in collation
            mask_tensor = torch.zeros_like(image_tensor)

            return image_tensor, mask_tensor, depth_tensor, img_id


def get_dataloaders(batch_size=32, load_cached_data=True, num_workers=2):
    """
    Creates and returns DataLoaders for train, val, and test sets.
    """
    # 1. Load Data (with caching)
    train_imgs, train_masks, train_depths, train_ids = load_and_preprocess_data(
        "train", load_cached_data
    )
    val_imgs, val_masks, val_depths, val_ids = load_and_preprocess_data(
        "val", load_cached_data
    )
    test_imgs, test_masks, test_depths, test_ids = load_and_preprocess_data(
        "test", load_cached_data
    )

    # 2. Instantiate Datasets
    train_dataset = SaltDataset(
        train_imgs, train_masks, train_depths, train_ids, mode="train"
    )
    val_dataset = SaltDataset(val_imgs, val_masks, val_depths, val_ids, mode="val")
    test_dataset = SaltDataset(
        test_imgs, test_masks, test_depths, test_ids, mode="test"
    )

    # 3. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
