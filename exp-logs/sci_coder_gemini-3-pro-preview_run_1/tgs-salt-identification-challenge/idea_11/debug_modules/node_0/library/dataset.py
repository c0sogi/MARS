import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_pad_amounts(orig_size=101, target_size=128):
    """
    Calculates padding amounts to center the image.
    Returns: (top, bottom, left, right)
    """
    diff = target_size - orig_size
    pad_top = diff // 2
    pad_bottom = diff - pad_top
    pad_left = diff // 2
    pad_right = diff - pad_left
    return pad_top, pad_bottom, pad_left, pad_right


def load_dataset_arrays(mode, csv_path, load_cached_data=True, limit=None):
    """
    Loads dataset arrays from cache or processes them from scratch.

    Args:
        mode (str): 'train', 'val', or 'test'.
        csv_path (str): Path to the metadata CSV.
        load_cached_data (bool): Whether to attempt loading from cache.
        limit (int, optional): Limit number of samples (for debugging).

    Returns:
        tuple: (ids, images, masks, depths)
            ids: np.array of strings
            images: np.array of shape (N, 128, 128, 1)
            masks: np.array of shape (N, 128, 128, 1) (or None for test)
            depths: np.array of shape (N, 1)
    """
    # Define cache paths
    cache_dir = Config.CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    ids_path = os.path.join(cache_dir, f"{mode}_ids.npy")
    images_path = os.path.join(cache_dir, f"{mode}_images.npy")
    masks_path = os.path.join(cache_dir, f"{mode}_masks.npy")
    depths_path = os.path.join(cache_dir, f"{mode}_depths.npy")

    # Check if cache exists
    cache_exists = (
        os.path.exists(ids_path)
        and os.path.exists(images_path)
        and os.path.exists(depths_path)
    )

    if mode != "test":
        cache_exists = cache_exists and os.path.exists(masks_path)

    # 1. Load from Cache
    if load_cached_data and cache_exists:
        print(f"Loading {mode} data from cache: {cache_dir}")
        ids = np.load(ids_path)
        images = np.load(images_path)
        depths = np.load(depths_path)

        if mode != "test":
            masks = np.load(masks_path)
        else:
            masks = np.zeros_like(images)  # Dummy masks for test

        # Handle limit after loading
        if limit is not None:
            ids = ids[:limit]
            images = images[:limit]
            masks = masks[:limit]
            depths = depths[:limit]

        return ids, images, masks, depths

    # 2. Process from Scratch
    print(f"Processing {mode} data from source...")
    df = pd.read_csv(csv_path)

    if limit is not None:
        df = df.iloc[:limit]

    ids_list = []
    images_list = []
    masks_list = []
    depths_list = []

    pad_top, pad_bot, pad_left, pad_right = get_pad_amounts(
        Config.ORIG_SIZE, Config.INPUT_SIZE
    )

    for idx, row in df.iterrows():
        # Load ID and Depth
        img_id = str(row["id"])
        depth = float(row["z"])

        # Load Image
        img_rel_path = row["image_path"]
        img_full_path = os.path.join(Config.INPUT_DIR, img_rel_path)

        # Read as grayscale
        img = cv2.imread(img_full_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_full_path}")

        # Reflection Padding
        img_padded = cv2.copyMakeBorder(
            img, pad_top, pad_bot, pad_left, pad_right, cv2.BORDER_REFLECT
        )
        # Normalize to 0-1
        img_padded = img_padded.astype(np.float32) / 255.0
        # Add channel dim: (H, W) -> (H, W, 1)
        img_padded = np.expand_dims(img_padded, axis=-1)

        # Load Mask (if available)
        mask_padded = None
        if mode != "test":
            mask_rel_path = row["mask_path"]
            mask_full_path = os.path.join(Config.INPUT_DIR, mask_rel_path)

            mask = cv2.imread(mask_full_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"Mask not found: {mask_full_path}")

            # Reflection Padding for Mask
            mask_padded = cv2.copyMakeBorder(
                mask, pad_top, pad_bot, pad_left, pad_right, cv2.BORDER_REFLECT
            )
            # Normalize to 0-1 binary
            mask_padded = (mask_padded > 127).astype(np.float32)
            mask_padded = np.expand_dims(mask_padded, axis=-1)
        else:
            # Create dummy mask for test
            mask_padded = np.zeros(
                (Config.INPUT_SIZE, Config.INPUT_SIZE, 1), dtype=np.float32
            )

        ids_list.append(img_id)
        images_list.append(img_padded)
        masks_list.append(mask_padded)
        depths_list.append(depth)

    # Convert to numpy arrays
    ids_arr = np.array(ids_list)
    images_arr = np.array(images_list, dtype=np.float32)
    masks_arr = np.array(masks_list, dtype=np.float32)
    depths_arr = np.array(depths_list, dtype=np.float32).reshape(-1, 1)

    # Save to cache (only if not limited, to avoid overwriting full cache with partial data)
    if limit is None:
        print(f"Saving {mode} data to cache: {cache_dir}")
        np.save(ids_path, ids_arr)
        np.save(images_path, images_arr)
        np.save(depths_path, depths_arr)
        if mode != "test":
            np.save(masks_path, masks_arr)

    return ids_arr, images_arr, masks_arr, depths_arr


class SaltDataset(Dataset):
    """
    PyTorch Dataset for Salt Segmentation.
    Handles loading, preprocessing, and augmentation.
    """

    def __init__(self, mode="train", load_cached_data=True, limit=None):
        """
        Args:
            mode (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to use cached .npy files.
            limit (int, optional): Limit dataset size for debugging.
        """
        self.mode = mode

        # Select CSV path based on mode
        if mode == "train":
            self.csv_path = Config.TRAIN_CSV
        elif mode == "val":
            self.csv_path = Config.VAL_CSV
        elif mode == "test":
            self.csv_path = Config.TEST_CSV
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Load data
        self.ids, self.images, self.masks, self.depths = load_dataset_arrays(
            mode, self.csv_path, load_cached_data, limit
        )

        print(f"[{mode.upper()} Dataset] Loaded {len(self.ids)} samples.")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # Retrieve data
        image = self.images[idx]  # (H, W, 1)
        mask = self.masks[idx]  # (H, W, 1)
        depth = self.depths[idx]  # (1,)
        img_id = self.ids[idx]

        # Augmentation (Horizontal Flip only) - Training only
        if self.mode == "train" and Config.AUGMENTATION_FLIP_ONLY:
            if np.random.rand() > 0.5:
                # Flip along width (axis 1)
                image = np.flip(image, axis=1).copy()
                mask = np.flip(mask, axis=1).copy()

        # Convert to PyTorch Tensors
        # Image/Mask: (H, W, C) -> (C, H, W)
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
        mask_tensor = torch.from_numpy(mask.transpose(2, 0, 1)).float()
        depth_tensor = torch.tensor(depth, dtype=torch.float32)

        return image_tensor, mask_tensor, depth_tensor, img_id
