import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import seed_everything


def load_and_preprocess(csv_path, mode, load_cached_data=True):
    """
    Loads data from CSV, reads images/masks, applies padding, and caches results to .npy files.
    Strictly follows the caching logic: Try Load -> (Fail) -> Compute & Save.
    """
    # Ensure working directory exists
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # Define cache file paths
    f_imgs = os.path.join(cache_dir, f"{mode}_images.npy")
    f_masks = os.path.join(cache_dir, f"{mode}_masks.npy")
    f_depths = os.path.join(cache_dir, f"{mode}_depths.npy")
    f_ids = os.path.join(cache_dir, f"{mode}_ids.npy")

    # 1. IF load_cached_data is True: Try to load the file.
    if load_cached_data:
        if (
            os.path.exists(f_imgs)
            and os.path.exists(f_masks)
            and os.path.exists(f_depths)
            and os.path.exists(f_ids)
        ):
            try:
                images = np.load(f_imgs)
                masks = np.load(f_masks)
                depths = np.load(f_depths)
                ids = np.load(f_ids)
                return images, masks, depths, ids
            except Exception as e:
                print(f"Failed to load cached data for {mode}: {e}. Recomputing...")
        else:
            print(f"Cached data for {mode} not found. Computing from scratch...")

    # 2. IF loading fails OR load_cached_data is False: Compute/process from scratch.
    print(f"Processing {mode} data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Pre-allocate lists
    img_list = []
    mask_list = []
    depth_list = []
    id_list = []

    # Calculate padding
    # Target: 128, Original: 101
    # Total Pad: 27 -> Top: 13, Bottom: 14
    pad_h = Config.IMG_H - Config.ORIG_H
    pad_top = pad_h // 2
    pad_bot = pad_h - pad_top

    pad_w = Config.IMG_W - Config.ORIG_W
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    for idx, row in df.iterrows():
        # Load Image
        # Path in CSV is relative: train/images/xxxx.png
        # Full path: ./input/train/images/xxxx.png
        img_path = os.path.join(Config.INPUT_DIR, row["image_path"])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        # Reflection Padding
        img_padded = cv2.copyMakeBorder(
            img, pad_top, pad_bot, pad_left, pad_right, cv2.BORDER_REFLECT_101
        )
        img_list.append(img_padded)

        # Load Mask
        # Test set has no masks (mask_path is NaN or None)
        mask_path = row.get("mask_path")
        if pd.notna(mask_path) and isinstance(mask_path, str):
            full_mask_path = os.path.join(Config.INPUT_DIR, mask_path)
            mask = cv2.imread(full_mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                # Fallback if file missing but path exists (shouldn't happen per validation)
                mask = np.zeros((Config.ORIG_H, Config.ORIG_W), dtype=np.uint8)
        else:
            # Create dummy mask for test set
            mask = np.zeros((Config.ORIG_H, Config.ORIG_W), dtype=np.uint8)

        # Reflection Padding for Mask
        mask_padded = cv2.copyMakeBorder(
            mask, pad_top, pad_bot, pad_left, pad_right, cv2.BORDER_REFLECT_101
        )
        mask_list.append(mask_padded)

        # Depth and ID
        depth_list.append(row["z"])
        id_list.append(row["id"])

    # Convert to numpy arrays
    images = np.array(img_list, dtype=np.uint8)
    masks = np.array(mask_list, dtype=np.uint8)
    depths = np.array(depth_list, dtype=np.float32)
    ids = np.array(id_list)

    # Save the result to the cache directory
    np.save(f_imgs, images)
    np.save(f_masks, masks)
    np.save(f_depths, depths)
    np.save(f_ids, ids)

    print(f"Saved processed {mode} data to {cache_dir}")

    # 3. Return the data.
    return images, masks, depths, ids


class SaltDataset(Dataset):
    def __init__(self, images, masks, depths, ids, mode="train"):
        """
        Args:
            images: np.array (N, H, W) uint8
            masks: np.array (N, H, W) uint8
            depths: np.array (N,) float32
            ids: np.array (N,) string
            mode: 'train', 'val', or 'test'
        """
        self.images = images
        self.masks = masks
        self.depths = depths
        self.ids = ids
        self.mode = mode

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Retrieve data
        img = self.images[idx]
        mask = self.masks[idx]
        z = self.depths[idx]
        img_id = self.ids[idx]

        # Augmentation (Train only)
        # Horizontal Flip
        if self.mode == "train" and np.random.rand() > 0.5:
            img = cv2.flip(img, 1)
            mask = cv2.flip(mask, 1)

        # Normalize Image: 0-255 -> 0.0-1.0
        img = img.astype(np.float32) / 255.0

        # Normalize Mask: 0-255 -> 0.0-1.0 (Binary)
        # Threshold to ensure binary (some resizing might introduce artifacts, though reflection shouldn't)
        mask = (mask > 127).astype(np.float32)

        # Convert to Tensor (C, H, W)
        img_tensor = torch.from_numpy(img).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)

        # Depth is a scalar float32
        z_tensor = torch.tensor(z, dtype=torch.float32)

        return img_tensor, mask_tensor, z_tensor, img_id


def get_loaders(debug=False, load_cached_data=True):
    """
    Creates DataLoaders for training and validation.

    Args:
        debug (bool): If True, subsets data to 100 samples.
        load_cached_data (bool): Whether to use cached .npy files.

    Returns:
        train_loader, val_loader
    """
    # Set seed for reproducibility
    seed_everything(Config.SEED)

    # Load Train Data
    train_imgs, train_masks, train_depths, train_ids = load_and_preprocess(
        Config.TRAIN_CSV, "train", load_cached_data=load_cached_data
    )

    # Load Val Data
    val_imgs, val_masks, val_depths, val_ids = load_and_preprocess(
        Config.VAL_CSV, "val", load_cached_data=load_cached_data
    )

    # Debug subsetting
    if debug:
        train_imgs = train_imgs[:100]
        train_masks = train_masks[:100]
        train_depths = train_depths[:100]
        train_ids = train_ids[:100]

        val_imgs = val_imgs[:100]
        val_masks = val_masks[:100]
        val_depths = val_depths[:100]
        val_ids = val_ids[:100]

    # Create Datasets
    train_dataset = SaltDataset(
        train_imgs, train_masks, train_depths, train_ids, mode="train"
    )
    val_dataset = SaltDataset(val_imgs, val_masks, val_depths, val_ids, mode="val")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def get_test_loader(load_cached_data=True):
    """
    Creates DataLoader for the test set.
    """
    test_imgs, test_masks, test_depths, test_ids = load_and_preprocess(
        Config.TEST_CSV, "test", load_cached_data=load_cached_data
    )

    test_dataset = SaltDataset(
        test_imgs, test_masks, test_depths, test_ids, mode="test"
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return test_loader
