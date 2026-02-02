import os
import cv2
import torch
import numpy as np
import pandas as pd
import random
from torch.utils.data import Dataset, DataLoader
from library.config import (
    INPUT_DIR,
    PATCH_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    WORKING_DIR,
)
from library.utils import seed_everything


def load_and_cache_data(metadata_path, mode, load_cached_data=True):
    """
    Loads image data from disk or cache.
    Implements a flat-array caching strategy to handle variable image sizes
    without using pickle, ensuring strict compliance with constraints.
    """
    cache_filename = f"{mode}_cache.npz"
    cache_path = os.path.join(WORKING_DIR, cache_filename)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            cached = np.load(cache_path)

            # Reconstruct IDs
            ids = list(cached["ids"])

            # Reconstruct Noisy Images
            noisy_flat = cached["noisy_flat"]
            noisy_shapes = cached["noisy_shapes"]
            noisy_imgs = []
            start_idx = 0
            for shape in noisy_shapes:
                size = shape[0] * shape[1]
                img = noisy_flat[start_idx : start_idx + size].reshape(shape)
                noisy_imgs.append(img)
                start_idx += size

            # Reconstruct Clean Images (if they exist)
            clean_imgs = []
            if "clean_flat" in cached and "clean_shapes" in cached:
                clean_flat = cached["clean_flat"]
                clean_shapes = cached["clean_shapes"]
                start_idx = 0
                for shape in clean_shapes:
                    size = shape[0] * shape[1]
                    img = clean_flat[start_idx : start_idx + size].reshape(shape)
                    clean_imgs.append(img)
                    start_idx += size

            return ids, noisy_imgs, clean_imgs
        except Exception as e:
            print(f"Failed to load cache for {mode}: {e}. Reloading from disk.")

    # 2. Load from disk if cache missing or failed
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)

    ids = []
    noisy_imgs = []
    clean_imgs = []

    # Pre-allocate lists for flat cache storage
    noisy_flat_list = []
    noisy_shapes_list = []
    clean_flat_list = []
    clean_shapes_list = []

    for _, row in df.iterrows():
        img_id = str(row["id"])
        ids.append(img_id)

        # Load Noisy Image
        noisy_path = os.path.join(INPUT_DIR, row["noisy_image_path"])
        n_img = cv2.imread(noisy_path, cv2.IMREAD_GRAYSCALE)
        if n_img is None:
            raise ValueError(f"Could not load image: {noisy_path}")

        # Normalize to [0, 1] float32
        n_img = n_img.astype(np.float32) / 255.0
        noisy_imgs.append(n_img)

        # Prepare for cache
        noisy_flat_list.append(n_img.flatten())
        noisy_shapes_list.append(n_img.shape)

        # Load Clean Image (if available)
        if "clean_image_path" in row and pd.notna(row["clean_image_path"]):
            clean_path = os.path.join(INPUT_DIR, row["clean_image_path"])
            c_img = cv2.imread(clean_path, cv2.IMREAD_GRAYSCALE)
            if c_img is None:
                raise ValueError(f"Could not load image: {clean_path}")

            # Normalize
            c_img = c_img.astype(np.float32) / 255.0
            clean_imgs.append(c_img)

            # Prepare for cache
            clean_flat_list.append(c_img.flatten())
            clean_shapes_list.append(c_img.shape)

    # 3. Save to cache
    os.makedirs(WORKING_DIR, exist_ok=True)

    save_dict = {
        "ids": np.array(ids),
        "noisy_flat": np.concatenate(noisy_flat_list),
        "noisy_shapes": np.array(noisy_shapes_list),
    }

    if clean_flat_list:
        save_dict["clean_flat"] = np.concatenate(clean_flat_list)
        save_dict["clean_shapes"] = np.array(clean_shapes_list)

    np.savez(cache_path, **save_dict)

    return ids, noisy_imgs, clean_imgs


class TextDenoisingDataset(Dataset):
    def __init__(self, metadata_path, mode="train", load_cached_data=True):
        """
        Args:
            metadata_path (str): Path to the metadata CSV.
            mode (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to use cached .npz data.
        """
        self.mode = mode
        self.patch_size = PATCH_SIZE

        # Load data
        self.ids, self.noisy_imgs, self.clean_imgs = load_and_cache_data(
            metadata_path, mode, load_cached_data
        )

        self.has_clean = len(self.clean_imgs) > 0

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        noisy = self.noisy_imgs[idx]
        img_id = self.ids[idx]

        clean = None
        if self.has_clean:
            clean = self.clean_imgs[idx]

        # --- Training: Random Crop & Augmentation ---
        if self.mode == "train":
            h, w = noisy.shape
            th, tw = self.patch_size, self.patch_size

            # Random Crop
            if w > tw and h > th:
                i = np.random.randint(0, h - th + 1)
                j = np.random.randint(0, w - tw + 1)
                noisy = noisy[i : i + th, j : j + tw]
                if clean is not None:
                    clean = clean[i : i + th, j : j + tw]
            else:
                # Resize or Pad if image is smaller than patch (unlikely given EDA)
                # For safety, we just center crop or return as is if smaller
                pass

            # Geometric Augmentations (D4 Group)
            # 1. Random Horizontal Flip
            if np.random.rand() < 0.5:
                noisy = np.fliplr(noisy)
                if clean is not None:
                    clean = np.fliplr(clean)

            # 2. Random Vertical Flip
            if np.random.rand() < 0.5:
                noisy = np.flipud(noisy)
                if clean is not None:
                    clean = np.flipud(clean)

            # 3. Random 90-degree Rotation
            k = np.random.randint(0, 4)
            if k > 0:
                noisy = np.rot90(noisy, k)
                if clean is not None:
                    clean = np.rot90(clean, k)

            # Ensure memory continuity after numpy flips/rotations for torch conversion
            noisy = noisy.copy()
            if clean is not None:
                clean = clean.copy()

        # --- Convert to Tensor ---
        # Add channel dimension: (H, W) -> (1, H, W)
        noisy_t = torch.from_numpy(noisy).float().unsqueeze(0)

        # For Test/Val where we might need ID
        if self.mode == "test" or self.mode == "val":
            # For validation, we might still want clean image if available
            if clean is not None:
                clean_t = torch.from_numpy(clean).float().unsqueeze(0)
                return noisy_t, clean_t, img_id
            return noisy_t, img_id

        if clean is not None:
            clean_t = torch.from_numpy(clean).float().unsqueeze(0)
            return noisy_t, clean_t

        return noisy_t


def worker_init_fn(worker_id):
    """
    Ensures reproducible randomness in data loader workers.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataloaders(load_cached_data=True):
    """
    Creates and returns DataLoaders for train, validation, and test sets.
    """
    # Train Dataset
    train_ds = TextDenoisingDataset(
        TRAIN_METADATA_PATH, mode="train", load_cached_data=load_cached_data
    )

    # Validation Dataset
    val_ds = TextDenoisingDataset(
        VAL_METADATA_PATH, mode="val", load_cached_data=load_cached_data
    )

    # Test Dataset
    test_ds = TextDenoisingDataset(
        TEST_METADATA_PATH, mode="test", load_cached_data=load_cached_data
    )

    # Train Loader
    # Uses configured BATCH_SIZE and shuffling
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        drop_last=True,
    )

    # Val & Test Loaders
    # Must use batch_size=1 because images have variable heights/widths
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )

    return train_loader, val_loader, test_loader
