import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

from library.config import (
    INPUT_DIR,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    WORKING_DIR,
    NUM_WORKERS,
    SEED,
    INVERT_INTENSITY,
)
from library.utils import invert_intensity


class DenoisingDataset(Dataset):
    """
    Dataset class for the Denoising Task.
    Handles signal inversion, caching, random cropping (dual-scale), and augmentation.
    """

    def __init__(
        self, ids, noisy_imgs, clean_imgs=None, patch_size=None, augment=False
    ):
        """
        Args:
            ids (list): List of image IDs.
            noisy_imgs (list of np.ndarray): List of noisy images (normalized, inverted).
            clean_imgs (list of np.ndarray, optional): List of clean images (normalized, inverted).
            patch_size (int, optional): Size of the square crop for training.
            augment (bool): Whether to apply geometric augmentations.
        """
        self.ids = ids
        self.noisy_imgs = noisy_imgs
        self.clean_imgs = clean_imgs
        self.patch_size = patch_size
        self.augment = augment

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # Retrieve images
        img_n = self.noisy_imgs[idx]  # (H, W)

        # Handle Clean Image (Target)
        if self.clean_imgs is not None:
            img_c = self.clean_imgs[idx]  # (H, W)
        else:
            # For test set, we don't have clean images.
            # We return a dummy target or just the input for consistency if needed,
            # but usually just returning input and ID is enough for inference.
            # Here we return the noisy image as placeholder for target to keep signature consistent.
            img_c = img_n.copy()

        # --- Training Logic: Random Crop & Augment ---
        if self.patch_size is not None:
            h, w = img_n.shape

            # Pad if image is smaller than patch_size
            pad_h = max(0, self.patch_size - h)
            pad_w = max(0, self.patch_size - w)

            if pad_h > 0 or pad_w > 0:
                # Pad with 0 (Background, since we inverted intensity)
                img_n = np.pad(
                    img_n, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0
                )
                img_c = np.pad(
                    img_c, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0
                )
                h, w = img_n.shape

            # Random Crop
            # Ensure we don't go out of bounds
            y = np.random.randint(0, h - self.patch_size + 1)
            x = np.random.randint(0, w - self.patch_size + 1)

            img_n = img_n[y : y + self.patch_size, x : x + self.patch_size]
            img_c = img_c[y : y + self.patch_size, x : x + self.patch_size]

            # Augmentation
            if self.augment:
                # Random Flip
                if np.random.rand() > 0.5:
                    img_n = np.flipud(img_n)
                    img_c = np.flipud(img_c)
                if np.random.rand() > 0.5:
                    img_n = np.fliplr(img_n)
                    img_c = np.fliplr(img_c)

                # Random Rotation (0, 90, 180, 270)
                k = np.random.randint(0, 4)
                if k > 0:
                    img_n = np.rot90(img_n, k)
                    img_c = np.rot90(img_c, k)

        # Convert to Tensor and add Channel Dimension: (H, W) -> (1, H, W)
        # Copy is required because numpy strides might be negative after flips
        tensor_n = torch.from_numpy(img_n.copy()).float().unsqueeze(0)
        tensor_c = torch.from_numpy(img_c.copy()).float().unsqueeze(0)

        return {"id": self.ids[idx], "input": tensor_n, "target": tensor_c}


def _process_image(path):
    """
    Reads an image, converts to grayscale, normalizes to [0, 1],
    and optionally inverts intensity.
    """
    full_path = os.path.join(INPUT_DIR, path)
    img = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise FileNotFoundError(f"Image not found at {full_path}")

    # Handle channels
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Normalize
    img = img.astype(np.float32) / 255.0

    # Invert Intensity (Signal Alignment)
    if INVERT_INTENSITY:
        img = invert_intensity(img)

    return img


def _load_data(metadata_path, split_name, load_cached_data=True):
    """
    Loads data from metadata CSV. Uses caching to speed up reloading.

    Args:
        metadata_path (str): Path to the CSV file.
        split_name (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from .npz cache.

    Returns:
        tuple: (ids, noisy_imgs, clean_imgs)
    """
    cache_path = os.path.join(WORKING_DIR, f"{split_name}_cache.npz")

    # 1. Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading {split_name} data from cache: {cache_path}")
        try:
            data = np.load(cache_path, allow_pickle=True)
            ids = data["ids"].tolist()
            noisy_imgs = list(data["noisy_imgs"])

            if "clean_imgs" in data:
                clean_imgs = list(data["clean_imgs"])
            else:
                clean_imgs = None

            return ids, noisy_imgs, clean_imgs
        except Exception as e:
            print(f"Failed to load cache: {e}. Reloading from source.")

    # 2. Load from source
    print(f"Processing {split_name} data from source...")
    df = pd.read_csv(metadata_path)

    ids = []
    noisy_imgs = []
    clean_imgs = [] if "clean_image_path" in df.columns else None

    for _, row in df.iterrows():
        ids.append(str(row["id"]))

        # Load Noisy
        noisy_imgs.append(_process_image(row["noisy_image_path"]))

        # Load Clean (if available)
        if clean_imgs is not None:
            clean_imgs.append(_process_image(row["clean_image_path"]))

    # 3. Save to cache
    print(f"Saving {split_name} data to cache: {cache_path}")
    save_dict = {
        "ids": np.array(ids),
        "noisy_imgs": np.array(
            noisy_imgs, dtype=object
        ),  # Object array for variable sizes
    }
    if clean_imgs is not None:
        save_dict["clean_imgs"] = np.array(clean_imgs, dtype=object)

    np.savez_compressed(cache_path, **save_dict)

    return ids, noisy_imgs, clean_imgs


def get_dataloaders(batch_size, patch_size=None, mode="train", load_cached_data=True):
    """
    Factory function to create DataLoaders.

    Args:
        batch_size (int): Batch size.
        patch_size (int, optional): Patch size for random cropping (Train only).
        mode (str): 'train' (returns train & val loaders), 'test' (returns test loader).
        load_cached_data (bool): Whether to use cached .npz files.

    Returns:
        DataLoader(s): (train_loader, val_loader) if mode='train', else test_loader.
    """

    # Reproducibility generator
    g = torch.Generator()
    g.manual_seed(SEED)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        import random

        random.seed(worker_seed)

    if mode == "train":
        # --- Load Train Data ---
        t_ids, t_noisy, t_clean = _load_data(
            TRAIN_METADATA_PATH, "train", load_cached_data
        )
        train_ds = DenoisingDataset(
            t_ids,
            t_noisy,
            t_clean,
            patch_size=patch_size,
            augment=True,  # Always augment training data
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=NUM_WORKERS,
            worker_init_fn=seed_worker,
            generator=g,
            pin_memory=True if torch.cuda.is_available() else False,
            drop_last=True,  # Drop incomplete batches for stability
        )

        # --- Load Val Data ---
        v_ids, v_noisy, v_clean = _load_data(VAL_METADATA_PATH, "val", load_cached_data)
        # Validation: No patching (full image), No augmentation
        val_ds = DenoisingDataset(
            v_ids, v_noisy, v_clean, patch_size=None, augment=False
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=1,  # Validation on full images -> batch size 1 is safest
            shuffle=False,
            num_workers=NUM_WORKERS,
            worker_init_fn=seed_worker,
            generator=g,
            pin_memory=True if torch.cuda.is_available() else False,
        )

        return train_loader, val_loader

    elif mode == "test":
        # --- Load Test Data ---
        test_ids, test_noisy, _ = _load_data(
            TEST_METADATA_PATH, "test", load_cached_data
        )

        test_ds = DenoisingDataset(
            test_ids, test_noisy, None, patch_size=None, augment=False
        )

        test_loader = DataLoader(
            test_ds,
            batch_size=1,  # Test on full images
            shuffle=False,
            num_workers=NUM_WORKERS,
            worker_init_fn=seed_worker,
            generator=g,
            pin_memory=True if torch.cuda.is_available() else False,
        )

        return test_loader

    else:
        raise ValueError(f"Unknown mode: {mode}")
