import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from library.config import INPUT_DIR, WORKING_DIR
from library.utils import invert_signal


def load_and_cache_data(metadata_path, cache_name, load_cached_data=True, limit=None):
    """
    Loads image data based on metadata CSV and caches the result.

    Args:
        metadata_path (str): Path to the metadata CSV file.
        cache_name (str): Filename for the cache (e.g., 'train_cache.npz').
        load_cached_data (bool): If True, attempts to load from existing cache.
        limit (int, optional): Limits the number of samples loaded (for debugging).

    Returns:
        list: A list of dictionaries containing 'id', 'noisy', and optionally 'clean' image data.
    """
    cache_path = os.path.join(WORKING_DIR, cache_name)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            print(f"Loading cached data from {cache_path}")
            data = np.load(cache_path, allow_pickle=True)
            ids = data["ids"]
            noisy = data["noisy"]

            dataset = []
            if "clean" in data:
                clean = data["clean"]
                for i in range(len(ids)):
                    dataset.append(
                        {"id": str(ids[i]), "noisy": noisy[i], "clean": clean[i]}
                    )
            else:
                for i in range(len(ids)):
                    dataset.append({"id": str(ids[i]), "noisy": noisy[i]})

            if limit:
                return dataset[:limit]
            return dataset
        except Exception as e:
            print(f"Failed to load cache: {e}. Processing from source.")

    # 2. Process from source
    print(f"Processing data from {metadata_path}")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)
    if limit:
        df = df.iloc[:limit]

    ids = []
    noisy_imgs = []
    clean_imgs = []
    has_clean = "clean_image_path" in df.columns

    for _, row in df.iterrows():
        # Load Noisy Image
        n_path = os.path.join(INPUT_DIR, row["noisy_image_path"])
        n_img = cv2.imread(n_path, cv2.IMREAD_GRAYSCALE)

        if n_img is None:
            continue

        ids.append(str(row["id"]))
        noisy_imgs.append(n_img)  # Store as uint8 to save memory/cache size

        # Load Clean Image if available
        if has_clean:
            c_path = os.path.join(INPUT_DIR, row["clean_image_path"])
            c_img = cv2.imread(c_path, cv2.IMREAD_GRAYSCALE)
            if c_img is None:
                # Fallback if file missing, though metadata check passed
                c_img = np.zeros_like(n_img)
            clean_imgs.append(c_img)

    # 3. Save to Cache
    # Use object array to handle potentially variable image sizes
    save_dict = {"ids": np.array(ids), "noisy": np.array(noisy_imgs, dtype=object)}
    if has_clean:
        save_dict["clean"] = np.array(clean_imgs, dtype=object)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(cache_path, **save_dict)

    # 4. Construct Return List
    dataset = []
    for i in range(len(ids)):
        item = {"id": ids[i], "noisy": noisy_imgs[i]}
        if has_clean:
            item["clean"] = clean_imgs[i]
        dataset.append(item)

    return dataset


class DenoisingDataset(Dataset):
    def __init__(self, data, patch_size=None, augment=False, mode="train"):
        """
        Args:
            data (list): List of data dictionaries.
            patch_size (int, optional): Size of the square crop (e.g. 320 or 160).
            augment (bool): Whether to apply geometric augmentations.
            mode (str): 'train', 'val', or 'test'.
        """
        self.data = data
        self.patch_size = patch_size
        self.augment = augment
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Normalize to [0, 1] float
        noisy = item["noisy"].astype(np.float32) / 255.0
        clean = None
        if "clean" in item:
            clean = item["clean"].astype(np.float32) / 255.0

        h, w = noisy.shape
        original_shape = (h, w)

        # Padding Value: 1.0 (White)
        # We pad with 1.0 because we will later invert the signal (1.0 - x).
        # So 1.0 becomes 0.0 (background), compatible with zero-padding bias.
        pad_val = 1.0

        if self.mode == "train" and self.patch_size:
            # --- Training: Random Crop & Augment ---

            # 1. Pad if image is smaller than patch_size
            if h < self.patch_size or w < self.patch_size:
                pad_h = max(0, self.patch_size - h)
                pad_w = max(0, self.patch_size - w)
                noisy = np.pad(noisy, ((0, pad_h), (0, pad_w)), constant_values=pad_val)
                if clean is not None:
                    clean = np.pad(
                        clean, ((0, pad_h), (0, pad_w)), constant_values=pad_val
                    )
                h, w = noisy.shape

            # 2. Random Crop
            y = np.random.randint(0, h - self.patch_size + 1)
            x = np.random.randint(0, w - self.patch_size + 1)

            noisy = noisy[y : y + self.patch_size, x : x + self.patch_size]
            if clean is not None:
                clean = clean[y : y + self.patch_size, x : x + self.patch_size]

            # 3. Augmentation
            if self.augment:
                # Random Rotate 90
                k = np.random.randint(0, 4)
                noisy = np.rot90(noisy, k)
                if clean is not None:
                    clean = np.rot90(clean, k)

                # Random Flip Vertical
                if np.random.random() > 0.5:
                    noisy = np.flipud(noisy)
                    if clean is not None:
                        clean = np.flipud(clean)

                # Random Flip Horizontal
                if np.random.random() > 0.5:
                    noisy = np.fliplr(noisy)
                    if clean is not None:
                        clean = np.fliplr(clean)
        else:
            # --- Val/Test: Pad to multiple of 32 ---
            # Ensures compatibility with U-Net downsampling (factor 16 or 32)
            multiple = 32
            new_h = ((h - 1) // multiple + 1) * multiple
            new_w = ((w - 1) // multiple + 1) * multiple

            pad_h = new_h - h
            pad_w = new_w - w

            if pad_h > 0 or pad_w > 0:
                noisy = np.pad(noisy, ((0, pad_h), (0, pad_w)), constant_values=pad_val)
                if clean is not None:
                    clean = np.pad(
                        clean, ((0, pad_h), (0, pad_w)), constant_values=pad_val
                    )

        # Convert to Tensor [C, H, W]
        # .copy() is required because numpy flips/rotations can create negative strides
        noisy_t = torch.from_numpy(noisy.copy()).float().unsqueeze(0)

        # Invert Signal (1.0 - x)
        # This maps white background (1.0) to 0.0, aligning with zero-padding
        noisy_t = invert_signal(noisy_t)

        ret = {
            "noisy": noisy_t,
            "id": item["id"],
            "original_shape": torch.tensor(original_shape),
        }

        if clean is not None:
            clean_t = torch.from_numpy(clean.copy()).float().unsqueeze(0)
            clean_t = invert_signal(clean_t)
            ret["clean"] = clean_t

        return ret


def worker_init_fn(worker_id):
    """
    Initializes workers with deterministic seeds.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    import random

    random.seed(worker_seed)
