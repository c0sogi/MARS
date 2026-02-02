import os
import cv2
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import seed_everything


def save_cache(path, data_list):
    """
    Saves data_list to a compressed .npz file to avoid pickle object serialization.
    data_list: List of dicts {'id': str, 'noisy': np.ndarray, 'clean': np.ndarray or None}
    """
    save_dict = {}
    ids = []
    has_clean = []

    for i, item in enumerate(data_list):
        save_dict[f"n_{i}"] = item["noisy"]
        if item["clean"] is not None:
            save_dict[f"c_{i}"] = item["clean"]
            has_clean.append(True)
        else:
            has_clean.append(False)
        ids.append(item["id"])

    save_dict["ids"] = np.array(ids)
    save_dict["has_clean"] = np.array(has_clean)

    np.savez_compressed(path, **save_dict)


def load_cache(path):
    """
    Loads data from .npz file.
    """
    loaded = np.load(path)
    ids = loaded["ids"]
    has_clean = loaded["has_clean"]
    data_list = []

    for i, img_id in enumerate(ids):
        item = {"id": str(img_id)}
        item["noisy"] = loaded[f"n_{i}"]
        if has_clean[i]:
            item["clean"] = loaded[f"c_{i}"]
        else:
            item["clean"] = None
        data_list.append(item)

    return data_list


def load_data_from_metadata(metadata_path, cache_path, load_cached_data=True):
    """
    Loads image data from metadata CSV. Uses caching mechanism.
    """
    # 1. Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            data = load_cache(cache_path)
            # Apply debug limit if necessary after loading
            if Config.DEBUG:
                data = data[: Config.MAX_DEBUG_SAMPLES]
            return data
        except Exception as e:
            print(f"Cache load failed: {e}. Re-processing data.")

    # 2. Process from source
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)
    data_list = []

    for _, row in df.iterrows():
        entry = {"id": str(row["id"])}

        # Load Noisy Image
        noisy_path = os.path.join(Config.INPUT_DIR, row["noisy_image_path"])
        # Load as grayscale (H, W)
        noisy_img = cv2.imread(noisy_path, cv2.IMREAD_GRAYSCALE)
        if noisy_img is None:
            continue
        entry["noisy"] = noisy_img

        # Load Clean Image (if available)
        if "clean_image_path" in row and pd.notna(row["clean_image_path"]):
            clean_path = os.path.join(Config.INPUT_DIR, row["clean_image_path"])
            clean_img = cv2.imread(clean_path, cv2.IMREAD_GRAYSCALE)
            entry["clean"] = clean_img
        else:
            entry["clean"] = None

        data_list.append(entry)

    # 3. Save to cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    save_cache(cache_path, data_list)

    # Apply debug limit
    if Config.DEBUG:
        data_list = data_list[: Config.MAX_DEBUG_SAMPLES]

    return data_list


class DenoisingDataset(Dataset):
    def __init__(self, data_list, mode="train"):
        self.data = data_list
        self.mode = mode
        self.patch_size = Config.PATCH_SIZE

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Normalize to [0, 1]
        noisy = item["noisy"].astype(np.float32) / 255.0
        clean = (
            item["clean"].astype(np.float32) / 255.0
            if item["clean"] is not None
            else None
        )
        img_id = item["id"]

        if self.mode == "train":
            # --- Training Augmentations ---

            # 1. Pad if smaller than patch size
            h, w = noisy.shape
            pad_h = max(0, self.patch_size - h)
            pad_w = max(0, self.patch_size - w)

            if pad_h > 0 or pad_w > 0:
                noisy = np.pad(noisy, ((0, pad_h), (0, pad_w)), mode="reflect")
                if clean is not None:
                    clean = np.pad(clean, ((0, pad_h), (0, pad_w)), mode="reflect")

            # 2. Random Crop
            h_new, w_new = noisy.shape
            top = np.random.randint(0, h_new - self.patch_size + 1)
            left = np.random.randint(0, w_new - self.patch_size + 1)

            noisy = noisy[top : top + self.patch_size, left : left + self.patch_size]
            if clean is not None:
                clean = clean[
                    top : top + self.patch_size, left : left + self.patch_size
                ]

            # 3. Geometric Augmentations
            # Random Horizontal Flip
            if np.random.rand() > 0.5:
                noisy = np.flip(noisy, axis=1)
                if clean is not None:
                    clean = np.flip(clean, axis=1)

            # Random Vertical Flip
            if np.random.rand() > 0.5:
                noisy = np.flip(noisy, axis=0)
                if clean is not None:
                    clean = np.flip(clean, axis=0)

            # Random 90-degree Rotations
            k = np.random.randint(0, 4)
            if k > 0:
                noisy = np.rot90(noisy, k)
                if clean is not None:
                    clean = np.rot90(clean, k)

            # Ensure memory contiguity after numpy strides
            noisy = np.ascontiguousarray(noisy)
            if clean is not None:
                clean = np.ascontiguousarray(clean)

        # Convert to Tensor (C, H, W) -> (1, H, W)
        noisy_t = torch.from_numpy(noisy).unsqueeze(0)

        if clean is not None:
            clean_t = torch.from_numpy(clean).unsqueeze(0)
            return noisy_t, clean_t, img_id
        else:
            return noisy_t, img_id


def get_dataloaders(load_cached_data=True):
    """
    Creates DataLoaders for train, val, and test sets.
    """
    # Define cache paths
    train_cache = os.path.join(Config.WORKING_DIR, "train_cache.npz")
    val_cache = os.path.join(Config.WORKING_DIR, "val_cache.npz")
    test_cache = os.path.join(Config.WORKING_DIR, "test_cache.npz")

    # Load Data
    train_data = load_data_from_metadata(
        Config.TRAIN_METADATA_PATH, train_cache, load_cached_data
    )
    val_data = load_data_from_metadata(
        Config.VAL_METADATA_PATH, val_cache, load_cached_data
    )
    test_data = load_data_from_metadata(
        Config.TEST_METADATA_PATH, test_cache, load_cached_data
    )

    # Create Datasets
    train_ds = DenoisingDataset(train_data, mode="train")
    val_ds = DenoisingDataset(val_data, mode="val")
    test_ds = DenoisingDataset(test_data, mode="test")

    # Deterministic Worker Init
    def worker_init_fn(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    # Create Loaders
    # Train: Batched, Shuffled
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        worker_init_fn=worker_init_fn,
        pin_memory=True,
    )

    # Val/Test: Batch Size 1 (images have variable heights), No Shuffle
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
