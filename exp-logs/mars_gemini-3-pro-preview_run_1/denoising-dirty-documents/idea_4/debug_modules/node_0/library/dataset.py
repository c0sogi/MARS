import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import random
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import seed_everything


def _load_image(path):
    """
    Loads an image from disk, converts to grayscale, and normalizes to [0, 1].
    Returns a numpy array of shape (1, H, W).
    """
    full_path = os.path.join(Config.INPUT_DIR, path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Image not found: {full_path}")

    # Load as grayscale
    img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to load image: {full_path}")

    # Normalize to [0, 1] float32
    img = img.astype(np.float32) / 255.0

    # Add channel dimension: (H, W) -> (1, H, W)
    img = img[np.newaxis, :, :]
    return img


def _load_and_cache_data(csv_path, cache_name, load_cached_data=True):
    """
    Loads data from CSV/Images or from a cached .npz file.
    Enforces deterministic loading and caching without pickle.
    """
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    cache_path = os.path.join(Config.WORKING_DIR, f"{cache_name}.npz")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading {cache_name} data from cache: {cache_path}")
        try:
            # allow_pickle=False is default, ensuring we use safe loading
            with np.load(cache_path) as data:
                ids = data["ids"]
                dataset = []
                for i, img_id in enumerate(ids):
                    item = {"id": str(img_id)}
                    # Retrieve arrays using keys
                    item["noisy"] = data[f"noisy_{i}"]
                    if f"clean_{i}" in data:
                        item["clean"] = data[f"clean_{i}"]
                    dataset.append(item)
            return dataset
        except Exception as e:
            print(f"Failed to load cache: {e}. Reloading from source.")

    # 2. Load from source
    print(f"Loading {cache_name} data from source CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    dataset = []
    cache_dict = {}
    ids_list = []

    for idx, row in df.iterrows():
        img_id = str(row["id"])
        item = {"id": img_id}

        # Load Noisy
        noisy_img = _load_image(row["noisy_image_path"])
        item["noisy"] = noisy_img
        cache_dict[f"noisy_{idx}"] = noisy_img

        # Load Clean (if available)
        if "clean_image_path" in row and pd.notna(row["clean_image_path"]):
            clean_img = _load_image(row["clean_image_path"])
            item["clean"] = clean_img
            cache_dict[f"clean_{idx}"] = clean_img

        dataset.append(item)
        ids_list.append(img_id)

    # 3. Save to cache
    print(f"Saving {cache_name} data to cache: {cache_path}")
    # We save ids as one array, and images as separate arrays to handle variable sizes
    np.savez(cache_path, ids=np.array(ids_list), **cache_dict)

    return dataset


class DenoisingDataset(Dataset):
    def __init__(self, data, mode="train"):
        """
        Args:
            data (list): List of dicts containing 'noisy', 'clean' (optional), 'id'.
            mode (str): 'train', 'val', or 'test'.
        """
        self.data = data
        self.mode = mode
        self.patch_size = Config.PATCH_SIZE

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Convert numpy to torch tensor
        noisy = torch.from_numpy(item["noisy"])

        # Handle Test Mode (No clean target)
        if self.mode == "test":
            return noisy, item["id"]

        clean = torch.from_numpy(item["clean"])

        # Validation Mode (Full images, no augmentation)
        if self.mode == "val":
            return noisy, clean

        # Training Mode (Augmentation)
        if self.mode == "train":
            # 1. Random Crop
            # Pad if image is smaller than patch size
            _, h, w = noisy.shape
            pad_h = max(0, self.patch_size - h)
            pad_w = max(0, self.patch_size - w)

            if pad_h > 0 or pad_w > 0:
                # Pad format: (left, right, top, bottom)
                padding = (0, pad_w, 0, pad_h)
                noisy = F.pad(noisy, padding, mode="reflect")
                clean = F.pad(clean, padding, mode="reflect")
                # Update dimensions
                _, h, w = noisy.shape

            # Get random crop coordinates
            top = random.randint(0, h - self.patch_size)
            left = random.randint(0, w - self.patch_size)

            noisy = noisy[:, top : top + self.patch_size, left : left + self.patch_size]
            clean = clean[:, top : top + self.patch_size, left : left + self.patch_size]

            # 2. Random Horizontal Flip
            if random.random() > 0.5:
                noisy = F.hflip(noisy)
                clean = F.hflip(clean)

            # 3. Random Vertical Flip
            if random.random() > 0.5:
                noisy = F.vflip(noisy)
                clean = F.vflip(clean)

            # 4. Random 90-degree Rotation
            k = random.randint(0, 3)
            if k > 0:
                noisy = torch.rot90(noisy, k, dims=[-2, -1])
                clean = torch.rot90(clean, k, dims=[-2, -1])

            return noisy, clean


def worker_init_fn(worker_id):
    """
    Sets random seeds for data loader workers to ensure reproducibility.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataloaders(load_cached_data=True, batch_size=None, num_workers=None):
    """
    Creates DataLoaders for train, val, and test sets.

    Args:
        load_cached_data (bool): Whether to use cached .npz files.
        batch_size (int, optional): Override Config.BATCH_SIZE.
        num_workers (int, optional): Override Config.NUM_WORKERS.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Use config defaults if not provided
    bs = batch_size if batch_size is not None else Config.BATCH_SIZE
    nw = num_workers if num_workers is not None else Config.NUM_WORKERS

    # Load Data
    train_data = _load_and_cache_data(Config.TRAIN_CSV, "train_cache", load_cached_data)
    val_data = _load_and_cache_data(Config.VAL_CSV, "val_cache", load_cached_data)
    test_data = _load_and_cache_data(Config.TEST_CSV, "test_cache", load_cached_data)

    # Debugging: Limit samples if Config.MAX_SAMPLES is set
    if Config.MAX_SAMPLES is not None:
        train_data = train_data[: Config.MAX_SAMPLES]
        val_data = val_data[: Config.MAX_SAMPLES]
        # We usually don't limit test data unless strictly debugging the pipeline flow

    # Create Datasets
    train_dataset = DenoisingDataset(train_data, mode="train")
    val_dataset = DenoisingDataset(val_data, mode="val")
    test_dataset = DenoisingDataset(test_data, mode="test")

    # Create DataLoaders
    # Using a generator with a fixed seed for the main process shuffling
    g = torch.Generator()
    g.manual_seed(Config.SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=bs,
        shuffle=True,
        num_workers=nw,
        worker_init_fn=worker_init_fn,
        generator=g,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # Validation images are full size, must be batch 1
        shuffle=False,
        num_workers=nw,
        worker_init_fn=worker_init_fn,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Test images are full size
        shuffle=False,
        num_workers=nw,
        worker_init_fn=worker_init_fn,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )

    return train_loader, val_loader, test_loader
