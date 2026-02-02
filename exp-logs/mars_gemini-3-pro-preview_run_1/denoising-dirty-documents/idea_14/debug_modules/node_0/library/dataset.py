import os
import cv2
import torch
import numpy as np
import pandas as pd
import random
from torch.utils.data import Dataset, DataLoader
from library import config, utils


class DenoisingDataset(Dataset):
    def __init__(
        self,
        metadata_df,
        root_dir,
        mode="train",
        patch_size=config.PATCH_SIZE,
        cache_path=None,
        load_cached_data=True,
    ):
        """
        Args:
            metadata_df (pd.DataFrame): DataFrame containing image IDs and paths.
            root_dir (str): Root directory for image files.
            mode (str): 'train', 'val', or 'test'.
            patch_size (int): Size of random crops for training.
            cache_path (str): Path to save/load the cached dataset (.npz).
            load_cached_data (bool): Whether to attempt loading from cache.
        """
        self.mode = mode
        self.patch_size = patch_size
        self.root_dir = root_dir
        self.data = []

        # Caching Logic
        data_loaded = False
        if load_cached_data and cache_path and os.path.exists(cache_path):
            try:
                # Load from .npz
                # We expect keys: ids, noisy_0, clean_0, noisy_1, ...
                loaded = np.load(cache_path)
                ids = loaded["ids"]

                # Reconstruct list of dictionaries
                for i, img_id in enumerate(ids):
                    sample = {"id": str(img_id)}
                    sample["noisy"] = loaded[f"noisy_{i}"]

                    # Check for clean image
                    clean_key = f"clean_{i}"
                    if clean_key in loaded:
                        sample["clean"] = loaded[clean_key]

                    self.data.append(sample)

                data_loaded = True
            except Exception as e:
                print(
                    f"Failed to load cache from {cache_path}: {e}. processing from scratch."
                )

        if not data_loaded:
            self._process_and_cache(metadata_df, cache_path)

    def _process_and_cache(self, df, cache_path):
        """Reads images from disk, normalizes them, and saves to cache."""
        self.data = []

        for _, row in df.iterrows():
            img_id = str(row["id"])
            noisy_path = os.path.join(self.root_dir, row["noisy_image_path"])

            # Load noisy image (Grayscale)
            # Using IMREAD_GRAYSCALE ensures 2D array (H, W)
            img_n = cv2.imread(noisy_path, cv2.IMREAD_GRAYSCALE)
            if img_n is None:
                continue

            # Normalize to [0, 1] float32
            img_n = img_n.astype(np.float32) / 255.0

            sample = {"id": img_id, "noisy": img_n}

            # Load clean image if available
            if "clean_image_path" in row and pd.notna(row["clean_image_path"]):
                clean_path = os.path.join(self.root_dir, row["clean_image_path"])
                img_c = cv2.imread(clean_path, cv2.IMREAD_GRAYSCALE)
                if img_c is not None:
                    img_c = img_c.astype(np.float32) / 255.0
                    sample["clean"] = img_c

            self.data.append(sample)

        # Save to cache if path provided
        if cache_path:
            self._save_cache(cache_path)

    def _save_cache(self, cache_path):
        """Saves the in-memory data to an .npz file without object pickling."""
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)

        save_dict = {}
        ids = []

        for i, sample in enumerate(self.data):
            ids.append(sample["id"])
            # Save each array as a separate key to use efficient binary storage
            save_dict[f"noisy_{i}"] = sample["noisy"]
            if "clean" in sample:
                save_dict[f"clean_{i}"] = sample["clean"]

        save_dict["ids"] = np.array(ids)

        # Use savez_compressed for space efficiency
        np.savez_compressed(cache_path, **save_dict)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        img_n = sample["noisy"]
        img_c = sample.get("clean", None)
        img_id = sample["id"]

        if self.mode == "train":
            # --- Training: Random Crop & Augmentation ---
            h, w = img_n.shape

            # 1. Random Crop
            # Ensure image is large enough, otherwise take full or resize (unlikely given dataset stats)
            if h >= self.patch_size and w >= self.patch_size:
                y = np.random.randint(0, h - self.patch_size + 1)
                x = np.random.randint(0, w - self.patch_size + 1)

                img_n = img_n[y : y + self.patch_size, x : x + self.patch_size]
                if img_c is not None:
                    img_c = img_c[y : y + self.patch_size, x : x + self.patch_size]

            # 2. Augmentation
            # Random Horizontal Flip
            if np.random.rand() > 0.5:
                img_n = np.flip(img_n, axis=1).copy()
                if img_c is not None:
                    img_c = np.flip(img_c, axis=1).copy()

            # Random Vertical Flip
            if np.random.rand() > 0.5:
                img_n = np.flip(img_n, axis=0).copy()
                if img_c is not None:
                    img_c = np.flip(img_c, axis=0).copy()

            # Random 90-degree Rotation
            k = np.random.randint(0, 4)
            if k > 0:
                img_n = np.rot90(img_n, k).copy()
                if img_c is not None:
                    img_c = np.rot90(img_c, k).copy()

            # 3. To Tensor (Add Channel Dim: H,W -> 1,H,W)
            img_n_t = torch.from_numpy(img_n).unsqueeze(0).float()

            if img_c is not None:
                img_c_t = torch.from_numpy(img_c).unsqueeze(0).float()
                return img_n_t, img_c_t

            return img_n_t

        else:
            # --- Validation/Test: Full Image with Padding ---
            # Pad to multiple of 8 for 3-level UNet
            img_n_padded, pads = utils.pad_image_to_multiple(img_n, multiple=8)

            # To Tensor
            img_n_t = torch.from_numpy(img_n_padded).unsqueeze(0).float()

            meta = {
                "id": img_id,
                "pads": torch.tensor(pads),  # (top, bottom, left, right)
                "orig_h": img_n.shape[0],
                "orig_w": img_n.shape[1],
            }

            if img_c is not None:
                # Also pad ground truth for consistent loss calculation
                img_c_padded, _ = utils.pad_image_to_multiple(img_c, multiple=8)
                img_c_t = torch.from_numpy(img_c_padded).unsqueeze(0).float()
                return img_n_t, img_c_t, meta

            return img_n_t, meta


def worker_init_fn(worker_id):
    """
    Sets seeds for DataLoader workers to ensure deterministic augmentations.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataloaders(
    batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, load_cached=True
):
    """
    Factory function to create DataLoaders for train, val, and test sets.
    """
    # Load Metadata DataFrames
    train_df = pd.read_csv(config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(config.VAL_METADATA_PATH)
    test_df = pd.read_csv(config.TEST_METADATA_PATH)

    # Define Cache Paths
    train_cache = config.get_cache_path("train")
    val_cache = config.get_cache_path("val")
    test_cache = config.get_cache_path("test")

    # Instantiate Datasets
    train_ds = DenoisingDataset(
        train_df,
        config.INPUT_DIR,
        mode="train",
        patch_size=config.PATCH_SIZE,
        cache_path=train_cache,
        load_cached_data=load_cached,
    )

    val_ds = DenoisingDataset(
        val_df,
        config.INPUT_DIR,
        mode="val",
        cache_path=val_cache,
        load_cached_data=load_cached,
    )

    test_ds = DenoisingDataset(
        test_df,
        config.INPUT_DIR,
        mode="test",
        cache_path=test_cache,
        load_cached_data=load_cached,
    )

    # Create DataLoaders
    # Train loader uses the defined batch size and shuffling
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        pin_memory=True,
    )

    # Val and Test loaders must use batch_size=1 because images have variable sizes
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader
