import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold

from library.config import (
    INPUT_DIR,
    TRAIN_METADATA,
    VAL_METADATA,
    TEST_METADATA,
    WORKING_DIR,
    IMG_SIZE,
    BATCH_SIZE,
    SEED,
    NUM_WORKERS,
    N_FOLDS,
)
from library.utils import worker_init_fn


def _load_and_cache_data(metadata_paths, cache_name, load_cached_data=True):
    """
    Loads images based on metadata CSVs. Caches the result to a .npz file.
    """
    cache_path = os.path.join(WORKING_DIR, cache_name)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}...")
        try:
            data = np.load(cache_path, allow_pickle=True)
            # Reconstruct list of dicts
            dataset = []
            ids = data["ids"]
            noisy_imgs = data["noisy_imgs"]
            # Clean images might not exist for test set
            if "clean_imgs" in data:
                clean_imgs = data["clean_imgs"]
                for i, img_id in enumerate(ids):
                    dataset.append(
                        {
                            "id": str(img_id),
                            "noisy": noisy_imgs[i],
                            "clean": clean_imgs[i],
                        }
                    )
            else:
                for i, img_id in enumerate(ids):
                    dataset.append({"id": str(img_id), "noisy": noisy_imgs[i]})
            return dataset
        except Exception as e:
            print(f"Failed to load cache: {e}. Reloading from source.")

    # 2. Load from source
    print(f"Processing data from {metadata_paths}...")
    dfs = [pd.read_csv(p) for p in metadata_paths]
    full_df = pd.concat(dfs, ignore_index=True)

    ids = []
    noisy_imgs = []
    clean_imgs = []
    has_clean = "clean_image_path" in full_df.columns

    for _, row in full_df.iterrows():
        # Load Noisy
        n_path = os.path.join(INPUT_DIR, row["noisy_image_path"])
        n_img = cv2.imread(n_path, cv2.IMREAD_GRAYSCALE)
        if n_img is None:
            continue

        ids.append(row["id"])
        noisy_imgs.append(n_img)

        # Load Clean if available
        if has_clean:
            c_path = os.path.join(INPUT_DIR, row["clean_image_path"])
            c_img = cv2.imread(c_path, cv2.IMREAD_GRAYSCALE)
            clean_imgs.append(c_img)

    # 3. Save to cache
    os.makedirs(WORKING_DIR, exist_ok=True)
    save_dict = {
        "ids": np.array(ids),
        "noisy_imgs": np.array(noisy_imgs, dtype=object),
    }
    if has_clean:
        save_dict["clean_imgs"] = np.array(clean_imgs, dtype=object)

    np.savez_compressed(cache_path, **save_dict)
    print(f"Data cached to {cache_path}")

    # Return constructed list
    dataset = []
    for i, img_id in enumerate(ids):
        item = {"id": str(img_id), "noisy": noisy_imgs[i]}
        if has_clean:
            item["clean"] = clean_imgs[i]
        dataset.append(item)

    return dataset


class DenoisingDataset(Dataset):
    def __init__(self, data_list, mode="train"):
        """
        Args:
            data_list (list): List of dicts containing images.
            mode (str): 'train', 'val', or 'test'.
        """
        self.data = data_list
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Normalize to [0, 1] float32
        noisy = item["noisy"].astype(np.float32) / 255.0

        if self.mode == "test":
            # Test mode: Return noisy image and ID
            # Add channel dimension: (H, W) -> (1, H, W)
            noisy_tensor = torch.from_numpy(noisy).unsqueeze(0)
            return noisy_tensor, item["id"]

        clean = item["clean"].astype(np.float32) / 255.0

        if self.mode == "train":
            # --- Augmentation Pipeline ---
            h, w = noisy.shape
            crop_size = IMG_SIZE

            # 1. Random Crop
            # Ensure image is large enough, otherwise pad (though analysis says images are > 160)
            if h < crop_size or w < crop_size:
                pad_h = max(0, crop_size - h)
                pad_w = max(0, crop_size - w)
                noisy = np.pad(noisy, ((0, pad_h), (0, pad_w)), mode="reflect")
                clean = np.pad(clean, ((0, pad_h), (0, pad_w)), mode="reflect")
                h, w = noisy.shape

            top = np.random.randint(0, h - crop_size + 1)
            left = np.random.randint(0, w - crop_size + 1)

            noisy = noisy[top : top + crop_size, left : left + crop_size]
            clean = clean[top : top + crop_size, left : left + crop_size]

            # 2. Random Flips
            if np.random.rand() > 0.5:
                noisy = np.fliplr(noisy)
                clean = np.fliplr(clean)
            if np.random.rand() > 0.5:
                noisy = np.flipud(noisy)
                clean = np.flipud(clean)

            # 3. Random 90-degree Rotations
            k = np.random.randint(0, 4)
            if k > 0:
                noisy = np.rot90(noisy, k)
                clean = np.rot90(clean)

        # Compute Residual Target (Noise = Noisy - Clean)
        residual = noisy - clean

        # Convert to Tensor (C, H, W)
        noisy_tensor = torch.from_numpy(noisy.copy()).unsqueeze(0)
        residual_tensor = torch.from_numpy(residual.copy()).unsqueeze(0)

        return noisy_tensor, residual_tensor


def get_kfold_loaders(n_folds=N_FOLDS, load_cached_data=True):
    """
    Creates K-Fold DataLoaders for training and validation.
    Combines train and val metadata to use all available data.
    """
    # Load combined data
    full_data = _load_and_cache_data(
        [TRAIN_METADATA, VAL_METADATA], "train_val_cache.npz", load_cached_data
    )

    # Initialize KFold
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)

    loaders = []

    # Iterate folds
    # We map indices to the data list
    indices = np.arange(len(full_data))

    for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
        # Create Subsets
        train_subset = [full_data[i] for i in train_idx]
        val_subset = [full_data[i] for i in val_idx]

        # Create Datasets
        train_ds = DenoisingDataset(train_subset, mode="train")
        val_ds = DenoisingDataset(val_subset, mode="val")

        # Create Loaders
        train_loader = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            worker_init_fn=worker_init_fn,
            pin_memory=True,
        )

        # Validation loader batch_size=1 to handle variable image sizes
        val_loader = DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=NUM_WORKERS,
            worker_init_fn=worker_init_fn,
            pin_memory=True,
        )

        loaders.append((train_loader, val_loader))

    return loaders


def get_test_loader(load_cached_data=True):
    """
    Creates DataLoader for the test set.
    """
    test_data = _load_and_cache_data(
        [TEST_METADATA], "test_cache.npz", load_cached_data
    )

    test_ds = DenoisingDataset(test_data, mode="test")

    # Batch size 1 for variable sizes in inference
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        worker_init_fn=worker_init_fn,
        pin_memory=True,
    )

    return test_loader
