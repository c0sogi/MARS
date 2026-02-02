import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from library.config import Config


def compute_global_stats(images):
    """
    Compute global min and max per channel for the given images.
    images: (N, 3, H, W)
    Returns: min_vals (3,), max_vals (3,)
    """
    # Compute min/max across batch (0) and spatial dimensions (2, 3)
    # Resulting shape: (3,)
    min_vals = np.min(images, axis=(0, 2, 3))
    max_vals = np.max(images, axis=(0, 2, 3))
    return min_vals, max_vals


def process_data(load_cached_data=True):
    """
    Loads, processes, and caches the data.
    Returns:
        train_data (dict): keys 'images', 'angles', 'labels', 'ids'
        test_data (dict): keys 'images', 'angles', 'ids'
        global_stats (dict): keys 'min', 'max'
    """
    cache_path = Config.PROCESSED_DATA_PATH

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            data = np.load(cache_path, allow_pickle=True)
            train_data = {
                "images": data["train_images"],
                "angles": data["train_angles"],
                "labels": data["train_labels"],
                "ids": data["train_ids"],
            }
            test_data = {
                "images": data["test_images"],
                "angles": data["test_angles"],
                "ids": data["test_ids"],
            }
            global_stats = {"min": data["global_min"], "max": data["global_max"]}
            return train_data, test_data, global_stats
        except Exception:
            # If load fails, fall back to processing
            pass

    # 2. Process from scratch
    # Load Raw JSONs
    with open(Config.TRAIN_JSON, "r") as f:
        train_json = json.load(f)
    with open(Config.TEST_JSON, "r") as f:
        test_json = json.load(f)

    # --- Process Training Data ---
    df_train = pd.DataFrame(train_json)

    # Process Images: Band 1, Band 2, Mean
    # Reshape from list to (75, 75)
    b1_train = np.array(
        [
            np.array(band).astype(np.float32).reshape(75, 75)
            for band in df_train["band_1"]
        ]
    )
    b2_train = np.array(
        [
            np.array(band).astype(np.float32).reshape(75, 75)
            for band in df_train["band_2"]
        ]
    )
    b3_train = (b1_train + b2_train) / 2.0

    # Stack channels: (N, 3, 75, 75)
    train_images = np.stack([b1_train, b2_train, b3_train], axis=1)

    # Process Angles: Handle 'na' by imputing with mean
    df_train["inc_angle"] = pd.to_numeric(df_train["inc_angle"], errors="coerce")
    angle_mean = df_train["inc_angle"].mean()
    df_train["inc_angle"] = df_train["inc_angle"].fillna(angle_mean)
    train_angles = df_train["inc_angle"].values.astype(np.float32)

    # Process Labels and IDs
    train_labels = df_train["is_iceberg"].values.astype(np.float32)
    train_ids = df_train["id"].values

    # Compute Global Stats (on full training set)
    global_min, global_max = compute_global_stats(train_images)

    # --- Process Test Data ---
    df_test = pd.DataFrame(test_json)

    b1_test = np.array(
        [
            np.array(band).astype(np.float32).reshape(75, 75)
            for band in df_test["band_1"]
        ]
    )
    b2_test = np.array(
        [
            np.array(band).astype(np.float32).reshape(75, 75)
            for band in df_test["band_2"]
        ]
    )
    b3_test = (b1_test + b2_test) / 2.0

    test_images = np.stack([b1_test, b2_test, b3_test], axis=1)

    # Process Angles: Use mean from training set
    df_test["inc_angle"] = pd.to_numeric(df_test["inc_angle"], errors="coerce")
    df_test["inc_angle"] = df_test["inc_angle"].fillna(angle_mean)
    test_angles = df_test["inc_angle"].values.astype(np.float32)
    test_ids = df_test["id"].values

    # 3. Save to Cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(
        cache_path,
        train_images=train_images,
        train_angles=train_angles,
        train_labels=train_labels,
        train_ids=train_ids,
        test_images=test_images,
        test_angles=test_angles,
        test_ids=test_ids,
        global_min=global_min,
        global_max=global_max,
    )

    train_data = {
        "images": train_images,
        "angles": train_angles,
        "labels": train_labels,
        "ids": train_ids,
    }
    test_data = {"images": test_images, "angles": test_angles, "ids": test_ids}
    global_stats = {"min": global_min, "max": global_max}

    return train_data, test_data, global_stats


class IcebergDataset(Dataset):
    def __init__(
        self, images, angles, labels=None, ids=None, transform=False, global_stats=None
    ):
        self.images = images
        self.angles = angles
        self.labels = labels
        self.ids = ids
        self.transform = transform
        self.global_stats = global_stats

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Retrieve data
        img = self.images[idx].copy()  # (3, 75, 75)
        angle = self.angles[idx]

        # 1. Normalization (Global Min-Max)
        if self.global_stats is not None:
            g_min = self.global_stats["min"][:, None, None]  # Broadcast to (3, 1, 1)
            g_max = self.global_stats["max"][:, None, None]

            # Avoid division by zero
            denom = g_max - g_min
            denom[denom == 0] = 1.0

            # Scale to roughly [0, 1], but allow outliers (No Hard Clipping)
            img = (img - g_min) / denom

        # 2. Augmentation
        if self.transform:
            # Random Horizontal Flip
            if Config.AUG_HFLIP and np.random.rand() < 0.5:
                # Axis 2 is Width in (C, H, W)
                img = np.flip(img, axis=2)

            # Random Rotation (0, 90, 180, 270)
            if Config.AUG_ROTATION:
                k = np.random.randint(0, 4)
                if k > 0:
                    # Rotate in spatial plane (axes 1 and 2)
                    img = np.rot90(img, k, axes=(1, 2))

        # 3. Convert to Tensor
        # Copy is necessary after numpy flip/rot operations to ensure positive strides for PyTorch
        img_tensor = torch.from_numpy(img.copy()).float()
        angle_tensor = torch.tensor(angle, dtype=torch.float32)

        if self.labels is not None:
            label_tensor = torch.tensor(self.labels[idx], dtype=torch.float32)
            return img_tensor, angle_tensor, label_tensor, self.ids[idx]
        else:
            return img_tensor, angle_tensor, self.ids[idx]


def get_dataloaders(fold_index=None, load_cached_data=True, debug=Config.DEBUG):
    """
    Creates DataLoaders for training and validation.

    Args:
        fold_index (int, optional): If provided, performs K-Fold split.
                                    If None, uses fixed split from metadata.
        load_cached_data (bool): Whether to use cached .npz data.
        debug (bool): If True, subsets data for debugging.
    """
    train_data, _, global_stats = process_data(load_cached_data)

    # Define indices for split
    if fold_index is not None:
        # Stratified K-Fold on Full Dataset
        skf = StratifiedKFold(
            n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
        )
        folds = list(skf.split(train_data["images"], train_data["labels"]))
        train_idx, val_idx = folds[fold_index]
    else:
        # Fixed Split from Metadata
        df_train_meta = pd.read_csv(Config.TRAIN_META)
        df_val_meta = pd.read_csv(Config.VAL_META)

        # Map IDs to indices in the loaded arrays
        id_to_idx = {uid: i for i, uid in enumerate(train_data["ids"])}

        train_idx = np.array(
            [id_to_idx[uid] for uid in df_train_meta["id"].values if uid in id_to_idx]
        )
        val_idx = np.array(
            [id_to_idx[uid] for uid in df_val_meta["id"].values if uid in id_to_idx]
        )

    # Debug Subsetting
    if debug:
        train_idx = train_idx[: Config.DEBUG_SIZE]
        val_idx = val_idx[: Config.DEBUG_SIZE]

    # Create Datasets
    train_dataset = IcebergDataset(
        images=train_data["images"][train_idx],
        angles=train_data["angles"][train_idx],
        labels=train_data["labels"][train_idx],
        ids=train_data["ids"][train_idx],
        transform=True,  # Apply Augmentation
        global_stats=global_stats,
    )

    val_dataset = IcebergDataset(
        images=train_data["images"][val_idx],
        angles=train_data["angles"][val_idx],
        labels=train_data["labels"][val_idx],
        ids=train_data["ids"][val_idx],
        transform=False,  # No Augmentation
        global_stats=global_stats,
    )

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    return train_loader, val_loader


def get_test_dataloader(load_cached_data=True):
    """
    Creates DataLoader for the test set.
    """
    _, test_data, global_stats = process_data(load_cached_data)

    test_dataset = IcebergDataset(
        images=test_data["images"],
        angles=test_data["angles"],
        ids=test_data["ids"],
        transform=False,
        global_stats=global_stats,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    return test_loader
