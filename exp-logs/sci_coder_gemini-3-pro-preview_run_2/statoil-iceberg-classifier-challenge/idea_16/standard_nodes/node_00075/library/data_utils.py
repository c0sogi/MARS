import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across numpy and torch.
    """
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class IcebergDataset(Dataset):
    def __init__(self, images, angles, labels=None, transform=False):
        """
        PyTorch Dataset for Iceberg Detection.

        Args:
            images (np.ndarray): Shape (N, 3, 75, 75), normalized to [0, 1].
            angles (np.ndarray): Shape (N,).
            labels (np.ndarray, optional): Shape (N,).
            transform (bool): Whether to apply augmentation.
        """
        self.images = images
        self.angles = angles
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 1. Retrieve 3-channel normalized image
        img_3ch = self.images[idx]  # (3, 75, 75)

        # 2. Construct Complementary Channels (1 - X)
        # Since images are [0, 1], this inverts the signal (shadows -> peaks)
        complement = 1.0 - img_3ch

        # 3. Stack to form 6-channel input
        # Channels: [HH, HV, Avg, 1-HH, 1-HV, 1-Avg]
        img_6ch = np.concatenate([img_3ch, complement], axis=0)  # (6, 75, 75)

        # 4. Convert to Tensor
        tensor = torch.from_numpy(img_6ch).float()

        # 5. Apply Augmentation (if requested)
        if self.transform:
            tensor = augment_tensor(tensor)

        # 6. Prepare Metadata
        angle = torch.tensor(self.angles[idx], dtype=torch.float32)

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return tensor, angle, label
        else:
            return tensor, angle


def augment_tensor(img_tensor):
    """
    Applies random 90-degree rotations and horizontal flips.
    Args:
        img_tensor (torch.Tensor): Shape (C, H, W)
    Returns:
        torch.Tensor: Augmented tensor
    """
    # Random Horizontal Flip (Flip width dimension, dim=2)
    if torch.rand(1) < 0.5:
        img_tensor = torch.flip(img_tensor, [2])

    # Random Rotation (0, 90, 180, 270 degrees)
    # k is number of times to rotate by 90 degrees
    k = int(torch.randint(0, 4, (1,)).item())
    if k > 0:
        img_tensor = torch.rot90(img_tensor, k, [1, 2])

    return img_tensor


def load_and_process_json(json_path):
    """
    Loads JSON data and converts bands to numpy arrays.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"File not found: {json_path}")

    df = pd.read_json(json_path)

    # Convert lists to numpy arrays
    # Stack creates (N, 5625) -> Reshape to (N, 75, 75)
    band_1 = np.stack(df["band_1"].values).reshape(-1, 75, 75)
    band_2 = np.stack(df["band_2"].values).reshape(-1, 75, 75)

    # Compute Average Band
    band_avg = (band_1 + band_2) / 2.0

    # Stack into (N, 3, 75, 75)
    images = np.stack([band_1, band_2, band_avg], axis=1)

    ids = df["id"].values

    # Process Incidence Angles (coerce 'na' to NaN)
    angles = pd.to_numeric(df["inc_angle"], errors="coerce").values

    labels = None
    if "is_iceberg" in df.columns:
        labels = df["is_iceberg"].values

    return images, ids, angles, labels


def compute_global_stats(images):
    """
    Computes global min and max for each channel.
    Args:
        images (np.ndarray): (N, 3, H, W)
    Returns:
        min_vals (np.ndarray): (1, 3, 1, 1)
        max_vals (np.ndarray): (1, 3, 1, 1)
    """
    # Aggregate over samples (0), height (2), width (3)
    min_vals = np.min(images, axis=(0, 2, 3)).reshape(1, 3, 1, 1)
    max_vals = np.max(images, axis=(0, 2, 3)).reshape(1, 3, 1, 1)
    return min_vals, max_vals


def load_data(config, load_cached_data=True):
    """
    Main function to load, process, and split data.
    Handles caching and debug mode.
    Returns dictionaries for train, val, and test sets.
    """
    set_seed(config.SEED)

    # Determine cache file based on debug mode
    cache_filename = (
        "processed_data_debug.npz" if config.DEBUG else "processed_data.npz"
    )
    cache_path = os.path.join(config.CACHE_DIR, cache_filename)

    # 1. Try Loading from Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        try:
            data = np.load(cache_path, allow_pickle=True)
            train_set = {
                "images": data["train_images"],
                "angles": data["train_angles"],
                "labels": data["train_labels"],
                "ids": data["train_ids"],
            }
            val_set = {
                "images": data["val_images"],
                "angles": data["val_angles"],
                "labels": data["val_labels"],
                "ids": data["val_ids"],
            }
            test_set = {
                "images": data["test_images"],
                "angles": data["test_angles"],
                "ids": data["test_ids"],
            }
            return train_set, val_set, test_set
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    # 2. Process from Scratch
    print("Processing data from scratch...")

    # Load Raw Data
    print(f"Loading {config.TRAIN_JSON}...")
    train_imgs_raw, train_ids_raw, train_angles_raw, train_labels_raw = (
        load_and_process_json(config.TRAIN_JSON)
    )
    print(f"Loading {config.TEST_JSON}...")
    test_imgs_raw, test_ids_raw, test_angles_raw, _ = load_and_process_json(
        config.TEST_JSON
    )

    # Load Metadata Splits
    train_meta = pd.read_csv(config.TRAIN_META_PATH)
    val_meta = pd.read_csv(config.VAL_META_PATH)
    test_meta = pd.read_csv(config.TEST_META_PATH)

    # Map IDs to Indices for fast lookup
    train_id_map = {id_: i for i, id_ in enumerate(train_ids_raw)}
    test_id_map = {id_: i for i, id_ in enumerate(test_ids_raw)}

    # Helper to extract subset based on metadata IDs
    def extract_subset(meta_df, id_map, imgs, angles, labels=None):
        # Find indices in the raw array corresponding to the IDs in the metadata split
        indices = [id_map[id_] for id_ in meta_df["id"].values]

        # Apply Debug Slicing if enabled
        if config.DEBUG:
            indices = indices[: config.DEBUG_SUBSET_SIZE]

        subset_imgs = imgs[indices]
        subset_angles = angles[indices]
        subset_ids = meta_df["id"].values[: len(indices)]

        subset_labels = None
        if labels is not None:
            subset_labels = labels[indices]

        return subset_imgs, subset_angles, subset_labels, subset_ids

    # Create Splits
    X_train, ang_train, y_train, ids_train = extract_subset(
        train_meta, train_id_map, train_imgs_raw, train_angles_raw, train_labels_raw
    )
    X_val, ang_val, y_val, ids_val = extract_subset(
        val_meta, train_id_map, train_imgs_raw, train_angles_raw, train_labels_raw
    )
    X_test, ang_test, _, ids_test = extract_subset(
        test_meta, test_id_map, test_imgs_raw, test_angles_raw
    )

    # 3. Compute Stats & Normalize
    # Stats computed ONLY on training split to prevent leakage
    min_vals, max_vals = compute_global_stats(X_train)

    def normalize(X, mn, mx):
        # Scale to [0, 1]
        return np.clip((X - mn) / (mx - mn + 1e-8), 0.0, 1.0)

    X_train_norm = normalize(X_train, min_vals, max_vals)
    X_val_norm = normalize(X_val, min_vals, max_vals)
    X_test_norm = normalize(X_test, min_vals, max_vals)

    # 4. Impute Missing Angles
    # Compute mean from training split (ignoring NaNs)
    angle_mean = np.nanmean(ang_train)

    def fill_na(angles, fill_value):
        a = angles.copy()
        a[np.isnan(a)] = fill_value
        return a

    ang_train_filled = fill_na(ang_train, angle_mean)
    ang_val_filled = fill_na(ang_val, angle_mean)
    ang_test_filled = fill_na(ang_test, angle_mean)

    # 5. Save to Cache
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    np.savez(
        cache_path,
        train_images=X_train_norm,
        train_angles=ang_train_filled,
        train_labels=y_train,
        train_ids=ids_train,
        val_images=X_val_norm,
        val_angles=ang_val_filled,
        val_labels=y_val,
        val_ids=ids_val,
        test_images=X_test_norm,
        test_angles=ang_test_filled,
        test_ids=ids_test,
    )
    print(f"Data processed and saved to {cache_path}")

    return (
        {
            "images": X_train_norm,
            "angles": ang_train_filled,
            "labels": y_train,
            "ids": ids_train,
        },
        {
            "images": X_val_norm,
            "angles": ang_val_filled,
            "labels": y_val,
            "ids": ids_val,
        },
        {"images": X_test_norm, "angles": ang_test_filled, "ids": ids_test},
    )
