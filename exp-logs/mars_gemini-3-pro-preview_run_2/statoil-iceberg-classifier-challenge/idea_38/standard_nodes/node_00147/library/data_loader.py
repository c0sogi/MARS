import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config


def process_and_cache_data(load_cached_data=True):
    """
    Loads raw JSON data, processes it into numpy arrays (images, angles, labels),
    and caches it to disk. If cache exists and load_cached_data is True, loads from disk.

    Returns:
        dict: Dictionary containing processed arrays:
              'X_train_full', 'y_train_full', 'angles_train_full', 'ids_train_full',
              'X_test', 'angles_test', 'ids_test'
    """
    cache_path = Config.CACHE_PATH

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        try:
            data = np.load(cache_path, allow_pickle=True)
            # Cite debug_lesson_1: Validate Cache Schema on Load
            required_keys = [
                "X_train_full",
                "y_train_full",
                "angles_train_full",
                "ids_train_full",
                "X_test",
                "angles_test",
                "ids_test",
            ]
            if all(key in data for key in required_keys):
                return data
            else:
                print(
                    f"Cache invalid (missing keys). Found: {list(data.keys())}. Regenerating..."
                )
        except Exception as e:
            print(f"Error loading cache: {e}. Regenerating...")

    print("Processing raw data from scratch...")

    # --- Helper to process a single JSON file ---
    def load_json_file(filepath, is_train=True):
        with open(filepath, "r") as f:
            data = json.load(f)

        df = pd.DataFrame(data)

        # Process Images
        # Band 1 and Band 2 are lists of floats. Reshape to (N, 75, 75)
        band_1 = np.stack([np.array(b).reshape(75, 75) for b in df["band_1"]])
        band_2 = np.stack([np.array(b).reshape(75, 75) for b in df["band_2"]])

        # Construct 3rd Channel: Mean of Band 1 and Band 2
        band_3 = (band_1 + band_2) / 2.0

        # Stack into (N, 3, 75, 75) - Channel First for PyTorch
        # Note: We stack as (N, 3, 75, 75) here directly
        X = np.stack([band_1, band_2, band_3], axis=1).astype(np.float32)

        # Process Incidence Angles
        # Replace 'na' with NaN and convert to float
        angles = pd.to_numeric(df["inc_angle"], errors="coerce").values.astype(
            np.float32
        )

        ids = df["id"].values

        if is_train:
            y = df["is_iceberg"].values.astype(np.float32)
            return X, angles, y, ids
        else:
            return X, angles, ids

    # --- Load Train and Test ---
    print(f"Loading {Config.TRAIN_JSON}...")
    X_train_full, angles_train_full, y_train_full, ids_train_full = load_json_file(
        Config.TRAIN_JSON, is_train=True
    )

    print(f"Loading {Config.TEST_JSON}...")
    X_test, angles_test, ids_test = load_json_file(Config.TEST_JSON, is_train=False)

    # --- Impute Missing Incidence Angles ---
    # Strategy: Fill NaNs with the mean of the valid training angles
    train_angle_mean = np.nanmean(angles_train_full)

    # Identify indices where angle is NaN
    train_nan_mask = np.isnan(angles_train_full)
    test_nan_mask = np.isnan(angles_test)

    angles_train_full[train_nan_mask] = train_angle_mean
    angles_test[test_nan_mask] = train_angle_mean  # Use train mean for test as well

    print(
        f"Imputed {np.sum(train_nan_mask)} missing angles in Train with mean {train_angle_mean:.4f}"
    )
    print(f"Imputed {np.sum(test_nan_mask)} missing angles in Test")

    # --- Save to Cache ---
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez(
        cache_path,
        X_train_full=X_train_full,
        y_train_full=y_train_full,
        angles_train_full=angles_train_full,
        ids_train_full=ids_train_full,
        X_test=X_test,
        angles_test=angles_test,
        ids_test=ids_test,
    )
    print(f"Data cached to {cache_path}")

    # Reload to return consistent dict-like object (NpzFile)
    return np.load(cache_path, allow_pickle=True)


def get_global_stats(X_train):
    """
    Computes global min and max per channel for normalization.
    X_train shape: (N, 3, 75, 75)
    """
    # Reshape to (C, N*H*W) to compute stats per channel
    # Channel 0
    min_c0 = X_train[:, 0, :, :].min()
    max_c0 = X_train[:, 0, :, :].max()

    # Channel 1
    min_c1 = X_train[:, 1, :, :].min()
    max_c1 = X_train[:, 1, :, :].max()

    # Channel 2
    min_c2 = X_train[:, 2, :, :].min()
    max_c2 = X_train[:, 2, :, :].max()

    stats = {
        "min": np.array([min_c0, min_c1, min_c2], dtype=np.float32).reshape(3, 1, 1),
        "max": np.array([max_c0, max_c1, max_c2], dtype=np.float32).reshape(3, 1, 1),
    }
    return stats


class IcebergDataset(Dataset):
    def __init__(self, X, angles, labels=None, ids=None, transform=False, stats=None):
        """
        Args:
            X (np.ndarray): Images (N, 3, 75, 75)
            angles (np.ndarray): Incidence angles (N,)
            labels (np.ndarray, optional): Labels (N,)
            ids (np.ndarray, optional): IDs (N,)
            transform (bool): Whether to apply augmentation
            stats (dict): Global min/max stats for normalization
        """
        self.X = X
        self.angles = angles
        self.labels = labels
        self.ids = ids
        self.transform = transform
        self.stats = stats

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Load data
        img = self.X[idx].copy()  # (3, 75, 75)
        angle = self.angles[idx]

        # 1. Normalization
        # (x - min) / (max - min)
        if self.stats:
            img = (img - self.stats["min"]) / (self.stats["max"] - self.stats["min"])
            # No clipping as per instructions

        # 2. Augmentation (Training only)
        if self.transform:
            # Convert to (H, W, C) for easier geometric manipulation if using cv2,
            # but here we can use numpy/torch on (C, H, W)

            # Random Rotation: 0, 90, 180, 270
            k = np.random.randint(0, 4)
            img = np.rot90(img, k, axes=(1, 2)).copy()

            # Random Horizontal Flip
            if Config.AUG_HFLIP and np.random.rand() < 0.5:
                img = np.flip(img, axis=2).copy()  # Flip W dimension

            # Vertical Flip is False in Config, so skipping

        # Convert to Tensor
        img_tensor = torch.from_numpy(img).float()
        angle_tensor = torch.tensor(angle, dtype=torch.float32)

        if self.labels is not None:
            label_tensor = torch.tensor(self.labels[idx], dtype=torch.float32)
            return img_tensor, angle_tensor, label_tensor
        else:
            # For test set, return ID as well to track predictions
            id_val = self.ids[idx]
            return img_tensor, angle_tensor, id_val


def get_dataloaders(debug=False):
    """
    Creates DataLoaders for Train, Validation, and Test sets.
    Uses metadata files to split the full training data.
    """
    # 1. Load Processed Data
    data = process_and_cache_data(load_cached_data=True)
    X_train_full = data["X_train_full"]
    y_train_full = data["y_train_full"]
    angles_train_full = data["angles_train_full"]
    ids_train_full = data["ids_train_full"]

    X_test = data["X_test"]
    angles_test = data["angles_test"]
    ids_test = data["ids_test"]

    # 2. Compute Global Stats from FULL training data
    # This ensures consistency regardless of fold splits
    stats = get_global_stats(X_train_full)

    # 3. Load Metadata for Splits
    df_train_meta = pd.read_csv(Config.TRAIN_META)
    df_val_meta = pd.read_csv(Config.VAL_META)

    # Map IDs to indices in the full numpy arrays
    # Create a lookup dictionary: id -> index
    id_to_idx = {id_val: i for i, id_val in enumerate(ids_train_full)}

    train_indices = [
        id_to_idx[uid] for uid in df_train_meta["id"].values if uid in id_to_idx
    ]
    val_indices = [
        id_to_idx[uid] for uid in df_val_meta["id"].values if uid in id_to_idx
    ]

    # 4. Subset Data
    X_train = X_train_full[train_indices]
    y_train = y_train_full[train_indices]
    angles_train = angles_train_full[train_indices]
    ids_train = ids_train_full[train_indices]

    X_val = X_train_full[val_indices]
    y_val = y_train_full[val_indices]
    angles_val = angles_train_full[val_indices]
    ids_val = ids_train_full[val_indices]

    # Debug Mode
    if debug:
        print(f"Debug mode: trimming datasets to {Config.DEBUG_SIZE} samples.")
        X_train = X_train[: Config.DEBUG_SIZE]
        y_train = y_train[: Config.DEBUG_SIZE]
        angles_train = angles_train[: Config.DEBUG_SIZE]

        X_val = X_val[: Config.DEBUG_SIZE]
        y_val = y_val[: Config.DEBUG_SIZE]
        angles_val = angles_val[: Config.DEBUG_SIZE]

        X_test = X_test[: Config.DEBUG_SIZE]
        angles_test = angles_test[: Config.DEBUG_SIZE]
        ids_test = ids_test[: Config.DEBUG_SIZE]

    # 5. Create Datasets
    train_dataset = IcebergDataset(
        X_train, angles_train, y_train, ids_train, transform=True, stats=stats
    )

    val_dataset = IcebergDataset(
        X_val, angles_val, y_val, ids_val, transform=False, stats=stats
    )

    test_dataset = IcebergDataset(
        X_test, angles_test, labels=None, ids=ids_test, transform=False, stats=stats
    )

    # 6. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
