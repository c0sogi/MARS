import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from library.config import Config
from library.utils import calculate_global_stats


class IcebergDataset(Dataset):
    def __init__(self, X, inc_angles, labels=None, transform=None, global_stats=None):
        """
        Args:
            X (np.ndarray): Images of shape (N, 3, 75, 75).
            inc_angles (np.ndarray): Incidence angles of shape (N,).
            labels (np.ndarray, optional): Labels of shape (N,).
            transform (albumentations.Compose, optional): Augmentation pipeline.
            global_stats (dict): Dictionary containing min/max for scaling.
        """
        self.X = X
        self.inc_angles = inc_angles
        self.labels = labels
        self.transform = transform
        self.stats = global_stats

        # Pre-calculate denominators for scaling
        if self.stats:
            self.b1_min = self.stats["b1_min"]
            self.b2_min = self.stats["b2_min"]
            self.b3_min = self.stats["b3_min"]

            self.b1_range = self.stats["b1_max"] - self.stats["b1_min"]
            self.b2_range = self.stats["b2_max"] - self.stats["b2_min"]
            self.b3_range = self.stats["b3_max"] - self.stats["b3_min"]

            # Avoid division by zero
            self.b1_range = self.b1_range if self.b1_range > 1e-6 else 1.0
            self.b2_range = self.b2_range if self.b2_range > 1e-6 else 1.0
            self.b3_range = self.b3_range if self.b3_range > 1e-6 else 1.0

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Retrieve data
        # shape: (3, 75, 75)
        img = self.X[idx].copy()
        inc = self.inc_angles[idx]

        # Transpose for Albumentations: (C, H, W) -> (H, W, C)
        img = np.transpose(img, (1, 2, 0))

        # Augmentation
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented["image"]

        # Normalization (Independent Per-Channel Min-Max)
        # img is (75, 75, 3)
        if self.stats:
            img[:, :, 0] = (img[:, :, 0] - self.b1_min) / self.b1_range
            img[:, :, 1] = (img[:, :, 1] - self.b2_min) / self.b2_range
            img[:, :, 2] = (img[:, :, 2] - self.b3_min) / self.b3_range

        # Transpose back to (C, H, W) and convert to tensor
        img = np.transpose(img, (2, 0, 1))
        img_tensor = torch.from_numpy(img).float()

        # Metadata
        inc_tensor = torch.tensor([inc], dtype=torch.float32)

        if self.labels is not None:
            label_tensor = torch.tensor([self.labels[idx]], dtype=torch.float32)
            return img_tensor, inc_tensor, label_tensor
        else:
            return img_tensor, inc_tensor


def process_data(load_cached_data=True):
    """
    Loads raw JSON data, aligns with metadata CSVs, constructs 3-channel images,
    handles missing incidence angles, and caches the result.
    """
    cache_path = Config.PROCESSED_DATA_CACHE
    if Config.DEBUG:
        cache_path = cache_path.replace(".npz", "_debug.npz")

    # 1. Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            print(f"Loading processed data from {cache_path}...")
            data = np.load(cache_path, allow_pickle=True)
            # Convert to dict
            return {k: data[k] for k in data.files}
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    print("Processing data from scratch...")

    # 2. Load Raw Data
    with open(Config.TRAIN_JSON, "r") as f:
        train_json = json.load(f)
    with open(Config.TEST_JSON, "r") as f:
        test_json = json.load(f)

    # Map ID to raw item
    raw_map = {item["id"]: item for item in train_json + test_json}

    # 3. Load Metadata
    df_train = pd.read_csv(Config.TRAIN_META_PATH)
    df_val = pd.read_csv(Config.VAL_META_PATH)
    df_test = pd.read_csv(Config.TEST_META_PATH)

    # 4. Compute Imputation Value for Inc Angle
    # Use mean of training set (excluding NaNs)
    train_inc_mean = df_train["inc_angle"].mean()
    if pd.isna(train_inc_mean):
        train_inc_mean = 0.0

    def build_arrays(df, is_test=False):
        ids = df["id"].values

        # Debug subsetting
        if Config.DEBUG and Config.MAX_SAMPLES:
            ids = ids[: Config.MAX_SAMPLES]

        count = len(ids)
        X = np.zeros((count, 3, 75, 75), dtype=np.float32)
        inc_angles = np.zeros((count,), dtype=np.float32)
        y = np.zeros((count,), dtype=np.float32) if not is_test else None

        for i, img_id in enumerate(ids):
            item = raw_map[img_id]

            # Construct Bands
            b1 = np.array(item["band_1"], dtype=np.float32).reshape(75, 75)
            b2 = np.array(item["band_2"], dtype=np.float32).reshape(75, 75)
            b3 = (b1 + b2) / 2.0

            X[i, 0] = b1
            X[i, 1] = b2
            X[i, 2] = b3

            # Incidence Angle
            # df has 'inc_angle' column, but we need to match by ID to be safe
            # or rely on the order if metadata generation preserved it.
            # Using explicit lookup is safer.
            angle = df.loc[df["id"] == img_id, "inc_angle"].values[0]
            if pd.isna(angle):
                inc_angles[i] = train_inc_mean
            else:
                inc_angles[i] = angle

            # Label
            if not is_test:
                label = df.loc[df["id"] == img_id, "is_iceberg"].values[0]
                y[i] = label

        return X, inc_angles, y, ids

    # Build splits
    X_train, inc_train, y_train, ids_train = build_arrays(df_train, is_test=False)
    X_val, inc_val, y_val, ids_val = build_arrays(df_val, is_test=False)
    X_test, inc_test, _, ids_test = build_arrays(df_test, is_test=True)

    # 5. Save to Cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez(
        cache_path,
        X_train=X_train,
        inc_train=inc_train,
        y_train=y_train,
        ids_train=ids_train,
        X_val=X_val,
        inc_val=inc_val,
        y_val=y_val,
        ids_val=ids_val,
        X_test=X_test,
        inc_test=inc_test,
        ids_test=ids_test,
        train_inc_mean=train_inc_mean,
    )
    print(f"Saved processed data to {cache_path}")

    return {
        "X_train": X_train,
        "inc_train": inc_train,
        "y_train": y_train,
        "ids_train": ids_train,
        "X_val": X_val,
        "inc_val": inc_val,
        "y_val": y_val,
        "ids_val": ids_val,
        "X_test": X_test,
        "inc_test": inc_test,
        "ids_test": ids_test,
        "train_inc_mean": train_inc_mean,
    }


def get_transforms(mode="train"):
    """
    Returns the augmentation pipeline.
    """
    if mode == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=1.0),  # Uniformly selects 0, 90, 180, 270
            ]
        )
    else:
        return None


def get_loaders(batch_size=Config.BATCH_SIZE, debug=Config.DEBUG):
    """
    Constructs DataLoaders for train, val, and test sets.
    """
    # 1. Process/Load Data
    data = process_data(load_cached_data=True)

    # 2. Get Global Stats
    stats = calculate_global_stats(load_cached_data=True, debug=debug)

    # Compute B3 (Mean channel) stats from the training data loaded
    # We use the full training set (X_train) to compute these stats
    b3_train = data["X_train"][:, 2, :, :]
    stats["b3_min"] = float(b3_train.min())
    stats["b3_max"] = float(b3_train.max())

    # 3. Create Datasets
    train_dataset = IcebergDataset(
        X=data["X_train"],
        inc_angles=data["inc_train"],
        labels=data["y_train"],
        transform=get_transforms("train"),
        global_stats=stats,
    )

    val_dataset = IcebergDataset(
        X=data["X_val"],
        inc_angles=data["inc_val"],
        labels=data["y_val"],
        transform=get_transforms("val"),
        global_stats=stats,
    )

    test_dataset = IcebergDataset(
        X=data["X_test"],
        inc_angles=data["inc_test"],
        labels=None,
        transform=get_transforms("test"),
        global_stats=stats,
    )

    # 4. Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )

    return train_loader, val_loader, test_loader, data["ids_test"]
