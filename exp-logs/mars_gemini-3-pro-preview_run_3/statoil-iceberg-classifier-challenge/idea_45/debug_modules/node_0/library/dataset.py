import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from sklearn.model_selection import StratifiedKFold
from library.config import Config


class IcebergDataset(Dataset):
    """
    Custom Dataset for Iceberg/Ship classification.
    Handles 3-channel tensor construction and augmentations.
    """

    def __init__(self, X, angles, y=None, ids=None, transform=None):
        self.X = X
        self.angles = angles
        self.y = y
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # X is (N, 3, 75, 75) numpy array
        img = self.X[idx]
        angle = self.angles[idx]

        # Convert to tensor
        img_tensor = torch.from_numpy(img).float()
        angle_tensor = torch.tensor(angle, dtype=torch.float32)

        # Apply augmentations if provided
        if self.transform:
            img_tensor = self.transform(img_tensor)

        # Return (img, angle, label) for train/val
        if self.y is not None:
            label_tensor = torch.tensor(self.y[idx], dtype=torch.float32)
            return img_tensor, angle_tensor, label_tensor
        # Return (img, angle, id) for test
        else:
            id_str = self.ids[idx]
            return img_tensor, angle_tensor, id_str


def process_data(load_cached_data=True):
    """
    Loads raw JSON data, processes it into numpy arrays, and caches it.
    Returns tuple of (train_data, test_data).
    """
    # Define cache paths
    cache_files = {
        "X_train": os.path.join(Config.CACHE_DIR, "X_train_full.npy"),
        "angle_train": os.path.join(Config.CACHE_DIR, "angle_train_full.npy"),
        "y_train": os.path.join(Config.CACHE_DIR, "y_train_full.npy"),
        "X_test": os.path.join(Config.CACHE_DIR, "X_test.npy"),
        "angle_test": os.path.join(Config.CACHE_DIR, "angle_test.npy"),
        "ids_test": os.path.join(Config.CACHE_DIR, "ids_test.npy"),
    }

    # Check if cache exists and we want to load it
    if load_cached_data:
        all_exist = all(os.path.exists(p) for p in cache_files.values())
        if all_exist:
            print("Loading data from cache...")
            X_train = np.load(cache_files["X_train"])
            angle_train = np.load(cache_files["angle_train"])
            y_train = np.load(cache_files["y_train"])

            X_test = np.load(cache_files["X_test"])
            angle_test = np.load(cache_files["angle_test"])
            ids_test = np.load(cache_files["ids_test"])

            return (X_train, angle_train, y_train), (X_test, angle_test, ids_test)

    print("Processing data from raw JSON files...")

    def load_and_process(json_path, is_train=True):
        with open(json_path, "r") as f:
            data = json.load(f)

        X_list = []
        angle_list = []
        y_list = []
        id_list = []

        for item in data:
            # Process Bands
            # Reshape flattened 5625 to 75x75
            b1 = np.array(item["band_1"]).reshape(75, 75)
            b2 = np.array(item["band_2"]).reshape(75, 75)
            # Create 3rd channel: Average
            avg = (b1 + b2) / 2.0

            # Stack channels: (3, 75, 75)
            img = np.stack([b1, b2, avg], axis=0)
            X_list.append(img)

            # Process Angle
            ang = item["inc_angle"]
            if ang == "na":
                angle_list.append(np.nan)
            else:
                angle_list.append(float(ang))

            # Process ID
            id_list.append(item["id"])

            # Process Label
            if is_train:
                y_list.append(item["is_iceberg"])

        X = np.array(X_list, dtype=np.float32)
        angles = np.array(angle_list, dtype=np.float32)
        ids = np.array(id_list)

        if is_train:
            y = np.array(y_list, dtype=np.float32)
            return X, angles, y, ids
        else:
            return X, angles, None, ids

    # Process Train
    X_train, angle_train, y_train, _ = load_and_process(
        Config.TRAIN_JSON, is_train=True
    )

    # Process Test
    X_test, angle_test, _, ids_test = load_and_process(Config.TEST_JSON, is_train=False)

    # Save to cache
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    np.save(cache_files["X_train"], X_train)
    np.save(cache_files["angle_train"], angle_train)
    np.save(cache_files["y_train"], y_train)
    np.save(cache_files["X_test"], X_test)
    np.save(cache_files["angle_test"], angle_test)
    np.save(cache_files["ids_test"], ids_test)

    return (X_train, angle_train, y_train), (X_test, angle_test, ids_test)


def get_loaders(
    fold,
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    load_cached_data=True,
):
    """
    Generates Stratified K-Fold train and validation loaders for a specific fold.
    Imputes missing angles using the training fold median.
    """
    (X_all, angle_all, y_all), _ = process_data(load_cached_data)

    # Stratified K-Fold
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Find the indices for the requested fold
    train_idx, val_idx = None, None
    for i, (t_idx, v_idx) in enumerate(skf.split(X_all, y_all)):
        if i == fold:
            train_idx = t_idx
            val_idx = v_idx
            break

    if train_idx is None:
        raise ValueError(f"Fold {fold} out of range for {Config.N_FOLDS} splits.")

    # Split data
    X_train, angle_train, y_train = (
        X_all[train_idx],
        angle_all[train_idx],
        y_all[train_idx],
    )
    X_val, angle_val, y_val = X_all[val_idx], angle_all[val_idx], y_all[val_idx]

    # Impute missing angles (using training median to prevent leakage)
    valid_angles = angle_train[~np.isnan(angle_train)]
    median_val = np.median(valid_angles)

    angle_train = np.nan_to_num(angle_train, nan=median_val)
    angle_val = np.nan_to_num(angle_val, nan=median_val)

    # Augmentation for training
    train_transform = T.Compose(
        [
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
        ]
    )

    # Datasets
    train_ds = IcebergDataset(X_train, angle_train, y_train, transform=train_transform)
    val_ds = IcebergDataset(X_val, angle_val, y_val, transform=None)

    # Loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def get_test_loader(
    batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, load_cached_data=True
):
    """
    Generates the test loader.
    Imputes missing angles using the global training median.
    """
    (X_train_full, angle_train_full, _), (X_test, angle_test, ids_test) = process_data(
        load_cached_data
    )

    # Impute missing angles in test using global training median
    valid_angles = angle_train_full[~np.isnan(angle_train_full)]
    median_val = np.median(valid_angles)

    angle_test = np.nan_to_num(angle_test, nan=median_val)

    test_ds = IcebergDataset(X_test, angle_test, ids=ids_test, transform=None)

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return test_loader
