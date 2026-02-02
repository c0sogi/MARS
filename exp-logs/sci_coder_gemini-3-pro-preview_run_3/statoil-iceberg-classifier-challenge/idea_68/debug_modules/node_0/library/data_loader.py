import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
from library.config import Config


class IcebergDataset(Dataset):
    """
    PyTorch Dataset for the Iceberg/Ship classification task.
    """

    def __init__(self, X, angles, y=None, ids=None, transform=None):
        """
        Args:
            X (np.ndarray): Images of shape (N, 3, 75, 75).
            angles (np.ndarray): Incidence angles of shape (N,).
            y (np.ndarray, optional): Labels of shape (N,).
            ids (np.ndarray, optional): IDs of shape (N,).
            transform (callable, optional): Transformations to apply to the images.
        """
        self.X = torch.FloatTensor(X)
        self.angles = torch.FloatTensor(angles)
        self.y = torch.FloatTensor(y) if y is not None else None
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = self.X[idx]
        angle = self.angles[idx]

        if self.transform:
            img = self.transform(img)

        if self.y is not None:
            return img, angle, self.y[idx]
        else:
            # For test set, return ID as well
            return img, angle, self.ids[idx]


def process_json_to_numpy(json_path, is_test=False):
    """
    Reads the raw JSON file and converts it to numpy arrays.
    Constructs 3-channel images: HH, HV, (HH+HV)/2.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    # Pre-allocate arrays
    num_samples = len(data)
    X = np.zeros((num_samples, 3, 75, 75), dtype=np.float32)
    angles = np.zeros(num_samples, dtype=np.float32)
    ids = []
    y = np.zeros(num_samples, dtype=np.float32) if not is_test else None

    for i, item in enumerate(data):
        # Process Bands
        # Band 1 (HH) and Band 2 (HV) are flattened 75x75 lists
        b1 = np.array(item["band_1"], dtype=np.float32).reshape(75, 75)
        b2 = np.array(item["band_2"], dtype=np.float32).reshape(75, 75)
        b3 = (b1 + b2) / 2.0  # Synthetic average band

        X[i, 0, :, :] = b1
        X[i, 1, :, :] = b2
        X[i, 2, :, :] = b3

        # Process Angle
        # "na" values are converted to NaN
        ang = item["inc_angle"]
        if ang == "na":
            angles[i] = np.nan
        else:
            angles[i] = float(ang)

        # Process ID
        ids.append(item["id"])

        # Process Label (only for train)
        if not is_test:
            y[i] = item["is_iceberg"]

    ids = np.array(ids)
    return X, angles, y, ids


def load_data_and_cache(load_cached_data=True):
    """
    Loads data from cache if available, otherwise processes raw JSONs and saves to cache.

    Returns:
        dict: Dictionary containing train and test arrays.
    """
    # Define cache paths
    cache_files = {
        "X_train": os.path.join(Config.CACHE_DIR, "X_train.npy"),
        "y_train": os.path.join(Config.CACHE_DIR, "y_train.npy"),
        "angle_train": os.path.join(Config.CACHE_DIR, "angle_train.npy"),
        "ids_train": os.path.join(Config.CACHE_DIR, "ids_train.npy"),
        "X_test": os.path.join(Config.CACHE_DIR, "X_test.npy"),
        "angle_test": os.path.join(Config.CACHE_DIR, "angle_test.npy"),
        "ids_test": os.path.join(Config.CACHE_DIR, "ids_test.npy"),
    }

    # Check if all cache files exist
    all_cached = all(os.path.exists(p) for p in cache_files.values())

    if load_cached_data and all_cached:
        print("Loading data from cache...")
        data = {}
        for k, v in cache_files.items():
            data[k] = np.load(
                v, allow_pickle=True
            )  # allow_pickle needed for object arrays (ids)
        return data

    print("Processing raw data from JSON...")
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Process Train
    X_train, angle_train, y_train, ids_train = process_json_to_numpy(
        Config.TRAIN_JSON, is_test=False
    )

    # Process Test
    X_test, angle_test, _, ids_test = process_json_to_numpy(
        Config.TEST_JSON, is_test=True
    )

    # Save to cache
    np.save(cache_files["X_train"], X_train)
    np.save(cache_files["y_train"], y_train)
    np.save(cache_files["angle_train"], angle_train)
    np.save(cache_files["ids_train"], ids_train)

    np.save(cache_files["X_test"], X_test)
    np.save(cache_files["angle_test"], angle_test)
    np.save(cache_files["ids_test"], ids_test)

    data = {
        "X_train": X_train,
        "y_train": y_train,
        "angle_train": angle_train,
        "ids_train": ids_train,
        "X_test": X_test,
        "angle_test": angle_test,
        "ids_test": ids_test,
    }
    return data


def get_loaders(fold_idx, load_cached_data=True):
    """
    Creates DataLoaders for a specific cross-validation fold.
    Performs leak-free imputation of incidence angles.

    Args:
        fold_idx (int): The current fold index (0 to N_FOLDS-1).
        load_cached_data (bool): Whether to use cached numpy files.

    Returns:
        train_loader, val_loader
    """
    data = load_data_and_cache(load_cached_data)

    X = data["X_train"]
    y = data["y_train"]
    angles = data["angle_train"]
    ids = data["ids_train"]

    # Stratified K-Fold
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Get indices for the requested fold
    # skf.split requires X and y, but X is 4D, so we pass zeros as placeholder or just y
    splits = list(skf.split(np.zeros(len(y)), y))
    train_idx, val_idx = splits[fold_idx]

    # Debugging subset
    if Config.DEBUG:
        train_idx = train_idx[: Config.DEBUG_SUBSET_SIZE]
        val_idx = val_idx[: Config.DEBUG_SUBSET_SIZE]

    # Split data
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    angle_train, angle_val = angles[train_idx], angles[val_idx]
    ids_train, ids_val = ids[train_idx], ids[val_idx]

    # Leak-Free Imputation: Calculate median ONLY on training data
    # Ignore NaNs for median calculation
    train_median_angle = np.nanmedian(angle_train)

    # Fill NaNs
    # Note: boolean indexing with isnan works for assignment
    angle_train[np.isnan(angle_train)] = train_median_angle
    angle_val[np.isnan(angle_val)] = train_median_angle

    # Augmentations
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()]
    )

    # Datasets
    train_dataset = IcebergDataset(
        X_train, angle_train, y_train, ids_train, transform=train_transform
    )
    val_dataset = IcebergDataset(X_val, angle_val, y_val, ids_val, transform=None)

    # Loaders
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


def get_test_loader(load_cached_data=True):
    """
    Creates a DataLoader for the test set.
    Imputes missing incidence angles using the median of the entire training set.

    Args:
        load_cached_data (bool): Whether to use cached numpy files.

    Returns:
        test_loader
    """
    data = load_data_and_cache(load_cached_data)

    X_test = data["X_test"]
    angle_test = data["angle_test"]
    ids_test = data["ids_test"]

    # For test imputation, we use the median of the FULL training set
    angle_train_full = data["angle_train"]
    global_train_median = np.nanmedian(angle_train_full)

    # Fill NaNs in test set
    angle_test[np.isnan(angle_test)] = global_train_median

    test_dataset = IcebergDataset(
        X_test, angle_test, y=None, ids=ids_test, transform=None
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    return test_loader
