import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
from library.config import Config


class IcebergDataset(Dataset):
    def __init__(self, X, angles, y=None, ids=None, transform=None):
        """
        Custom Dataset for Iceberg/Ship classification.

        Args:
            X (np.ndarray): Image data of shape (N, 3, 75, 75).
            angles (np.ndarray): Incidence angles of shape (N,).
            y (np.ndarray, optional): Target labels of shape (N,).
            ids (np.ndarray, optional): Image IDs of shape (N,).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.X = X
        self.angles = angles
        self.y = y
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Convert numpy arrays to float32 tensors
        img = torch.from_numpy(self.X[idx])
        angle = torch.tensor(self.angles[idx], dtype=torch.float32)

        # Apply augmentations if provided
        if self.transform:
            img = self.transform(img)

        # Return (img, angle, label) for training/validation
        if self.y is not None:
            label = torch.tensor(self.y[idx], dtype=torch.float32)
            return img, angle, label

        # Return (img, angle, id) for testing/inference
        else:
            id_val = self.ids[idx]
            return img, angle, id_val


def process_data(load_cached_data=True):
    """
    Loads raw JSON data, performs preprocessing (reshaping, channel creation, imputation),
    and caches the results as numpy arrays.

    Args:
        load_cached_data (bool): If True, attempts to load pre-processed data from disk.

    Returns:
        tuple: (X_train, angles_train, y_train, X_test, angles_test, ids_test)
    """
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # Define cache file paths
    files = {
        "X_train": os.path.join(cache_dir, "X_train.npy"),
        "angles_train": os.path.join(cache_dir, "angles_train.npy"),
        "y_train": os.path.join(cache_dir, "y_train.npy"),
        "X_test": os.path.join(cache_dir, "X_test.npy"),
        "angles_test": os.path.join(cache_dir, "angles_test.npy"),
        "ids_test": os.path.join(cache_dir, "ids_test.npy"),
    }

    # Attempt to load from cache
    if load_cached_data:
        all_exist = all(os.path.exists(f) for f in files.values())
        if all_exist:
            try:
                X_train = np.load(files["X_train"])
                angles_train = np.load(files["angles_train"])
                y_train = np.load(files["y_train"])
                X_test = np.load(files["X_test"])
                angles_test = np.load(files["angles_test"])
                ids_test = np.load(files["ids_test"], allow_pickle=True)
                return X_train, angles_train, y_train, X_test, angles_test, ids_test
            except Exception as e:
                print(f"Cache load failed ({e}). Reprocessing data...")
        else:
            pass  # Cache incomplete, reprocess

    # --- Process Training Data ---
    with open(Config.TRAIN_JSON, "r") as f:
        train_data = json.load(f)

    # Process Images: Reshape and create 3rd channel (Avg)
    # Band 1 (HH)
    train_b1 = np.array(
        [item["band_1"] for item in train_data], dtype=np.float32
    ).reshape(-1, 75, 75)
    # Band 2 (HV)
    train_b2 = np.array(
        [item["band_2"] for item in train_data], dtype=np.float32
    ).reshape(-1, 75, 75)
    # Band 3 (Average)
    train_b3 = (train_b1 + train_b2) / 2.0
    # Stack to (N, 3, 75, 75)
    X_train = np.stack([train_b1, train_b2, train_b3], axis=1)

    # Process Labels
    y_train = np.array([item["is_iceberg"] for item in train_data], dtype=np.float32)

    # Process Angles (Handle 'na')
    train_angles = []
    for item in train_data:
        angle = item["inc_angle"]
        if angle == "na":
            train_angles.append(np.nan)
        else:
            train_angles.append(float(angle))
    train_angles = np.array(train_angles, dtype=np.float32)

    # Impute missing angles with median of valid training data
    angle_median = np.nanmedian(train_angles)
    train_angles[np.isnan(train_angles)] = angle_median

    # --- Process Test Data ---
    with open(Config.TEST_JSON, "r") as f:
        test_data = json.load(f)

    test_b1 = np.array(
        [item["band_1"] for item in test_data], dtype=np.float32
    ).reshape(-1, 75, 75)
    test_b2 = np.array(
        [item["band_2"] for item in test_data], dtype=np.float32
    ).reshape(-1, 75, 75)
    test_b3 = (test_b1 + test_b2) / 2.0
    X_test = np.stack([test_b1, test_b2, test_b3], axis=1)

    ids_test = np.array([item["id"] for item in test_data])

    test_angles = []
    for item in test_data:
        angle = item["inc_angle"]
        if angle == "na":
            test_angles.append(angle_median)  # Use training median to avoid leakage
        else:
            test_angles.append(float(angle))
    test_angles = np.array(test_angles, dtype=np.float32)

    # --- Save to Cache ---
    np.save(files["X_train"], X_train)
    np.save(files["angles_train"], train_angles)
    np.save(files["y_train"], y_train)
    np.save(files["X_test"], X_test)
    np.save(files["angles_test"], test_angles)
    np.save(files["ids_test"], ids_test)

    return X_train, train_angles, y_train, X_test, test_angles, ids_test


def get_dataloaders(fold_idx=0, load_cached_data=True, debug=Config.DEBUG):
    """
    Constructs DataLoaders for the specified fold using Stratified K-Fold.

    Args:
        fold_idx (int): Index of the fold to retrieve (0 to N_FOLDS-1).
        load_cached_data (bool): Whether to use cached data.
        debug (bool): If True, uses a small subset of data for debugging.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load processed data
    X_all, angles_all, y_all, X_test, angles_test, ids_test = process_data(
        load_cached_data
    )

    # Apply debug subsetting if requested
    if debug:
        subset = Config.DEBUG_SUBSET_SIZE
        X_all = X_all[:subset]
        angles_all = angles_all[:subset]
        y_all = y_all[:subset]
        X_test = X_test[:subset]
        angles_test = angles_test[:subset]
        ids_test = ids_test[:subset]

    # Define Transforms (Augmentation for Train only)
    # Input is (3, 75, 75), transforms work on tensors
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()]
    )

    # Stratified K-Fold Split
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Generate splits (X is placeholder as we only need indices based on y)
    splits = list(skf.split(np.zeros(len(y_all)), y_all))

    if fold_idx >= len(splits):
        raise ValueError(
            f"fold_idx {fold_idx} is out of range for {Config.N_FOLDS} folds."
        )

    train_idx, val_idx = splits[fold_idx]

    # Create Datasets
    train_dataset = IcebergDataset(
        X_all[train_idx],
        angles_all[train_idx],
        y_all[train_idx],
        transform=train_transform,
    )

    val_dataset = IcebergDataset(
        X_all[val_idx], angles_all[val_idx], y_all[val_idx], transform=None
    )

    test_dataset = IcebergDataset(X_test, angles_test, ids=ids_test, transform=None)

    # Create DataLoaders
    # Pin memory for faster host-to-device transfer if CUDA is available
    use_cuda = Config.DEVICE == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=use_cuda,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=use_cuda,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=use_cuda,
    )

    return train_loader, val_loader, test_loader
