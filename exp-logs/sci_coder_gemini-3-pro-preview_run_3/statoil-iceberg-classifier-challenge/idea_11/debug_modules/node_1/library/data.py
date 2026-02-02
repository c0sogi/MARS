import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from library.config import Config
from library.utils import set_seed


class IcebergDataset(Dataset):
    def __init__(self, X, angles, y=None, ids=None, transform=None):
        """
        Args:
            X (np.ndarray): Images of shape (N, 3, 75, 75).
            angles (np.ndarray): Incidence angles of shape (N,).
            y (np.ndarray, optional): Labels of shape (N,).
            ids (np.ndarray, optional): IDs of shape (N,).
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
        # Retrieve data
        img = self.X[idx]  # Shape: (3, 75, 75)
        angle = self.angles[idx]

        # Convert to tensor
        # Data is float32, already normalized/scaled in dB, so we just convert to tensor
        img_tensor = torch.from_numpy(img).float()
        angle_tensor = torch.tensor(angle, dtype=torch.float32)

        # Apply transforms if any
        if self.transform:
            img_tensor = self.transform(img_tensor)

        # Prepare return tuple
        if self.y is not None:
            label = torch.tensor(self.y[idx], dtype=torch.float32)
            return img_tensor, angle_tensor, label
        else:
            # Inference mode
            id_val = self.ids[idx]
            return img_tensor, angle_tensor, id_val


def get_transforms(mode="train"):
    """
    Returns torchvision transforms for training or validation/test.
    Since input is already a float tensor (dB values), we use transforms that work on tensors.
    """
    if mode == "train":
        return transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
            ]
        )
    else:
        return None


def process_images(df):
    """
    Converts band_1 and band_2 lists from DataFrame into a (N, 3, 75, 75) numpy array.
    Channel 0: HH (band_1)
    Channel 1: HV (band_2)
    Channel 2: Average ((HH + HV) / 2)
    """
    # Stack band_1 and band_2 into arrays
    # Each band is a list of 5625 floats
    b1 = np.array(df["band_1"].tolist(), dtype=np.float32).reshape(-1, 75, 75)
    b2 = np.array(df["band_2"].tolist(), dtype=np.float32).reshape(-1, 75, 75)

    # Calculate average band
    avg = (b1 + b2) / 2.0

    # Stack to create (N, 3, 75, 75)
    # Axis 1 is the channel dimension
    X = np.stack([b1, b2, avg], axis=1)
    return X


def prepare_data(load_cached_data=True):
    """
    Loads data from cache or processes raw JSON files.
    Implements caching mechanism using .npy files.
    """
    # Check if all cache files exist
    cache_files = [
        Config.CACHE_TRAIN_X,
        Config.CACHE_TRAIN_Y,
        Config.CACHE_TRAIN_ANGLE,
        Config.CACHE_TRAIN_IDS,
        Config.CACHE_VAL_X,
        Config.CACHE_VAL_Y,
        Config.CACHE_VAL_ANGLE,
        Config.CACHE_VAL_IDS,
        Config.CACHE_TEST_X,
        Config.CACHE_TEST_ANGLE,
        Config.CACHE_TEST_IDS,
    ]

    cache_exists = all(os.path.exists(f) for f in cache_files)

    if load_cached_data and cache_exists:
        print("Loading data from cache...")
        X_train = np.load(Config.CACHE_TRAIN_X)
        y_train = np.load(Config.CACHE_TRAIN_Y)
        angle_train = np.load(Config.CACHE_TRAIN_ANGLE)
        ids_train = np.load(Config.CACHE_TRAIN_IDS, allow_pickle=True)

        X_val = np.load(Config.CACHE_VAL_X)
        y_val = np.load(Config.CACHE_VAL_Y)
        angle_val = np.load(Config.CACHE_VAL_ANGLE)
        ids_val = np.load(Config.CACHE_VAL_IDS, allow_pickle=True)

        X_test = np.load(Config.CACHE_TEST_X)
        angle_test = np.load(Config.CACHE_TEST_ANGLE)
        ids_test = np.load(Config.CACHE_TEST_IDS, allow_pickle=True)

        return (
            (X_train, angle_train, y_train, ids_train),
            (X_val, angle_val, y_val, ids_val),
            (X_test, angle_test, ids_test),
        )

    print("Processing raw data from JSON files...")

    # Load Metadata
    train_meta = pd.read_csv(Config.TRAIN_META_CSV)
    val_meta = pd.read_csv(Config.VAL_META_CSV)
    test_meta = pd.read_csv(Config.TEST_META_CSV)

    # Load Raw JSONs
    # Note: Reading JSON is memory intensive but fits in 220GB RAM
    print("Loading train.json...")
    df_train_full = pd.read_json(Config.TRAIN_JSON)
    print("Loading test.json...")
    df_test_full = pd.read_json(Config.TEST_JSON)

    # Handle Incidence Angle Imputation
    # Convert 'na' to NaN and then float
    df_train_full["inc_angle"] = pd.to_numeric(
        df_train_full["inc_angle"], errors="coerce"
    )
    df_test_full["inc_angle"] = pd.to_numeric(
        df_test_full["inc_angle"], errors="coerce"
    )

    # Calculate median from TRAINING subset only to prevent leakage
    # We use the indices from train_meta to identify the training subset
    train_indices = train_meta["original_index"].values
    train_subset_angles = df_train_full.iloc[train_indices]["inc_angle"]
    angle_median = train_subset_angles.median()

    print(f"Imputing missing incidence angles with median: {angle_median}")

    # Fill NaNs
    df_train_full["inc_angle"] = df_train_full["inc_angle"].fillna(angle_median)
    df_test_full["inc_angle"] = df_test_full["inc_angle"].fillna(angle_median)

    # --- Process Train Split ---
    print("Processing Train Split...")
    df_train_split = df_train_full.iloc[train_meta["original_index"].values]
    X_train = process_images(df_train_split)
    angle_train = df_train_split["inc_angle"].values.astype(np.float32)
    y_train = df_train_split["is_iceberg"].values.astype(np.float32)
    ids_train = df_train_split["id"].values

    # --- Process Validation Split ---
    print("Processing Validation Split...")
    df_val_split = df_train_full.iloc[val_meta["original_index"].values]
    X_val = process_images(df_val_split)
    angle_val = df_val_split["inc_angle"].values.astype(np.float32)
    y_val = df_val_split["is_iceberg"].values.astype(np.float32)
    ids_val = df_val_split["id"].values

    # --- Process Test Split ---
    print("Processing Test Split...")
    # Test metadata might not be in the same order as test.json, so we use indices
    df_test_split = df_test_full.iloc[test_meta["original_index"].values]
    X_test = process_images(df_test_split)
    angle_test = df_test_split["inc_angle"].values.astype(np.float32)
    ids_test = df_test_split["id"].values

    # Save to Cache
    print("Saving data to cache...")
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    np.save(Config.CACHE_TRAIN_X, X_train)
    np.save(Config.CACHE_TRAIN_Y, y_train)
    np.save(Config.CACHE_TRAIN_ANGLE, angle_train)
    np.save(Config.CACHE_TRAIN_IDS, ids_train)

    np.save(Config.CACHE_VAL_X, X_val)
    np.save(Config.CACHE_VAL_Y, y_val)
    np.save(Config.CACHE_VAL_ANGLE, angle_val)
    np.save(Config.CACHE_VAL_IDS, ids_val)

    np.save(Config.CACHE_TEST_X, X_test)
    np.save(Config.CACHE_TEST_ANGLE, angle_test)
    np.save(Config.CACHE_TEST_IDS, ids_test)

    return (
        (X_train, angle_train, y_train, ids_train),
        (X_val, angle_val, y_val, ids_val),
        (X_test, angle_test, ids_test),
    )


def get_dataloaders(load_cached_data=True):
    """
    Creates and returns DataLoaders for train, validation, and test sets.
    """
    set_seed(Config.SEED)

    # Load Data
    train_data, val_data, test_data = prepare_data(load_cached_data=load_cached_data)

    X_train, angle_train, y_train, ids_train = train_data
    X_val, angle_val, y_val, ids_val = val_data
    X_test, angle_test, ids_test = test_data

    # Create Datasets
    train_dataset = IcebergDataset(
        X_train, angle_train, y_train, ids_train, transform=get_transforms("train")
    )

    val_dataset = IcebergDataset(
        X_val, angle_val, y_val, ids_val, transform=get_transforms("val")
    )

    test_dataset = IcebergDataset(
        X_test, angle_test, y=None, ids=ids_test, transform=get_transforms("test")
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
