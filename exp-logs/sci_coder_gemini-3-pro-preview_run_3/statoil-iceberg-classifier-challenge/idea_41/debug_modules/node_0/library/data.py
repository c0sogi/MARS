import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from library.config import Config
from library.utils import set_seed


class IcebergDataset(Dataset):
    """
    Custom Dataset for Iceberg vs Ship classification.
    """

    def __init__(self, X, angles, y=None, transform=None, ids=None):
        """
        Args:
            X (np.ndarray): Images of shape (N, 75, 75, 3).
            angles (np.ndarray): Incidence angles of shape (N,).
            y (np.ndarray, optional): Labels of shape (N,).
            transform (callable, optional): Optional transform to be applied on a sample.
            ids (np.ndarray, optional): Image IDs.
        """
        self.X = X
        self.angles = angles
        self.y = y
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Retrieve image and angle
        # Input X is (75, 75, 3), convert to (3, 75, 75) for PyTorch
        img = self.X[idx]
        angle = self.angles[idx]

        # Convert to tensor (C, H, W)
        # We keep the raw float values (dB), so we just transpose and convert to float32
        img_tensor = torch.from_numpy(img).float().permute(2, 0, 1)
        angle_tensor = torch.tensor(angle, dtype=torch.float32)

        # Apply transforms (augmentation)
        if self.transform:
            img_tensor = self.transform(img_tensor)

        # Prepare return tuple
        if self.y is not None:
            label = torch.tensor(self.y[idx], dtype=torch.float32)
            return img_tensor, angle_tensor, label
        else:
            # For test set, we might need the ID for submission
            img_id = self.ids[idx] if self.ids is not None else ""
            return img_tensor, angle_tensor, img_id


def _process_json_to_numpy(json_path, metadata_df):
    """
    Helper to extract specific rows from raw JSON based on metadata indices.
    Returns X (images), angles, ids, and y (labels) if available.
    """
    # Load the full JSON file
    with open(json_path, "r") as f:
        raw_data = json.load(f)

    # Map original_index to the raw data list
    # The raw_data is a list of dicts. metadata_df['original_index'] points to the index in this list.
    indices = metadata_df["original_index"].values
    ids = metadata_df["id"].values

    # Pre-allocate arrays
    n_samples = len(indices)
    img_height = Config.IMG_HEIGHT
    img_width = Config.IMG_WIDTH

    X = np.zeros((n_samples, img_height, img_width, 3), dtype=np.float32)
    angles = np.full(n_samples, np.nan, dtype=np.float32)
    y = (
        np.zeros(n_samples, dtype=np.float32)
        if "is_iceberg" in metadata_df.columns
        else None
    )

    for i, original_idx in enumerate(indices):
        item = raw_data[original_idx]

        # Verify ID match to ensure data integrity
        if item["id"] != ids[i]:
            raise ValueError(
                f"ID mismatch at index {i}: Meta {ids[i]} vs JSON {item['id']}"
            )

        # Process Bands
        band_1 = np.array(item["band_1"]).reshape(img_height, img_width)
        band_2 = np.array(item["band_2"]).reshape(img_height, img_width)
        band_3 = (band_1 + band_2) / 2.0

        X[i, :, :, 0] = band_1
        X[i, :, :, 1] = band_2
        X[i, :, :, 2] = band_3

        # Process Angle
        # Note: metadata already has 'inc_angle' with 'na' converted to NaN,
        # but we pull from JSON to be consistent with raw processing logic if needed.
        # However, to save time parsing 'na' again, we can use metadata values if we trust them.
        # Let's use the value from the raw JSON to be safe and handle 'na' explicitly.
        ang = item["inc_angle"]
        if ang != "na":
            angles[i] = float(ang)

        # Process Label
        if y is not None:
            y[i] = item["is_iceberg"]

    return X, angles, ids, y


def _get_data_splits(load_cached_data=True):
    """
    Loads data from cache or processes from scratch.
    Implements imputation logic.
    """
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # Define filenames
    files = {
        "X_train": os.path.join(cache_dir, "X_train.npy"),
        "y_train": os.path.join(cache_dir, "y_train.npy"),
        "angles_train": os.path.join(cache_dir, "angles_train.npy"),
        "ids_train": os.path.join(cache_dir, "ids_train.npy"),
        "X_val": os.path.join(cache_dir, "X_val.npy"),
        "y_val": os.path.join(cache_dir, "y_val.npy"),
        "angles_val": os.path.join(cache_dir, "angles_val.npy"),
        "ids_val": os.path.join(cache_dir, "ids_val.npy"),
        "X_test": os.path.join(cache_dir, "X_test.npy"),
        "angles_test": os.path.join(cache_dir, "angles_test.npy"),
        "ids_test": os.path.join(cache_dir, "ids_test.npy"),
    }

    # Check if all files exist
    all_exist = all(os.path.exists(f) for f in files.values())

    if load_cached_data and all_exist:
        print("Loading data from cache...")
        data = {k: np.load(v, allow_pickle=True) for k, v in files.items()}
        # Handle IDs which might be object arrays
        data["ids_train"] = data["ids_train"].astype(str)
        data["ids_val"] = data["ids_val"].astype(str)
        data["ids_test"] = data["ids_test"].astype(str)
        return data

    print("Processing data from scratch...")

    # Load Metadata
    train_meta = pd.read_csv(Config.TRAIN_META)
    val_meta = pd.read_csv(Config.VAL_META)
    test_meta = pd.read_csv(Config.TEST_META)

    # Process Train and Val (from train.json)
    # We load train.json once to avoid I/O overhead
    print(f"Loading raw {Config.TRAIN_JSON}...")
    with open(Config.TRAIN_JSON, "r") as f:
        train_json_data = json.load(f)

    def extract_from_loaded_json(meta_df, raw_list):
        indices = meta_df["original_index"].values
        ids = meta_df["id"].values
        n = len(indices)
        X = np.zeros((n, Config.IMG_HEIGHT, Config.IMG_WIDTH, 3), dtype=np.float32)
        angles = np.full(n, np.nan, dtype=np.float32)
        y = np.zeros(n, dtype=np.float32)

        for i, idx in enumerate(indices):
            item = raw_list[idx]
            if item["id"] != ids[i]:
                raise ValueError("ID mismatch")

            b1 = np.array(item["band_1"]).reshape(Config.IMG_HEIGHT, Config.IMG_WIDTH)
            b2 = np.array(item["band_2"]).reshape(Config.IMG_HEIGHT, Config.IMG_WIDTH)
            X[i, ..., 0] = b1
            X[i, ..., 1] = b2
            X[i, ..., 2] = (b1 + b2) / 2.0

            if item["inc_angle"] != "na":
                angles[i] = float(item["inc_angle"])

            y[i] = item["is_iceberg"]

        return X, angles, ids, y

    X_train, angles_train, ids_train, y_train = extract_from_loaded_json(
        train_meta, train_json_data
    )
    X_val, angles_val, ids_val, y_val = extract_from_loaded_json(
        val_meta, train_json_data
    )

    # Free memory
    del train_json_data

    # Process Test (from test.json)
    print(f"Loading raw {Config.TEST_JSON}...")
    X_test, angles_test, ids_test, _ = _process_json_to_numpy(
        Config.TEST_JSON, test_meta
    )

    # Impute Incidence Angles
    # Calculate median from TRAIN set only (ignoring NaNs)
    median_angle = np.nanmedian(angles_train)
    print(f"Imputing missing angles with training median: {median_angle:.4f}")

    # Fill NaNs
    angles_train = np.nan_to_num(angles_train, nan=median_angle)
    angles_val = np.nan_to_num(angles_val, nan=median_angle)
    angles_test = np.nan_to_num(angles_test, nan=median_angle)

    # Save to Cache
    print("Saving processed data to cache...")
    np.save(files["X_train"], X_train)
    np.save(files["y_train"], y_train)
    np.save(files["angles_train"], angles_train)
    np.save(files["ids_train"], ids_train)

    np.save(files["X_val"], X_val)
    np.save(files["y_val"], y_val)
    np.save(files["angles_val"], angles_val)
    np.save(files["ids_val"], ids_val)

    np.save(files["X_test"], X_test)
    np.save(files["angles_test"], angles_test)
    np.save(files["ids_test"], ids_test)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "angles_train": angles_train,
        "ids_train": ids_train,
        "X_val": X_val,
        "y_val": y_val,
        "angles_val": angles_val,
        "ids_val": ids_val,
        "X_test": X_test,
        "angles_test": angles_test,
        "ids_test": ids_test,
    }


def get_dataloaders(load_cached_data=True):
    """
    Main function to get DataLoaders for Train, Val, and Test.

    Args:
        load_cached_data (bool): If True, attempts to load pre-processed .npy files.

    Returns:
        train_loader, val_loader, test_loader
    """
    set_seed(Config.SEED)

    # Load Data
    data = _get_data_splits(load_cached_data=load_cached_data)

    X_train, y_train, ang_train, ids_train = (
        data["X_train"],
        data["y_train"],
        data["angles_train"],
        data["ids_train"],
    )
    X_val, y_val, ang_val, ids_val = (
        data["X_val"],
        data["y_val"],
        data["angles_val"],
        data["ids_val"],
    )
    X_test, ang_test, ids_test = data["X_test"], data["angles_test"], data["ids_test"]

    # Debug Mode: Subset data
    if Config.DEBUG:
        print(f"DEBUG mode enabled. Subsetting to {Config.DEBUG_SAMPLE_SIZE} samples.")
        limit = Config.DEBUG_SAMPLE_SIZE
        X_train, y_train, ang_train, ids_train = (
            X_train[:limit],
            y_train[:limit],
            ang_train[:limit],
            ids_train[:limit],
        )
        X_val, y_val, ang_val, ids_val = (
            X_val[:limit],
            y_val[:limit],
            ang_val[:limit],
            ids_val[:limit],
        )
        X_test, ang_test, ids_test = X_test[:limit], ang_test[:limit], ids_test[:limit]

    # Define Transforms
    # Note: Input is (C, H, W) Tensor.
    train_transform = None
    if Config.USE_AUGMENTATION:
        train_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=Config.HORIZONTAL_FLIP_PROB),
                transforms.RandomVerticalFlip(p=Config.VERTICAL_FLIP_PROB),
            ]
        )

    # Create Datasets
    train_dataset = IcebergDataset(
        X_train, ang_train, y_train, transform=train_transform, ids=ids_train
    )
    val_dataset = IcebergDataset(X_val, ang_val, y_val, transform=None, ids=ids_val)
    test_dataset = IcebergDataset(
        X_test, ang_test, y=None, transform=None, ids=ids_test
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
        drop_last=True,  # Drop last incomplete batch to stabilize BatchNorm stats
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    print(
        f"DataLoaders created. Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )

    return train_loader, val_loader, test_loader
