import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from library.config import Config


class BirdDataset(Dataset):
    """
    PyTorch Dataset for Bird Species Prediction.
    """

    def __init__(self, features, labels, rec_ids):
        """
        Args:
            features (np.ndarray): Feature matrix of shape (N, input_dim).
            labels (np.ndarray): Label matrix of shape (N, num_classes).
            rec_ids (np.ndarray): Array of recording IDs of shape (N,).
        """
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.rec_ids = torch.tensor(rec_ids, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.rec_ids[idx]


def _load_histogram_features():
    """
    Loads the histogram features from the text file.
    Returns a DataFrame with 'rec_id' and feature columns.
    """
    hist_path = Config.HISTOGRAM_FILE_PATH
    if not os.path.exists(hist_path):
        raise FileNotFoundError(f"Histogram file not found at {hist_path}")

    # Check if header exists to determine if we need to skip a row
    with open(hist_path, "r") as f:
        first_line = f.readline()

    # If the first line starts with 'rec_id', it's a header
    has_header = "rec_id" in first_line
    skip_rows = 1 if has_header else 0

    # Cite debug_lesson_1: Prioritize Explicit Column Selection
    # Explicitly define column names: rec_id + 100 features
    col_names = ["rec_id"] + [f"feat_{i}" for i in range(Config.INPUT_DIM)]

    try:
        df = pd.read_csv(hist_path, header=None, names=col_names, skiprows=skip_rows)
    except Exception as e:
        raise RuntimeError(f"Failed to load histogram features: {e}")

    # Ensure rec_id is int
    df["rec_id"] = pd.to_numeric(df["rec_id"], errors="coerce").fillna(-1).astype(int)

    return df


def _process_split(metadata_path, hist_df, scaler=None, fit_scaler=False):
    """
    Processes a single split (train, val, or test).

    Args:
        metadata_path (str): Path to the metadata CSV.
        hist_df (pd.DataFrame): DataFrame containing histogram features.
        scaler (StandardScaler, optional): Scaler to transform features.
        fit_scaler (bool): Whether to fit the scaler on this data.

    Returns:
        tuple: (X, y, ids, scaler)
    """
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

    meta_df = pd.read_csv(metadata_path)

    # Identify label columns (species_0 to species_18)
    # Explicitly select the expected columns based on Config.NUM_CLASSES
    # This handles cases where metadata might have extra artifact columns (e.g. species_19)
    label_cols = [f"species_{i}" for i in range(Config.NUM_CLASSES)]

    # Verify these columns exist
    missing_cols = [c for c in label_cols if c not in meta_df.columns]
    if missing_cols:
        raise ValueError(f"Missing label columns in metadata: {missing_cols}")

    # Merge with histogram features
    # Left join to keep all records in metadata
    merged_df = pd.merge(meta_df, hist_df, on="rec_id", how="left")

    # Identify feature columns (those from hist_df excluding rec_id)
    feature_cols = [c for c in hist_df.columns if c != "rec_id"]

    # Extract features
    X = merged_df[feature_cols].values

    # Impute missing features with 0
    # This happens if a rec_id in metadata is not in histogram file
    if np.isnan(X).any():
        X = np.nan_to_num(X, nan=0.0)

    # Extract labels
    y = merged_df[label_cols].values

    # Extract IDs
    ids = merged_df["rec_id"].values

    # Normalize
    if fit_scaler:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    elif scaler is not None:
        X = scaler.transform(X)
    else:
        # If no scaler provided and not fitting, return raw (should not happen in this pipeline logic)
        pass

    return X, y, ids, scaler


def process_data(load_cached_data=True):
    """
    Loads, aligns, normalizes, and caches the data.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (train_data, val_data, test_data)
               Each data tuple contains (X, y, ids).
    """
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # Define cache filenames
    files = {
        "train": ["X_train.npy", "y_train.npy", "ids_train.npy"],
        "val": ["X_val.npy", "y_val.npy", "ids_val.npy"],
        "test": ["X_test.npy", "y_test.npy", "ids_test.npy"],
    }

    # Check if all cache files exist
    cache_exists = True
    for split in files:
        for fname in files[split]:
            if not os.path.exists(os.path.join(cache_dir, fname)):
                cache_exists = False
                break

    if load_cached_data and cache_exists:
        print("Loading data from cache...")
        data = {}
        for split in files:
            X = np.load(os.path.join(cache_dir, files[split][0]))
            y = np.load(os.path.join(cache_dir, files[split][1]))
            ids = np.load(os.path.join(cache_dir, files[split][2]))
            data[split] = (X, y, ids)
        return data["train"], data["val"], data["test"]

    print("Computing data from scratch...")

    # Load raw histogram features
    hist_df = _load_histogram_features()

    # Process Train (Fit Scaler)
    X_train, y_train, ids_train, scaler = _process_split(
        Config.TRAIN_METADATA_PATH, hist_df, fit_scaler=True
    )

    # Process Val (Apply Scaler)
    X_val, y_val, ids_val, _ = _process_split(
        Config.VAL_METADATA_PATH, hist_df, scaler=scaler, fit_scaler=False
    )

    # Process Test (Apply Scaler)
    X_test, y_test, ids_test, _ = _process_split(
        Config.TEST_METADATA_PATH, hist_df, scaler=scaler, fit_scaler=False
    )

    # Save to cache
    np.save(os.path.join(cache_dir, "X_train.npy"), X_train)
    np.save(os.path.join(cache_dir, "y_train.npy"), y_train)
    np.save(os.path.join(cache_dir, "ids_train.npy"), ids_train)

    np.save(os.path.join(cache_dir, "X_val.npy"), X_val)
    np.save(os.path.join(cache_dir, "y_val.npy"), y_val)
    np.save(os.path.join(cache_dir, "ids_val.npy"), ids_val)

    np.save(os.path.join(cache_dir, "X_test.npy"), X_test)
    np.save(os.path.join(cache_dir, "y_test.npy"), y_test)
    np.save(os.path.join(cache_dir, "ids_test.npy"), ids_test)

    return (
        (X_train, y_train, ids_train),
        (X_val, y_val, ids_val),
        (X_test, y_test, ids_test),
    )


def create_dataloaders(batch_size=Config.BATCH_SIZE, load_cached_data=True):
    """
    Creates PyTorch DataLoaders for train, validation, and test sets.

    Args:
        batch_size (int): Batch size for the dataloaders.
        load_cached_data (bool): Whether to use cached preprocessed data.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Get processed data
    train_data, val_data, test_data = process_data(load_cached_data=load_cached_data)

    # Unpack
    X_train, y_train, ids_train = train_data
    X_val, y_val, ids_val = val_data
    X_test, y_test, ids_test = test_data

    # Debugging subset logic (if configured)
    if Config.DEBUG_SUBSET_SIZE is not None and isinstance(
        Config.DEBUG_SUBSET_SIZE, int
    ):
        subset_size = min(len(X_train), Config.DEBUG_SUBSET_SIZE)
        X_train = X_train[:subset_size]
        y_train = y_train[:subset_size]
        ids_train = ids_train[:subset_size]
        print(f"Debug Mode: Reduced training set to {subset_size} samples.")

    # Create Datasets
    train_dataset = BirdDataset(X_train, y_train, ids_train)
    val_dataset = BirdDataset(X_val, y_val, ids_val)
    test_dataset = BirdDataset(X_test, y_test, ids_test)

    # Create DataLoaders
    # Shuffle training data, but not val/test
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    return train_loader, val_loader, test_loader
