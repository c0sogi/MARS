import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from library.config import (
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    TRAIN_CACHE_PATH,
    VAL_CACHE_PATH,
    TEST_CACHE_PATH,
    SCALERS_CACHE_PATH,
    INPUT_DIR,
    TARGET_COLS,
    WORKING_DIR,
)
from library.data_utils import get_atomic_features, get_global_features, parse_xyz


def process_subset(metadata_path, is_test=False):
    """
    Reads metadata and extracts atomic and global features for each sample.

    Args:
        metadata_path (str): Path to the metadata CSV file.
        is_test (bool): Whether processing the test set (targets will be NaN).

    Returns:
        dict: Dictionary containing flattened arrays for atomic features,
              batch indices, global features, targets, and IDs.
    """
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)

    all_atomic_features = []
    all_global_features = []
    all_targets = []
    all_batch_indices = []
    all_ids = []

    for idx, row in df.iterrows():
        # Construct full file path to geometry file
        full_path = os.path.join(INPUT_DIR, row["file_path"])

        if not os.path.exists(full_path):
            print(f"Warning: Geometry file not found: {full_path}")
            continue

        # 1. Extract Atomic Features (N_atoms, 9)
        # Features: [One-hot(4), Coords(3), NN_Dist(1), Potential(1)]
        atomic_feats = get_atomic_features(full_path)

        # 2. Extract Global Features (12,)
        # We need lattice vectors and num_atoms for global feature calculation
        lattice_vectors, species, _ = parse_xyz(full_path)
        num_atoms = len(species)
        global_feats = get_global_features(row, lattice_vectors, num_atoms)

        # 3. Extract Targets
        if not is_test:
            targets = row[TARGET_COLS].values.astype(np.float32)
        else:
            targets = np.array([np.nan, np.nan], dtype=np.float32)

        # Append to lists
        all_atomic_features.append(atomic_feats)
        all_global_features.append(global_feats)
        all_targets.append(targets)
        all_ids.append(row["id"])

        # Create batch indices mapping atoms to their sample index (0 to N_samples-1)
        # This is essential for reconstructing the graph structure from flattened arrays
        batch_idx = np.full(num_atoms, idx, dtype=np.int32)
        all_batch_indices.append(batch_idx)

    # Concatenate into flat arrays
    if not all_atomic_features:
        raise ValueError(f"No data processed from {metadata_path}")

    flat_atomic = np.vstack(all_atomic_features)
    flat_batch_indices = np.concatenate(all_batch_indices)
    flat_global = np.vstack(all_global_features)
    flat_targets = np.vstack(all_targets)
    flat_ids = np.array(all_ids, dtype=np.int32)

    return {
        "atomic_features": flat_atomic,
        "batch_indices": flat_batch_indices,
        "global_features": flat_global,
        "targets": flat_targets,
        "ids": flat_ids,
    }


def prepare_datasets(load_cached_data=True):
    """
    Main function to prepare training, validation, and test datasets.

    Performs the following steps:
    1. Checks for existing cache files.
    2. If not found or forced reload:
       a. Processes raw data from metadata CSVs and XYZ files.
       b. Fits StandardScalers on the training data (atomic and global features).
       c. Transforms all splits using the fitted scalers.
       d. Log-transforms the target variables (log1p).
       e. Saves processed data and scalers to the working directory.

    Args:
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        tuple: (train_data, val_data, test_data, scalers)
               Data are NpzFile objects or dicts, scalers is an NpzFile.
    """
    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    cache_files = [
        TRAIN_CACHE_PATH,
        VAL_CACHE_PATH,
        TEST_CACHE_PATH,
        SCALERS_CACHE_PATH,
    ]
    all_exist = all(os.path.exists(f) for f in cache_files)

    if load_cached_data and all_exist:
        print("Loading cached data...")
        try:
            train_data = np.load(TRAIN_CACHE_PATH)
            val_data = np.load(VAL_CACHE_PATH)
            test_data = np.load(TEST_CACHE_PATH)
            scalers = np.load(SCALERS_CACHE_PATH)
            return train_data, val_data, test_data, scalers
        except Exception as e:
            print(f"Error loading cache: {e}. Recomputing...")

    print("Processing raw data from scratch...")

    # 1. Process Subsets
    print(f"Processing Train set from {TRAIN_METADATA_PATH}...")
    train_raw = process_subset(TRAIN_METADATA_PATH, is_test=False)

    print(f"Processing Validation set from {VAL_METADATA_PATH}...")
    val_raw = process_subset(VAL_METADATA_PATH, is_test=False)

    print(f"Processing Test set from {TEST_METADATA_PATH}...")
    test_raw = process_subset(TEST_METADATA_PATH, is_test=True)

    # 2. Fit Scalers on Training Data
    # Atomic Features: indices 0-3 are one-hot (don't scale), indices 4-8 are continuous (scale)
    # 4: x, 5: y, 6: z, 7: nn_dist, 8: potential
    print("Fitting scalers on training data...")
    atomic_scaler = StandardScaler()
    atomic_scaler.fit(train_raw["atomic_features"][:, 4:])

    # Global Features: All 12 dimensions are continuous/numeric
    global_scaler = StandardScaler()
    global_scaler.fit(train_raw["global_features"])

    # 3. Transform and Save
    def transform_and_save(raw_data, path):
        # Scale Atomic Features
        atomic_feats = raw_data["atomic_features"].copy()
        atomic_feats[:, 4:] = atomic_scaler.transform(atomic_feats[:, 4:])

        # Scale Global Features
        global_feats = global_scaler.transform(raw_data["global_features"])

        # Log Transform Targets (log1p)
        # Handle NaNs for test set
        targets = raw_data["targets"].copy()
        mask = ~np.isnan(targets)
        targets[mask] = np.log1p(targets[mask])

        np.savez(
            path,
            atomic_features=atomic_feats,
            batch_indices=raw_data["batch_indices"],
            global_features=global_feats,
            targets=targets,
            ids=raw_data["ids"],
        )

    print(f"Saving processed Train data to {TRAIN_CACHE_PATH}...")
    transform_and_save(train_raw, TRAIN_CACHE_PATH)

    print(f"Saving processed Validation data to {VAL_CACHE_PATH}...")
    transform_and_save(val_raw, VAL_CACHE_PATH)

    print(f"Saving processed Test data to {TEST_CACHE_PATH}...")
    transform_and_save(test_raw, TEST_CACHE_PATH)

    # Save Scaler parameters for inverse transformation later
    print(f"Saving scalers to {SCALERS_CACHE_PATH}...")
    np.savez(
        SCALERS_CACHE_PATH,
        atomic_mean=atomic_scaler.mean_,
        atomic_scale=atomic_scaler.scale_,
        global_mean=global_scaler.mean_,
        global_scale=global_scaler.scale_,
    )

    print("Data preparation complete.")

    # Reload to return consistent types
    train_data = np.load(TRAIN_CACHE_PATH)
    val_data = np.load(VAL_CACHE_PATH)
    test_data = np.load(TEST_CACHE_PATH)
    scalers = np.load(SCALERS_CACHE_PATH)

    return train_data, val_data, test_data, scalers
