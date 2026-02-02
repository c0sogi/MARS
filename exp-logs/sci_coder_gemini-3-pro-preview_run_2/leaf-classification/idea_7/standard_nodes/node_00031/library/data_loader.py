import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from library.config import (
    TRAIN_PATH,
    VAL_PATH,
    TEST_PATH,
    WORKING_DIR,
    ID_COL,
    TARGET_COL,
    GENUS_COL,
    RANDOM_SEED,
)
from library.utils import extract_genus


def load_and_process_data(load_cached_data=True):
    """
    Loads, merges, and processes the leaf classification dataset.

    Implements caching for the intermediate numpy arrays to speed up subsequent runs.
    Scalers and Encoders are always fitted in-memory to ensure valid objects are returned.

    Args:
        load_cached_data (bool): If True, attempts to load pre-processed numpy arrays
                                 from the working directory.

    Returns:
        tuple: (X_train, X_test, y_species, y_genus, test_ids, scaler, species_le, genus_le)
            - X_train (np.ndarray): Scaled training features (N_train, 192).
            - X_test (np.ndarray): Scaled test features (N_test, 192).
            - y_species (np.ndarray): Encoded species labels (N_train,).
            - y_genus (np.ndarray): Encoded genus labels (N_train,).
            - test_ids (np.ndarray): IDs for the test set (N_test,).
            - scaler (StandardScaler): Fitted scaler object.
            - species_le (LabelEncoder): Fitted encoder for species.
            - genus_le (LabelEncoder): Fitted encoder for genus.
    """
    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Define cache file paths
    cache_files = {
        "X_train": os.path.join(WORKING_DIR, "X_train_raw.npy"),
        "y_species": os.path.join(WORKING_DIR, "y_species_raw.npy"),
        "y_genus": os.path.join(WORKING_DIR, "y_genus_raw.npy"),
        "X_test": os.path.join(WORKING_DIR, "X_test_raw.npy"),
        "test_ids": os.path.join(WORKING_DIR, "test_ids.npy"),
    }

    data_loaded_from_cache = False

    # 1. Try Loading from Cache
    if load_cached_data:
        try:
            if all(os.path.exists(path) for path in cache_files.values()):
                print("Loading data from cache...")
                X_train_raw = np.load(cache_files["X_train"], allow_pickle=True)
                y_species_raw = np.load(cache_files["y_species"], allow_pickle=True)
                y_genus_raw = np.load(cache_files["y_genus"], allow_pickle=True)
                X_test_raw = np.load(cache_files["X_test"], allow_pickle=True)
                test_ids = np.load(cache_files["test_ids"], allow_pickle=True)
                data_loaded_from_cache = True
            else:
                print("Cache files incomplete. Reloading from source...")
        except Exception as e:
            print(f"Error loading cache: {e}. Reloading from source...")
            data_loaded_from_cache = False

    # 2. Process from Source if Cache Failed or Disabled
    if not data_loaded_from_cache:
        print("Processing data from CSV files...")

        # Load datasets
        df_train_part = pd.read_csv(TRAIN_PATH)
        df_val_part = pd.read_csv(VAL_PATH)
        df_test = pd.read_csv(TEST_PATH)

        # Merge Train and Validation sets (as per strategy)
        df_train = pd.concat([df_train_part, df_val_part], axis=0, ignore_index=True)

        # Generate Genus Labels
        # Apply extract_genus to the species column
        df_train[GENUS_COL] = df_train[TARGET_COL].apply(extract_genus)

        # Identify Feature Columns
        # Exclude metadata columns to get the 192 features
        excluded_cols = {ID_COL, TARGET_COL, GENUS_COL, "image_path"}
        feature_cols = [c for c in df_train.columns if c not in excluded_cols]

        # Ensure column order consistency between train and test
        # Test CSV might have different order or missing target cols, so we select explicitly
        X_train_raw = df_train[feature_cols].values
        X_test_raw = df_test[feature_cols].values

        # Extract Targets and IDs
        y_species_raw = df_train[TARGET_COL].values
        y_genus_raw = df_train[GENUS_COL].values
        test_ids = df_test[ID_COL].values

        # Save to Cache
        np.save(cache_files["X_train"], X_train_raw)
        np.save(cache_files["y_species"], y_species_raw)
        np.save(cache_files["y_genus"], y_genus_raw)
        np.save(cache_files["X_test"], X_test_raw)
        np.save(cache_files["test_ids"], test_ids)
        print(f"Data processed and cached to {WORKING_DIR}")

    # 3. In-Memory Preprocessing (Scaling & Encoding)
    # We always re-fit these to return valid, live objects
    print("Fitting encoders and scalers...")

    # Encoders
    species_le = LabelEncoder()
    y_species = species_le.fit_transform(y_species_raw)

    genus_le = LabelEncoder()
    y_genus = genus_le.fit_transform(y_genus_raw)

    # Scaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    print(f"Processing complete.")
    print(f"Train Shape: {X_train.shape}, Test Shape: {X_test.shape}")
    print(
        f"Unique Species: {len(species_le.classes_)}, Unique Genera: {len(genus_le.classes_)}"
    )

    return X_train, X_test, y_species, y_genus, test_ids, scaler, species_le, genus_le
