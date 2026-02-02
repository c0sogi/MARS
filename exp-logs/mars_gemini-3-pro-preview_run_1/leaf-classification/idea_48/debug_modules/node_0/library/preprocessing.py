import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, StandardScaler, LabelEncoder
from library.config import Config
from library.feature_extraction import process_dataset


class HighPrecisionPipeline:
    """
    A wrapper class for the preprocessing pipeline ensuring float64 precision.
    Applies Yeo-Johnson transformation followed by Standard Scaling.
    """

    def __init__(self):
        # Yeo-Johnson handles positive and negative values, unlike Box-Cox.
        # standardize=False because we apply StandardScaler explicitly afterwards.
        self.pt = PowerTransformer(method="yeo-johnson", standardize=False)
        self.scaler = StandardScaler()

    def fit(self, X):
        """
        Fit the pipeline to the data.
        Args:
            X (np.ndarray): Training data.
        """
        X = X.astype(Config.DTYPE)
        self.pt.fit(X)
        # Transform strictly for the purpose of fitting the scaler
        X_pt = self.pt.transform(X)
        self.scaler.fit(X_pt)
        return self

    def transform(self, X):
        """
        Apply the transformations to the data.
        Args:
            X (np.ndarray): Data to transform.
        Returns:
            np.ndarray: Transformed data in float64.
        """
        X = X.astype(Config.DTYPE)
        X_pt = self.pt.transform(X)
        X_scaled = self.scaler.transform(X_pt)
        return X_scaled.astype(Config.DTYPE)

    def fit_transform(self, X):
        """
        Fit to data, then transform it.
        """
        return self.fit(X).transform(X)


def get_preprocessed_data(load_cached_data=True):
    """
    Loads data, extracts geometric features, merges, cleans, and applies
    the high-precision preprocessing pipeline.

    Args:
        load_cached_data (bool): If True, attempts to load processed arrays from disk.

    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, test_ids, classes)
    """
    Config.setup()

    # Define cache filenames
    cache_files = {
        "X_train": "X_train_processed.npy",
        "y_train": "y_train_processed.npy",
        "X_val": "X_val_processed.npy",
        "y_val": "y_val_processed.npy",
        "X_test": "X_test_processed.npy",
        "test_ids": "test_ids.npy",
        "classes": "classes.npy",
    }

    # Check if all cache files exist
    all_cached = all(
        os.path.exists(Config.get_cache_path(f)) for f in cache_files.values()
    )

    if load_cached_data and all_cached:
        print("Loading preprocessed data from cache...")
        try:
            X_train = np.load(Config.get_cache_path(cache_files["X_train"]))
            y_train = np.load(Config.get_cache_path(cache_files["y_train"]))
            X_val = np.load(Config.get_cache_path(cache_files["X_val"]))
            y_val = np.load(Config.get_cache_path(cache_files["y_val"]))
            X_test = np.load(Config.get_cache_path(cache_files["X_test"]))
            test_ids = np.load(Config.get_cache_path(cache_files["test_ids"]))
            classes = np.load(
                Config.get_cache_path(cache_files["classes"]), allow_pickle=True
            )
            return X_train, y_train, X_val, y_val, X_test, test_ids, classes
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing from scratch...")

    print("Starting data preprocessing pipeline...")

    # 1. Load Metadata
    if not os.path.exists(Config.TRAIN_DATA_PATH):
        raise FileNotFoundError(f"Train metadata not found at {Config.TRAIN_DATA_PATH}")

    df_train_meta = pd.read_csv(Config.TRAIN_DATA_PATH)
    df_val_meta = pd.read_csv(Config.VAL_DATA_PATH)
    df_test_meta = pd.read_csv(Config.TEST_DATA_PATH)

    if Config.DEBUG:
        print(f"DEBUG MODE: Subsampling data to {Config.DEBUG_SAMPLE_SIZE} rows.")
        df_train_meta = df_train_meta.iloc[: Config.DEBUG_SAMPLE_SIZE]
        df_val_meta = df_val_meta.iloc[: Config.DEBUG_SAMPLE_SIZE]
        df_test_meta = df_test_meta.iloc[: Config.DEBUG_SAMPLE_SIZE]

    # 2. Extract/Load Geometric Features
    # This step uses the feature_extraction module which handles its own caching
    df_train_geo = process_dataset(
        Config.TRAIN_DATA_PATH, "train", load_cached_data=load_cached_data
    )
    df_val_geo = process_dataset(
        Config.VAL_DATA_PATH, "val", load_cached_data=load_cached_data
    )
    df_test_geo = process_dataset(
        Config.TEST_DATA_PATH, "test", load_cached_data=load_cached_data
    )

    # 3. Merge Datasets
    # Merge on 'id'. Metadata is the left table to preserve labels.
    print("Merging tabular and geometric features...")
    df_train = pd.merge(df_train_meta, df_train_geo, on="id", how="left")
    df_val = pd.merge(df_val_meta, df_val_geo, on="id", how="left")
    df_test = pd.merge(df_test_meta, df_test_geo, on="id", how="left")

    # 4. Feature Selection & Ordering
    # Exclude non-feature columns
    exclude_cols = ["id", "species", "file_path"]
    feature_cols = [c for c in df_train.columns if c not in exclude_cols]

    # Sort alphabetically for deterministic behavior
    feature_cols = sorted(feature_cols)
    print(f"Total features selected: {len(feature_cols)}")

    # Extract X matrices (raw)
    X_train_raw = df_train[feature_cols].values.astype(Config.DTYPE)
    X_val_raw = df_val[feature_cols].values.astype(Config.DTYPE)
    X_test_raw = df_test[feature_cols].values.astype(Config.DTYPE)

    # Extract Targets
    y_train_raw = df_train["species"].values
    y_val_raw = df_val["species"].values
    test_ids = df_test["id"].values

    # 5. Encode Targets
    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)
    # Handle potential unseen classes in validation (though stratification should prevent this)
    # We map validation labels using the encoder fitted on train
    try:
        y_val = le.transform(y_val_raw)
    except ValueError as e:
        print(f"Warning: Validation set contains classes not in training set. {e}")
        # Fallback: filter or handle. For this task, we assume consistent classes.
        # We will re-fit on combined for safety if strictly necessary, but standard practice is fit on train.
        # Given the stratification in metadata generation, this should be safe.
        y_val = le.transform(y_val_raw)

    classes = le.classes_

    # 6. Apply High-Precision Pipeline
    print("Fitting HighPrecisionPipeline on Training Data...")
    pipeline = HighPrecisionPipeline()

    # Fit on Train
    pipeline.fit(X_train_raw)

    # Transform all sets
    print("Transforming feature matrices...")
    X_train = pipeline.transform(X_train_raw)
    X_val = pipeline.transform(X_val_raw)
    X_test = pipeline.transform(X_test_raw)

    # 7. Cache Results
    print("Caching processed data...")
    np.save(Config.get_cache_path(cache_files["X_train"]), X_train)
    np.save(Config.get_cache_path(cache_files["y_train"]), y_train)
    np.save(Config.get_cache_path(cache_files["X_val"]), X_val)
    np.save(Config.get_cache_path(cache_files["y_val"]), y_val)
    np.save(Config.get_cache_path(cache_files["X_test"]), X_test)
    np.save(Config.get_cache_path(cache_files["test_ids"]), test_ids)
    np.save(Config.get_cache_path(cache_files["classes"]), classes)

    print("Preprocessing complete.")
    return X_train, y_train, X_val, y_val, X_test, test_ids, classes
