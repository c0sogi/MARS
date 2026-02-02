import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, StandardScaler
from library.config import (
    METADATA_DIR,
    WORKING_DIR,
    FEATURE_COLUMNS,
    FLOAT_TYPE,
    SEED,
)


class RobustPreprocessor:
    """
    Encapsulates the inductive preprocessing pipeline:
    1. Cast to float64
    2. Yeo-Johnson Power Transformation (without standardization)
    3. Standard Scaling

    Ensures all operations maintain float64 precision.
    """

    def __init__(self):
        self.pt = PowerTransformer(method="yeo-johnson", standardize=False)
        self.scaler = StandardScaler()

    def fit(self, X):
        """Fits the transformers on the provided data (Training set)."""
        # Ensure input is float64
        X = X.astype(FLOAT_TYPE)

        # Fit PowerTransformer
        self.pt.fit(X)

        # Transform to get intermediate state for scaler fitting
        X_pt = self.pt.transform(X)

        # Fit StandardScaler
        self.scaler.fit(X_pt)
        return self

    def transform(self, X):
        """Applies the learned transformations to the data."""
        # Ensure input is float64
        X = X.astype(FLOAT_TYPE)

        # Apply PowerTransformer
        X_pt = self.pt.transform(X)

        # Apply StandardScaler
        X_scaled = self.scaler.transform(X_pt)

        # Return as DataFrame if input was DataFrame to preserve metadata
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(
                X_scaled, columns=X.columns, index=X.index, dtype=FLOAT_TYPE
            )
        return X_scaled.astype(FLOAT_TYPE)

    def fit_transform(self, X):
        """Fits and transforms the data."""
        self.fit(X)
        return self.transform(X)


def load_dataset(load_cached_data=True, debug=False, debug_sample_size=100):
    """
    Loads the dataset, applying the inductive preprocessing pipeline.
    Uses caching to speed up subsequent runs.

    Args:
        load_cached_data (bool): If True, attempts to load from ./working/idea_31/
        debug (bool): If True, loads a small subset of the raw data for debugging.
        debug_sample_size (int): Number of rows to use in debug mode.

    Returns:
        X_train (pd.DataFrame): Processed training features.
        y_train (np.ndarray): Training labels.
        X_val (pd.DataFrame): Processed validation features.
        y_val (np.ndarray): Validation labels.
        X_test (pd.DataFrame): Processed test features.
        ids_test (np.ndarray): Test image IDs.
        classes (np.ndarray): Unique class names found in training set.
    """
    # Define cache paths
    cache_dir = WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    files = {
        "X_train": os.path.join(cache_dir, "X_train.parquet"),
        "y_train": os.path.join(cache_dir, "y_train.npy"),
        "X_val": os.path.join(cache_dir, "X_val.parquet"),
        "y_val": os.path.join(cache_dir, "y_val.npy"),
        "X_test": os.path.join(cache_dir, "X_test.parquet"),
        "ids_test": os.path.join(cache_dir, "ids_test.npy"),
        "classes": os.path.join(cache_dir, "classes.npy"),
    }

    # 1. Attempt to load from cache
    if load_cached_data and not debug:
        all_exist = all(os.path.exists(p) for p in files.values())
        if all_exist:
            print(f"Loading cached dataset from {cache_dir}...")
            X_train = pd.read_parquet(files["X_train"])
            y_train = np.load(files["y_train"], allow_pickle=True)
            X_val = pd.read_parquet(files["X_val"])
            y_val = np.load(files["y_val"], allow_pickle=True)
            X_test = pd.read_parquet(files["X_test"])
            ids_test = np.load(files["ids_test"], allow_pickle=True)
            classes = np.load(files["classes"], allow_pickle=True)
            return X_train, y_train, X_val, y_val, X_test, ids_test, classes
        else:
            print("Cache miss. Processing dataset from scratch...")

    # 2. Load Raw Data from Metadata
    print("Loading raw metadata CSVs...")
    train_path = os.path.join(METADATA_DIR, "train.csv")
    val_path = os.path.join(METADATA_DIR, "val.csv")
    test_path = os.path.join(METADATA_DIR, "test.csv")

    if not all(os.path.exists(p) for p in [train_path, val_path, test_path]):
        raise FileNotFoundError(
            "Metadata files not found. Ensure metadata generation ran successfully."
        )

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    # 3. Debug Sampling
    if debug:
        print(f"DEBUG MODE: Sampling {debug_sample_size} rows per split.")
        df_train = df_train.iloc[:debug_sample_size]
        df_val = df_val.iloc[:debug_sample_size]
        df_test = df_test.iloc[:debug_sample_size]

    # 4. Extract Features and Targets
    # Using deterministic FEATURE_COLUMNS from config
    print("Extracting features and targets...")
    X_train_raw = df_train[FEATURE_COLUMNS]
    y_train = df_train["species"].values

    X_val_raw = df_val[FEATURE_COLUMNS]
    y_val = df_val["species"].values

    X_test_raw = df_test[FEATURE_COLUMNS]
    ids_test = df_test["id"].values

    # Identify classes
    classes = np.unique(y_train)

    # 5. Inductive Preprocessing
    print(
        "Applying Inductive Preprocessing (Yeo-Johnson + StandardScaler) in float64..."
    )
    preprocessor = RobustPreprocessor()

    # Fit ONLY on Train
    X_train = preprocessor.fit_transform(X_train_raw)

    # Transform Val and Test
    X_val = preprocessor.transform(X_val_raw)
    X_test = preprocessor.transform(X_test_raw)

    # 6. Save to Cache (only if not debugging)
    if not debug:
        print(f"Saving processed dataset to {cache_dir}...")
        X_train.to_parquet(files["X_train"])
        np.save(files["y_train"], y_train)

        X_val.to_parquet(files["X_val"])
        np.save(files["y_val"], y_val)

        X_test.to_parquet(files["X_test"])
        np.save(files["ids_test"], ids_test)

        np.save(files["classes"], classes)

    return X_train, y_train, X_val, y_val, X_test, ids_test, classes
