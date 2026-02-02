import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer, StandardScaler, LabelEncoder
from library.utils import set_seed

# Set global seed
set_seed(42)


class TreePreprocessor(BaseEstimator, TransformerMixin):
    """
    Identity transformer for Tree-based models (e.g., LightGBM).
    Returns the input features unchanged.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


class LinearKernelPreprocessor(BaseEstimator, TransformerMixin):
    """
    Preprocessor for Linear (LDA) and Kernel (SVM) models.
    Chains PowerTransformer (Yeo-Johnson) and StandardScaler.
    """

    def __init__(self):
        # standardize=False in PowerTransformer because we will apply StandardScaler explicitly
        self.pt = PowerTransformer(method="yeo-johnson", standardize=False)
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        # Fit the PowerTransformer
        self.pt.fit(X)
        # Transform to get intermediate state
        X_trans = self.pt.transform(X)
        # Fit the StandardScaler on the transformed data
        self.scaler.fit(X_trans)
        return self

    def transform(self, X):
        # Apply transformations in sequence
        X_pt = self.pt.transform(X)
        X_scaled = self.scaler.transform(X_pt)
        return X_scaled


def get_preprocessed_data(model_type="tree", load_cached_data=True):
    """
    Loads data, applies preprocessing based on model_type, and handles caching.

    Args:
        model_type (str): 'tree' or 'linear_kernel'.
        load_cached_data (bool): Whether to load from cache if available.

    Returns:
        tuple: (X_train, y_train), (X_val, y_val), (X_test, test_ids), classes
    """
    working_dir = "./working/idea_2"
    os.makedirs(working_dir, exist_ok=True)

    # Define cache file paths
    cache_files = {
        "X_train": os.path.join(working_dir, f"X_train_{model_type}.npy"),
        "y_train": os.path.join(working_dir, "y_train.npy"),
        "X_val": os.path.join(working_dir, f"X_val_{model_type}.npy"),
        "y_val": os.path.join(working_dir, "y_val.npy"),
        "X_test": os.path.join(working_dir, f"X_test_{model_type}.npy"),
        "test_ids": os.path.join(working_dir, "test_ids.npy"),
        "classes": os.path.join(working_dir, "classes.npy"),
    }

    # Check if all cache files exist
    cache_exists = all(os.path.exists(path) for path in cache_files.values())

    if load_cached_data and cache_exists:
        print(f"Loading cached data for model_type='{model_type}'...")
        X_train = np.load(cache_files["X_train"])
        y_train = np.load(cache_files["y_train"])
        X_val = np.load(cache_files["X_val"])
        y_val = np.load(cache_files["y_val"])
        X_test = np.load(cache_files["X_test"])
        test_ids = np.load(cache_files["test_ids"])
        classes = np.load(cache_files["classes"], allow_pickle=True)

        return (X_train, y_train), (X_val, y_val), (X_test, test_ids), classes

    print(f"Processing data for model_type='{model_type}'...")

    # Load metadata
    df_train = pd.read_csv("./metadata/train.csv")
    df_val = pd.read_csv("./metadata/val.csv")
    df_test = pd.read_csv("./metadata/test.csv")

    # Identify feature columns (exclude id, species, file_path)
    # Features start with margin, shape, or texture
    feature_cols = [
        c for c in df_train.columns if c.startswith(("margin", "shape", "texture"))
    ]

    # Extract features
    X_train_raw = df_train[feature_cols].values.astype(np.float32)
    X_val_raw = df_val[feature_cols].values.astype(np.float32)
    X_test_raw = df_test[feature_cols].values.astype(np.float32)

    # Extract targets and IDs
    y_train_raw = df_train["species"].values
    y_val_raw = df_val["species"].values
    test_ids = df_test["id"].values

    # Encode targets
    le = LabelEncoder()
    # Fit on training species. Stratified split ensures coverage, but safe to fit on combined if needed.
    # Here we fit on train as per standard practice.
    y_train = le.fit_transform(y_train_raw)
    y_val = le.transform(y_val_raw)
    classes = le.classes_

    # Select Preprocessor
    if model_type == "tree":
        preprocessor = TreePreprocessor()
    elif model_type == "linear_kernel":
        preprocessor = LinearKernelPreprocessor()
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Fit on Train, Transform All
    preprocessor.fit(X_train_raw)
    X_train = preprocessor.transform(X_train_raw)
    X_val = preprocessor.transform(X_val_raw)
    X_test = preprocessor.transform(X_test_raw)

    # Save to cache
    np.save(cache_files["X_train"], X_train)
    np.save(cache_files["y_train"], y_train)
    np.save(cache_files["X_val"], X_val)
    np.save(cache_files["y_val"], y_val)
    np.save(cache_files["X_test"], X_test)
    np.save(cache_files["test_ids"], test_ids)
    np.save(cache_files["classes"], classes)

    return (X_train, y_train), (X_val, y_val), (X_test, test_ids), classes
