import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from library.utils import set_seed


class FeatureProcessor:
    """
    Manages feature preprocessing and transformation logic.
    Creates a globally scaled view using StandardScaler (for Linear Models).
    Optionally applies PCA.
    """

    def __init__(self, n_pca_components=None, random_state=42):
        self.random_state = random_state
        self.n_pca_components = n_pca_components
        self.scaler = StandardScaler()
        self.pca = None
        if self.n_pca_components:
            self.pca = PCA(
                n_components=self.n_pca_components, random_state=self.random_state
            )

    def fit(self, X):
        """
        Fits the StandardScaler and optional PCA.
        """
        # Fit scaler on raw data
        self.scaler.fit(X)
        if self.pca:
            X_scaled = self.scaler.transform(X)
            self.pca.fit(X_scaled)
        return self

    def transform(self, X):
        """
        Transforms the data: Scaled (and optional PCA).
        """
        X_scaled = self.scaler.transform(X)
        if self.pca:
            X_pca = self.pca.transform(X_scaled)
            return X_scaled, X_pca
        return X_scaled

    def fit_transform(self, X):
        """
        Fits and transforms in one step.
        """
        X_scaled = self.scaler.fit_transform(X)
        if self.pca:
            X_pca = self.pca.fit_transform(X_scaled)
            return X_scaled, X_pca
        return X_scaled


def process_data(
    metadata_dir="./metadata",
    cache_dir="./working/idea_3",
    load_cached_data=True,
    n_pca_components=None,
    random_state=42,
):
    """
    Loads data, processes it using FeatureProcessor, and handles caching.
    Combines Train and Validation sets for full training.

    Returns:
        X_train_scaled, X_train_pca, y_train, X_test_scaled, X_test_pca, test_ids, classes
    """
    set_seed(random_state)
    os.makedirs(cache_dir, exist_ok=True)

    # Define cache paths
    cache_paths = {
        "X_train_scaled": os.path.join(cache_dir, "X_train_scaled.npy"),
        "X_train_pca": os.path.join(cache_dir, "X_train_pca.npy"),
        "y_train": os.path.join(cache_dir, "y_train.npy"),
        "X_test_scaled": os.path.join(cache_dir, "X_test_scaled.npy"),
        "X_test_pca": os.path.join(cache_dir, "X_test_pca.npy"),
        "test_ids": os.path.join(cache_dir, "test_ids.npy"),
        "classes": os.path.join(cache_dir, "classes.npy"),
    }

    # Check if cache exists
    required_keys = [
        "X_train_scaled",
        "y_train",
        "X_test_scaled",
        "test_ids",
        "classes",
    ]
    if n_pca_components:
        required_keys.extend(["X_train_pca", "X_test_pca"])

    cache_exists = all(os.path.exists(cache_paths[k]) for k in required_keys)

    if load_cached_data and cache_exists:
        print(f"Loading cached processed data from {cache_dir}...")
        X_train_scaled = np.load(cache_paths["X_train_scaled"])
        y_train = np.load(cache_paths["y_train"])
        X_test_scaled = np.load(cache_paths["X_test_scaled"])
        test_ids = np.load(cache_paths["test_ids"])
        classes = np.load(cache_paths["classes"], allow_pickle=True)

        X_train_pca = None
        X_test_pca = None
        if n_pca_components:
            X_train_pca = np.load(cache_paths["X_train_pca"])
            X_test_pca = np.load(cache_paths["X_test_pca"])

        return (
            X_train_scaled,
            X_train_pca,
            y_train,
            X_test_scaled,
            X_test_pca,
            test_ids,
            classes,
        )

    print("Processing data from scratch...")

    # Load metadata
    train_df = pd.read_csv(os.path.join(metadata_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(metadata_dir, "val.csv"))
    test_df = pd.read_csv(os.path.join(metadata_dir, "test.csv"))

    # Combine Train and Val
    full_train_df = pd.concat([train_df, val_df], axis=0, ignore_index=True)

    # Identify feature columns
    feature_cols = [
        c for c in full_train_df.columns if c.startswith(("margin", "shape", "texture"))
    ]

    # Extract raw data
    X_train_raw = full_train_df[feature_cols].values
    y_train_raw = full_train_df["species"].values
    X_test_raw = test_df[feature_cols].values
    test_ids = test_df["id"].values

    # Encode Labels
    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)
    classes = le.classes_

    # Initialize and Apply FeatureProcessor
    processor = FeatureProcessor(
        n_pca_components=n_pca_components, random_state=random_state
    )

    # Fit and Transform Train
    if n_pca_components:
        X_train_scaled, X_train_pca = processor.fit_transform(X_train_raw)
        X_test_scaled, X_test_pca = processor.transform(X_test_raw)
    else:
        X_train_scaled = processor.fit_transform(X_train_raw)
        X_test_scaled = processor.transform(X_test_raw)
        X_train_pca = None
        X_test_pca = None

    # Save to cache
    print(f"Saving processed data to {cache_dir}...")
    np.save(cache_paths["X_train_scaled"], X_train_scaled)
    np.save(cache_paths["y_train"], y_train)
    np.save(cache_paths["X_test_scaled"], X_test_scaled)
    np.save(cache_paths["test_ids"], test_ids)
    np.save(cache_paths["classes"], classes)

    if n_pca_components:
        np.save(cache_paths["X_train_pca"], X_train_pca)
        np.save(cache_paths["X_test_pca"], X_test_pca)

    return (
        X_train_scaled,
        X_train_pca,
        y_train,
        X_test_scaled,
        X_test_pca,
        test_ids,
        classes,
    )
