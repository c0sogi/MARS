import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from library.utils import set_seed
from library.features import DatasetLoader

# Constants
CACHE_DIR = "./working/idea_70"


class GlobalManifoldTransformer:
    """
    Transforms the global feature set (192 features) using a
    Power -> PCA (Full Rank) -> Power pipeline.
    """

    def __init__(self):
        self.pipeline = Pipeline(
            [
                ("pt1", PowerTransformer()),
                ("pca", PCA(whiten=False)),  # Full rank, no dimensionality reduction
                ("pt2", PowerTransformer()),
            ]
        )
        self.feature_cols = None

    def fit(self, X):
        # Select the 192 provided features (Margin, Shape, Texture)
        self.feature_cols = [
            c for c in X.columns if "margin" in c or "shape" in c or "texture" in c
        ]
        X_subset = X[self.feature_cols].astype(np.float64)
        self.pipeline.fit(X_subset)
        return self

    def transform(self, X):
        if self.feature_cols is None:
            raise ValueError("Transformer not fitted.")
        X_subset = X[self.feature_cols].astype(np.float64)
        return self.pipeline.transform(X_subset)


class StratifiedManifoldTransformer:
    """
    Splits features into Margin, Shape, and Texture groups.
    Applies independent Power -> PCA -> Power pipelines to each group.
    Concatenates the results.
    """

    def __init__(self):
        self.margin_pipe = Pipeline(
            [
                ("pt1", PowerTransformer()),
                ("pca", PCA(whiten=False)),
                ("pt2", PowerTransformer()),
            ]
        )
        self.shape_pipe = Pipeline(
            [
                ("pt1", PowerTransformer()),
                ("pca", PCA(whiten=False)),
                ("pt2", PowerTransformer()),
            ]
        )
        self.texture_pipe = Pipeline(
            [
                ("pt1", PowerTransformer()),
                ("pca", PCA(whiten=False)),
                ("pt2", PowerTransformer()),
            ]
        )

        self.margin_cols = []
        self.shape_cols = []
        self.texture_cols = []

    def fit(self, X):
        self.margin_cols = [c for c in X.columns if "margin" in c]
        self.shape_cols = [c for c in X.columns if "shape" in c]
        self.texture_cols = [c for c in X.columns if "texture" in c]

        # Fit independent pipelines
        self.margin_pipe.fit(X[self.margin_cols].astype(np.float64))
        self.shape_pipe.fit(X[self.shape_cols].astype(np.float64))
        self.texture_pipe.fit(X[self.texture_cols].astype(np.float64))
        return self

    def transform(self, X):
        if not self.margin_cols:
            raise ValueError("Transformer not fitted.")

        m_trans = self.margin_pipe.transform(X[self.margin_cols].astype(np.float64))
        s_trans = self.shape_pipe.transform(X[self.shape_cols].astype(np.float64))
        t_trans = self.texture_pipe.transform(X[self.texture_cols].astype(np.float64))

        return np.concatenate([m_trans, s_trans, t_trans], axis=1)


class PhysicalManifoldTransformer:
    """
    Transforms Polarity-Corrected Morphometrics using a
    Power -> Polynomial(degree=2) -> Power pipeline.
    Captures non-linear physical constraints.
    """

    def __init__(self):
        self.pipeline = Pipeline(
            [
                ("pt1", PowerTransformer()),
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                ("pt2", PowerTransformer()),
            ]
        )
        self.feature_cols = None

    def fit(self, X):
        # Select morphometric features (Hu Moments + Geometric Scalars)
        # These are columns that are NOT margin, shape, or texture
        self.feature_cols = [
            c
            for c in X.columns
            if c.startswith("hu_")
            or c in ["aspect_ratio", "solidity", "extent", "eccentricity"]
        ]
        X_subset = X[self.feature_cols].astype(np.float64)
        self.pipeline.fit(X_subset)
        return self

    def transform(self, X):
        if self.feature_cols is None:
            raise ValueError("Transformer not fitted.")
        X_subset = X[self.feature_cols].astype(np.float64)
        return self.pipeline.transform(X_subset)


class Preprocessor:
    """
    Manages data loading, transformation, and caching.
    """

    def __init__(self):
        self.global_transformer = GlobalManifoldTransformer()
        self.stratified_transformer = StratifiedManifoldTransformer()
        self.physical_transformer = PhysicalManifoldTransformer()

    def get_data(self, load_cached_data=True):
        """
        Loads data, applies transformations, and returns a dictionary of datasets.
        Uses caching to avoid re-computation.
        """
        set_seed(42)
        os.makedirs(CACHE_DIR, exist_ok=True)

        # Define cache file paths
        files = {
            "X_train_global": os.path.join(CACHE_DIR, "X_train_global.npy"),
            "X_val_global": os.path.join(CACHE_DIR, "X_val_global.npy"),
            "X_test_global": os.path.join(CACHE_DIR, "X_test_global.npy"),
            "X_train_stratified": os.path.join(CACHE_DIR, "X_train_stratified.npy"),
            "X_val_stratified": os.path.join(CACHE_DIR, "X_val_stratified.npy"),
            "X_test_stratified": os.path.join(CACHE_DIR, "X_test_stratified.npy"),
            "X_train_physical": os.path.join(CACHE_DIR, "X_train_physical.npy"),
            "X_val_physical": os.path.join(CACHE_DIR, "X_val_physical.npy"),
            "X_test_physical": os.path.join(CACHE_DIR, "X_test_physical.npy"),
            "y_train": os.path.join(CACHE_DIR, "y_train.npy"),
            "y_val": os.path.join(CACHE_DIR, "y_val.npy"),
            "test_ids": os.path.join(CACHE_DIR, "test_ids.npy"),
        }

        # Check if all cache files exist
        all_exist = all(os.path.exists(f) for f in files.values())

        if load_cached_data and all_exist:
            print("Loading preprocessed data from cache...")
            data = {k: np.load(v, allow_pickle=True) for k, v in files.items()}
            return data

        print("Preprocessing data from scratch...")
        loader = DatasetLoader()
        # load_data handles caching of the raw merged data
        X_train_raw, y_train, X_val_raw, y_val, X_test_raw, test_ids = loader.load_data(
            load_cached_data=load_cached_data
        )

        # Fit transformers on Training set only to prevent leakage
        print("Fitting Global Transformer...")
        self.global_transformer.fit(X_train_raw)

        print("Fitting Stratified Transformer...")
        self.stratified_transformer.fit(X_train_raw)

        print("Fitting Physical Transformer...")
        self.physical_transformer.fit(X_train_raw)

        # Transform all sets
        data = {}
        data["y_train"] = y_train
        data["y_val"] = y_val
        data["test_ids"] = test_ids

        # Global View
        data["X_train_global"] = self.global_transformer.transform(X_train_raw)
        data["X_val_global"] = self.global_transformer.transform(X_val_raw)
        data["X_test_global"] = self.global_transformer.transform(X_test_raw)

        # Stratified View
        data["X_train_stratified"] = self.stratified_transformer.transform(X_train_raw)
        data["X_val_stratified"] = self.stratified_transformer.transform(X_val_raw)
        data["X_test_stratified"] = self.stratified_transformer.transform(X_test_raw)

        # Physical View
        data["X_train_physical"] = self.physical_transformer.transform(X_train_raw)
        data["X_val_physical"] = self.physical_transformer.transform(X_val_raw)
        data["X_test_physical"] = self.physical_transformer.transform(X_test_raw)

        # Save to cache
        for k, v in data.items():
            np.save(files[k], v)

        print(f"Preprocessed data saved to {CACHE_DIR}")
        return data
