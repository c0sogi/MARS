import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer
from sklearn.decomposition import PCA
from library.config import (
    WORKING_DIR,
    FLOAT_PRECISION,
    TOPOLOGY_MARGINAL,
    TOPOLOGY_ROTATIONAL,
)


class MarginalTopology(BaseEstimator, TransformerMixin):
    """
    Topology A: Marginal Parametric Anchors.
    Applies a standard PowerTransformer (Yeo-Johnson) to Gaussianize features independently.
    """

    def __init__(
        self,
        method=TOPOLOGY_MARGINAL["method"],
        standardize=TOPOLOGY_MARGINAL["standardize"],
    ):
        self.method = method
        self.standardize = standardize
        self.pt = None

    def fit(self, X, y=None):
        # Enforce precision
        X = X.astype(FLOAT_PRECISION)
        self.pt = PowerTransformer(method=self.method, standardize=self.standardize)
        self.pt.fit(X)
        return self

    def transform(self, X):
        X = X.astype(FLOAT_PRECISION)
        X_trans = self.pt.transform(X)
        return X_trans.astype(FLOAT_PRECISION)


class RotationalTopology(BaseEstimator, TransformerMixin):
    """
    Topology B: Rotational Parametric Experts.
    Pipeline: PowerTransform -> PCA(whiten=False) -> PowerTransform.

    Mechanism:
    1. Stabilize marginals (PT).
    2. Rotate to principal axes (PCA without whitening).
    3. Re-Gaussianize principal components (PT).
    """

    def __init__(
        self,
        initial_pt_method=TOPOLOGY_ROTATIONAL["initial_pt_method"],
        pca_whiten=TOPOLOGY_ROTATIONAL["pca_whiten"],
        pca_components=TOPOLOGY_ROTATIONAL["pca_components"],
        final_pt_method=TOPOLOGY_ROTATIONAL["final_pt_method"],
    ):

        self.initial_pt_method = initial_pt_method
        self.pca_whiten = pca_whiten
        self.pca_components = pca_components
        self.final_pt_method = final_pt_method

        self.pt1 = None
        self.pca = None
        self.pt2 = None

    def fit(self, X, y=None):
        X = X.astype(FLOAT_PRECISION)

        # 1. Initial Marginal Gaussianization
        self.pt1 = PowerTransformer(method=self.initial_pt_method, standardize=True)
        X_pt1 = self.pt1.fit_transform(X)

        # 2. Rotation (PCA)
        # Note: We do NOT whiten here to avoid noise amplification on small eigenvalues
        self.pca = PCA(n_components=self.pca_components, whiten=self.pca_whiten)
        X_pca = self.pca.fit_transform(X_pt1)

        # 3. Final Component Gaussianization
        self.pt2 = PowerTransformer(method=self.final_pt_method, standardize=True)
        self.pt2.fit(X_pca)

        return self

    def transform(self, X):
        X = X.astype(FLOAT_PRECISION)

        # Apply pipeline
        X_pt1 = self.pt1.transform(X)
        X_pca = self.pca.transform(X_pt1)
        X_final = self.pt2.transform(X_pca)

        return X_final.astype(FLOAT_PRECISION)


def apply_topology(
    X_train, X_val, X_test, topology_name, cache_name, load_cached_data=True
):
    """
    Applies the specified Gaussianization topology to the data splits.
    Fits on X_train, transforms X_train, X_val, and X_test.
    Handles caching of the transformed numpy arrays.

    Args:
        X_train, X_val, X_test (np.ndarray): Input feature matrices.
        topology_name (str): 'marginal' or 'rotational'.
        cache_name (str): Unique identifier for caching (e.g., 'global_view', 'combined_view').
        load_cached_data (bool): Whether to load from cache if available.

    Returns:
        tuple: (X_train_trans, X_val_trans, X_test_trans) as float64 numpy arrays.
    """
    # Construct cache paths
    cache_train = os.path.join(
        WORKING_DIR, f"transform_{topology_name}_{cache_name}_train.npy"
    )
    cache_val = os.path.join(
        WORKING_DIR, f"transform_{topology_name}_{cache_name}_val.npy"
    )
    cache_test = os.path.join(
        WORKING_DIR, f"transform_{topology_name}_{cache_name}_test.npy"
    )

    # 1. Try Loading Cache
    if load_cached_data:
        if (
            os.path.exists(cache_train)
            and os.path.exists(cache_val)
            and os.path.exists(cache_test)
        ):
            # print(f"Loading cached transforms for {topology_name} - {cache_name}...")
            try:
                X_train_t = np.load(cache_train).astype(FLOAT_PRECISION)
                X_val_t = np.load(cache_val).astype(FLOAT_PRECISION)
                X_test_t = np.load(cache_test).astype(FLOAT_PRECISION)
                return X_train_t, X_val_t, X_test_t
            except Exception as e:
                print(f"Cache load failed ({e}). Recomputing...")

    # 2. Compute Transforms
    # print(f"Computing transforms for {topology_name} - {cache_name}...")

    # Instantiate Topology
    if topology_name == "marginal":
        transformer = MarginalTopology()
    elif topology_name == "rotational":
        transformer = RotationalTopology()
    else:
        raise ValueError(f"Unknown topology: {topology_name}")

    # Fit on Train
    transformer.fit(X_train)

    # Transform all splits
    X_train_t = transformer.transform(X_train)
    X_val_t = transformer.transform(X_val)
    X_test_t = transformer.transform(X_test)

    # 3. Save to Cache
    try:
        np.save(cache_train, X_train_t)
        np.save(cache_val, X_val_t)
        np.save(cache_test, X_test_t)
    except Exception as e:
        print(f"Warning: Failed to save transforms to cache: {e}")

    return X_train_t, X_val_t, X_test_t
