import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from library.feature_extraction import get_feature_views
from library.utils import set_seed

# Directory for caching processed data
CACHE_DIR = "./working/idea_69"
os.makedirs(CACHE_DIR, exist_ok=True)

# =============================================================================
# Transformer Classes
# =============================================================================


class GlobalRotationalTransformer(BaseEstimator, TransformerMixin):
    """
    Implements the Rotational View:
    PowerTransformer -> PCA(whiten=False) -> PowerTransformer.

    Aligns the data with its principal axes to optimize for LDA/QDA,
    ensuring the manifold is Gaussianized without the noise amplification
    associated with whitening.
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.pipeline = None

    def fit(self, X, y=None):
        # Construct pipeline
        # 1. Stabilize variance
        pt1 = PowerTransformer(method="yeo-johnson", standardize=True)
        # 2. Rotate (Full Rank, No Whitening)
        pca = PCA(n_components=None, whiten=False, random_state=self.random_state)
        # 3. Re-Gaussianize post-rotation
        pt2 = PowerTransformer(method="yeo-johnson", standardize=True)

        self.pipeline = Pipeline([("pt1", pt1), ("pca", pca), ("pt2", pt2)])

        self.pipeline.fit(X, y)
        return self

    def transform(self, X):
        if self.pipeline is None:
            raise RuntimeError("Transformer must be fitted before calling transform.")
        return self.pipeline.transform(X).astype(np.float64)


class StratifiedRotationalTransformer(BaseEstimator, TransformerMixin):
    """
    Splits the global feature vector into Margin, Shape, and Texture components.
    Applies GlobalRotationalTransformer to each component independently.
    Concatenates the results.

    Assumes input X has 192 columns:
    - 0-63: Margin
    - 64-127: Shape
    - 128-191: Texture
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.transformer_margin = None
        self.transformer_shape = None
        self.transformer_texture = None

    def fit(self, X, y=None):
        # Validate shape
        if X.shape[1] != 192:
            raise ValueError(f"Expected 192 features, got {X.shape[1]}")

        # Slice views
        X_margin = X[:, 0:64]
        X_shape = X[:, 64:128]
        X_texture = X[:, 128:192]

        # Initialize and fit independent transformers
        self.transformer_margin = GlobalRotationalTransformer(
            random_state=self.random_state
        )
        self.transformer_shape = GlobalRotationalTransformer(
            random_state=self.random_state
        )
        self.transformer_texture = GlobalRotationalTransformer(
            random_state=self.random_state
        )

        self.transformer_margin.fit(X_margin)
        self.transformer_shape.fit(X_shape)
        self.transformer_texture.fit(X_texture)

        return self

    def transform(self, X):
        if self.transformer_margin is None:
            raise RuntimeError("Transformer must be fitted before calling transform.")

        if X.shape[1] != 192:
            raise ValueError(f"Expected 192 features, got {X.shape[1]}")

        # Slice
        X_margin = X[:, 0:64]
        X_shape = X[:, 64:128]
        X_texture = X[:, 128:192]

        # Transform
        X_m_trans = self.transformer_margin.transform(X_margin)
        X_s_trans = self.transformer_shape.transform(X_shape)
        X_t_trans = self.transformer_texture.transform(X_texture)

        # Concatenate
        return np.hstack([X_m_trans, X_s_trans, X_t_trans]).astype(np.float64)


class RobustTransformer(BaseEstimator, TransformerMixin):
    """
    Wraps QuantileTransformer with specific settings for robust rank-based normalization.
    Constrains n_quantiles to 50 to prevent overfitting to noise in tails.
    """

    def __init__(self, n_quantiles=50, random_state=42):
        self.n_quantiles = n_quantiles
        self.random_state = random_state
        self.transformer = None

    def fit(self, X, y=None):
        self.transformer = QuantileTransformer(
            output_distribution="normal",
            n_quantiles=self.n_quantiles,
            random_state=self.random_state,
        )
        self.transformer.fit(X, y)
        return self

    def transform(self, X):
        if self.transformer is None:
            raise RuntimeError("Transformer must be fitted before calling transform.")
        return self.transformer.transform(X).astype(np.float64)


# =============================================================================
# Processing Logic
# =============================================================================


def get_preprocessed_data(strategy: str, load_cached_data: bool = True):
    """
    Orchestrates the loading, fitting, transforming, and caching of data
    based on the requested strategy.

    Strategies:
    - 'global_marginal': PowerTransformer on Global view.
    - 'global_rotational': GlobalRotationalTransformer on Global view.
    - 'global_robust': RobustTransformer on Global view.
    - 'stratified_rotational': StratifiedRotationalTransformer on Global view.
    - 'morph_physical': PowerTransformer on Morphometrics view.

    Returns:
        dict: {
            "X_train": np.ndarray, "y_train": np.ndarray,
            "X_val": np.ndarray, "y_val": np.ndarray,
            "X_test": np.ndarray, "ids_test": np.ndarray
        }
    """
    set_seed(42)

    # Define Cache Path
    cache_path = os.path.join(CACHE_DIR, f"preprocessed_{strategy}.npz")

    # 1. Try Loading from Cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            with np.load(cache_path, allow_pickle=True) as data:
                return {
                    "X_train": data["X_train"],
                    "y_train": data["y_train"],
                    "X_val": data["X_val"],
                    "y_val": data["y_val"],
                    "X_test": data["X_test"],
                    "ids_test": data["ids_test"],
                }
        except Exception:
            pass  # Fallback to compute

    # 2. Load Raw Data
    # We need all splits to perform consistent preprocessing (Fit on Train, Transform All)
    raw_train = get_feature_views("train", load_cached_data=load_cached_data)
    raw_val = get_feature_views("val", load_cached_data=load_cached_data)
    raw_test = get_feature_views("test", load_cached_data=load_cached_data)

    # Extract Targets and IDs
    y_train = raw_train["y"]
    y_val = raw_val["y"]
    ids_test = raw_test["ids"]

    # Select Input View and Transformer based on Strategy
    if strategy == "global_marginal":
        X_train_raw = raw_train["views"]["Global"]
        X_val_raw = raw_val["views"]["Global"]
        X_test_raw = raw_test["views"]["Global"]
        transformer = PowerTransformer(method="yeo-johnson", standardize=True)

    elif strategy == "global_rotational":
        X_train_raw = raw_train["views"]["Global"]
        X_val_raw = raw_val["views"]["Global"]
        X_test_raw = raw_test["views"]["Global"]
        transformer = GlobalRotationalTransformer(random_state=42)

    elif strategy == "global_robust":
        X_train_raw = raw_train["views"]["Global"]
        X_val_raw = raw_val["views"]["Global"]
        X_test_raw = raw_test["views"]["Global"]
        transformer = RobustTransformer(n_quantiles=50, random_state=42)

    elif strategy == "stratified_rotational":
        X_train_raw = raw_train["views"]["Global"]
        X_val_raw = raw_val["views"]["Global"]
        X_test_raw = raw_test["views"]["Global"]
        transformer = StratifiedRotationalTransformer(random_state=42)

    elif strategy == "morph_physical":
        X_train_raw = raw_train["views"]["Morphometrics"]
        X_val_raw = raw_val["views"]["Morphometrics"]
        X_test_raw = raw_test["views"]["Morphometrics"]
        transformer = PowerTransformer(method="yeo-johnson", standardize=True)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # 3. Fit on Train, Transform All
    # Ensure float64 input
    X_train_raw = X_train_raw.astype(np.float64)
    X_val_raw = X_val_raw.astype(np.float64)
    X_test_raw = X_test_raw.astype(np.float64)

    transformer.fit(X_train_raw)

    X_train_proc = transformer.transform(X_train_raw)
    X_val_proc = transformer.transform(X_val_raw)
    X_test_proc = transformer.transform(X_test_raw)

    # 4. Save to Cache
    np.savez(
        cache_path,
        X_train=X_train_proc,
        y_train=y_train,
        X_val=X_val_proc,
        y_val=y_val,
        X_test=X_test_proc,
        ids_test=ids_test,
    )

    return {
        "X_train": X_train_proc,
        "y_train": y_train,
        "X_val": X_val_proc,
        "y_val": y_val,
        "X_test": X_test_proc,
        "ids_test": ids_test,
    }
