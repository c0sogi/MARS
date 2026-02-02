import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, BayesianRidge
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.decomposition import PCA

from library.config import Config


class ImageMetaPCA(BaseEstimator, TransformerMixin):
    """
    Custom Transformer that applies PCA only to the image embedding portion of the input,
    while preserving the binary metadata features as-is.

    Assumes the input X is a concatenation of [Image_Embeddings, Metadata].
    The last `meta_count` columns are treated as metadata.
    """

    def __init__(self, n_components=64, meta_count=12, random_state=42):
        self.n_components = n_components
        self.meta_count = meta_count
        self.random_state = random_state
        self.pca = None

    def fit(self, X, y=None):
        # Split image features (all columns except the last meta_count)
        # and metadata (last meta_count columns)
        if self.meta_count > 0:
            X_img = X[:, : -self.meta_count]
        else:
            X_img = X

        # Adjust n_components to not exceed sample size or feature count
        # This prevents errors when n_samples < n_components (e.g., in debug mode)
        n_samples, n_features = X_img.shape
        n_components = min(self.n_components, n_samples, n_features)

        self.pca = PCA(n_components=n_components, random_state=self.random_state)
        self.pca.fit(X_img)
        return self

    def transform(self, X):
        if self.meta_count > 0:
            X_img = X[:, : -self.meta_count]
            X_meta = X[:, -self.meta_count :]

            # Apply PCA to image features
            X_pca = self.pca.transform(X_img)

            # Concatenate reduced image features with raw metadata
            return np.hstack([X_pca, X_meta])
        else:
            return self.pca.transform(X)


def get_linear_expert(random_state=Config.SEED):
    """
    Constructs the Linear Expert: Ridge Regression.

    Pipeline:
    1. StandardScaler: Scales all inputs (embeddings + metadata).
    2. RidgeCV: Linear regression with built-in cross-validation for alpha selection.
    """
    # Note: RidgeCV does not accept a random_state as it is deterministic
    # for the default solver (closed-form solution).

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("regressor", RidgeCV(alphas=Config.RIDGE_ALPHAS)),
        ]
    )

    return pipeline


def get_kernel_expert(random_state=Config.SEED):
    """
    Constructs the Kernel Expert: Support Vector Regression (SVR).

    Pipeline:
    1. StandardScaler: Scales all inputs.
    2. SVR: RBF Kernel regression to capture non-linear manifolds.
    """
    # Note: SVR implementation in sklearn (libsvm) does not accept random_state.

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "regressor",
                SVR(
                    C=Config.SVR_C,
                    epsilon=Config.SVR_EPSILON,
                    cache_size=Config.SVR_CACHE_SIZE,
                ),
            ),
        ]
    )

    return pipeline


def get_partitioning_expert(random_state=Config.SEED):
    """
    Constructs the Partitioning Expert: ExtraTrees Regressor.

    Pipeline:
    1. ImageMetaPCA: Reduces image embedding dimensionality via PCA, keeps metadata raw.
    2. ExtraTreesRegressor: Tree-based ensemble for conditional logic.
    """

    meta_count = len(Config.BINARY_FEATURES)

    pipeline = Pipeline(
        [
            (
                "pca_splitter",
                ImageMetaPCA(
                    n_components=Config.PCA_COMPONENTS,
                    meta_count=meta_count,
                    random_state=random_state,
                ),
            ),
            (
                "regressor",
                ExtraTreesRegressor(
                    n_estimators=Config.ET_N_ESTIMATORS,
                    max_depth=Config.ET_MAX_DEPTH,
                    min_samples_split=Config.ET_MIN_SAMPLES_SPLIT,
                    n_jobs=Config.ET_JOBS,
                    random_state=random_state,
                ),
            ),
        ]
    )

    return pipeline


def get_meta_learner(random_state=Config.SEED):
    """
    Constructs the Level-1 Meta-Learner: Bayesian Ridge Regression.

    This model aggregates predictions from all Level-0 experts.
    It is robust to correlated inputs and automatically infers regularization.
    """
    # BayesianRidge is iterative but generally deterministic given the data order.
    # It does not accept a random_state parameter.

    model = BayesianRidge(n_iter=Config.META_MODEL_ITER, verbose=False)

    return model
