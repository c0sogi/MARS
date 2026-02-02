import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, BayesianRidge
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from library.config import Config


class ExpertFactory:
    """
    Factory class to construct heterogeneous Level-0 expert models.
    Supports: 'ridge', 'svr', 'extratrees'.
    """

    @staticmethod
    def get_level0_expert(algorithm_name, input_dim, n_samples=None):
        """
        Constructs a scikit-learn pipeline for the specified algorithm.

        Args:
            algorithm_name (str): One of 'ridge', 'svr', 'extratrees'.
            input_dim (int): Total number of input features (Embedding + 12 Metadata).

        Returns:
            sklearn.pipeline.Pipeline: Configured model pipeline.
        """
        algorithm_name = algorithm_name.lower()

        if algorithm_name == "ridge":
            # Ridge Regression: StandardScaler -> RidgeCV
            # RidgeCV automatically selects the best alpha from the provided list
            return Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("regressor", RidgeCV(alphas=Config.RIDGE_ALPHAS)),
                ]
            )

        elif algorithm_name == "svr":
            # Support Vector Regression: StandardScaler -> SVR (RBF Kernel)
            return Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "regressor",
                        SVR(
                            C=Config.SVR_C,
                            epsilon=Config.SVR_EPSILON,
                            kernel=Config.SVR_KERNEL,
                        ),
                    ),
                ]
            )

        elif algorithm_name == "extratrees":
            # ExtraTrees Regressor:
            # 1. Split input into Embeddings (first N-12 cols) and Metadata (last 12 cols).
            # 2. Apply PCA to Embeddings to reduce dimensionality.
            # 3. Pass Metadata through unchanged.
            # 4. Concatenate and feed to ExtraTrees.

            # Calculate embedding dimension (Total - 12 binary metadata features)
            embedding_dim = input_dim - 12
            if embedding_dim <= 0:
                raise ValueError(
                    f"Input dimension {input_dim} is too small to contain embeddings and 12 metadata features."
                )

            # Ensure PCA components do not exceed available features
            n_components = min(embedding_dim, Config.PCA_COMPONENTS)

            if n_samples is not None:
                n_components = min(n_components, n_samples)

            # ColumnTransformer applies transformations to specific columns
            # slice(0, embedding_dim) selects the image embeddings
            # slice(embedding_dim, input_dim) selects the metadata
            preprocessor = ColumnTransformer(
                transformers=[
                    (
                        "pca",
                        PCA(n_components=n_components, random_state=Config.SEED),
                        slice(0, embedding_dim),
                    ),
                    ("meta", "passthrough", slice(embedding_dim, input_dim)),
                ]
            )

            return Pipeline(
                [
                    ("preprocessor", preprocessor),
                    (
                        "regressor",
                        ExtraTreesRegressor(
                            n_estimators=Config.ET_N_ESTIMATORS,
                            max_depth=Config.ET_MAX_DEPTH,
                            min_samples_split=Config.ET_MIN_SAMPLES_SPLIT,
                            random_state=Config.ET_RANDOM_STATE,
                            n_jobs=Config.ET_N_JOBS,
                        ),
                    ),
                ]
            )

        else:
            raise ValueError(
                f"Unknown algorithm: {algorithm_name}. Expected 'ridge', 'svr', or 'extratrees'."
            )


class MetaLearner:
    """
    Level-1 Meta-Learner using Bayesian Ridge Regression.
    Aggregates predictions from Level-0 experts.
    """

    def __init__(self):
        """
        Initialize the Bayesian Ridge model with config parameters.
        """
        self.model = BayesianRidge(
            max_iter=Config.META_N_ITER, tol=Config.META_TOL, verbose=False
        )

    def fit(self, X, y):
        """
        Fit the meta-learner.

        Args:
            X (array-like): Matrix of Level-0 OOF predictions (N_samples, N_experts).
            y (array-like): Target values.
        """
        self.model.fit(X, y)
        return self

    def predict(self, X):
        """
        Predict using the meta-learner.

        Args:
            X (array-like): Matrix of Level-0 test predictions (N_samples, N_experts).

        Returns:
            array-like: Final predictions.
        """
        return self.model.predict(X)
