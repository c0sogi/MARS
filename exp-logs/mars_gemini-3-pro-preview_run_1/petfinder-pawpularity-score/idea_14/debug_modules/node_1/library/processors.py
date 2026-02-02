import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from library.config import Config
from library.utils import logit_transform, inverse_logit_transform


class LogitTargetTransformer:
    """
    Handles the transformation of the target variable 'Pawpularity'.
    Maps the bounded range [1, 100] to an unbounded real space (-inf, inf)
    using a logit transformation, and provides the inverse mapping.

    This is crucial for Manifold-Linear Experts (Ridge, SVR) which assume
    Gaussian residuals and unbounded targets.
    """

    def __init__(self):
        pass

    def transform(self, y):
        """
        Applies logit transformation to the target values.

        Args:
            y (np.array or list): Target values in range [1, 100].

        Returns:
            np.array: Transformed values in unbounded space.
        """
        return logit_transform(y)

    def inverse_transform(self, z):
        """
        Applies inverse logit (sigmoid) transformation to predicted values.

        Args:
            z (np.array or list): Predicted values in unbounded space.

        Returns:
            np.array: Values mapped back to range [1, 100].
        """
        return inverse_logit_transform(z)


class FeaturePreprocessor:
    """
    Handles feature preprocessing logic tailored to specific model inductive biases.

    Strategies:
    - 'linear': For Ridge/SVR. Concatenates embeddings and metadata, then applies
      StandardScaler to the full vector. This standardizes scales for regularization.
    - 'tree': For ExtraTrees/LightGBM. Applies PCA to image embeddings to reduce
      dimensionality and noise, then concatenates with raw binary metadata.
      This preserves the discrete nature of metadata for tree splits.
    """

    def __init__(self, seed=Config.SEED, pca_components=Config.PCA_COMPONENTS):
        """
        Initialize the preprocessor.

        Args:
            seed (int): Random seed for PCA reproducibility.
            pca_components (int): Number of components to keep for PCA.
        """
        self.seed = seed
        self.pca_components = pca_components
        self.scaler = None
        self.pca = None
        self.strategy = None

    def fit(self, embeddings, metadata, strategy="linear"):
        """
        Fits the internal transformers (StandardScaler or PCA) based on the strategy.

        Args:
            embeddings (np.ndarray): Image embeddings of shape (N, D).
            metadata (np.ndarray): Binary metadata features of shape (N, 12).
            strategy (str): 'linear' or 'tree'.

        Returns:
            self
        """
        self.strategy = strategy

        if strategy == "linear":
            # Concatenate embeddings and metadata
            # We scale everything together to ensure equal contribution in L2 distance/regularization
            X = np.hstack([embeddings, metadata])
            self.scaler = StandardScaler()
            self.scaler.fit(X)

        elif strategy == "tree":
            # Apply PCA only to the high-dimensional image embeddings
            # Dynamically clamp n_components for small datasets
            n_samples, n_features = embeddings.shape
            n_components = min(self.pca_components, n_samples, n_features)

            self.pca = PCA(n_components=n_components, random_state=self.seed)
            self.pca.fit(embeddings)
            # Metadata is left raw (binary) as trees handle categorical splits well

        else:
            raise ValueError(
                f"Unknown strategy: {strategy}. Expected 'linear' or 'tree'."
            )

        return self

    def transform(self, embeddings, metadata):
        """
        Transforms the input data using the fitted transformers.

        Args:
            embeddings (np.ndarray): Image embeddings of shape (N, D).
            metadata (np.ndarray): Binary metadata features of shape (N, 12).

        Returns:
            np.ndarray: Processed feature matrix.
        """
        if self.strategy is None:
            raise RuntimeError("Preprocessor has not been fitted. Call fit() first.")

        if self.strategy == "linear":
            if self.scaler is None:
                raise RuntimeError("StandardScaler is not fitted.")

            X = np.hstack([embeddings, metadata])
            return self.scaler.transform(X)

        elif self.strategy == "tree":
            if self.pca is None:
                raise RuntimeError("PCA is not fitted.")

            # Transform embeddings using PCA
            emb_pca = self.pca.transform(embeddings)

            # Concatenate with raw metadata
            # Resulting shape: (N, PCA_COMPONENTS + 12)
            return np.hstack([emb_pca, metadata])

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def fit_transform(self, embeddings, metadata, strategy="linear"):
        """
        Fits and transforms the data in one step.

        Args:
            embeddings (np.ndarray): Image embeddings of shape (N, D).
            metadata (np.ndarray): Binary metadata features of shape (N, 12).
            strategy (str): 'linear' or 'tree'.

        Returns:
            np.ndarray: Processed feature matrix.
        """
        self.fit(embeddings, metadata, strategy=strategy)
        return self.transform(embeddings, metadata)
