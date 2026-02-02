import os
import numpy as np
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA
from library import config


class TabularGaussianizer:
    """
    Applies Quantile Transformer with Normal output distribution to tabular features.
    Enforces Gaussian assumption for downstream LDA.
    """

    def __init__(self, n_quantiles=1000, random_state=config.SEED):
        self.qt = QuantileTransformer(
            n_quantiles=n_quantiles,
            output_distribution="normal",
            random_state=random_state,
        )
        self.fitted = False

    def fit(self, X):
        """
        Fits the QuantileTransformer on X.
        """
        self.qt.fit(X)
        self.fitted = True
        return self

    def transform(self, X):
        """
        Transforms X using the fitted transformer.
        """
        if not self.fitted and not hasattr(self.qt, "quantiles_"):
            raise RuntimeError(
                "TabularGaussianizer must be fitted before calling transform."
            )

        return self.qt.transform(X)

    def save(self, output_dir, prefix="tabular_qt"):
        """
        Saves the transformer state to .npy files.
        """
        if not hasattr(self.qt, "quantiles_"):
            raise RuntimeError("Cannot save unfitted TabularGaussianizer.")

        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, f"{prefix}_quantiles.npy"), self.qt.quantiles_)
        np.save(
            os.path.join(output_dir, f"{prefix}_references.npy"), self.qt.references_
        )

    def load(self, output_dir, prefix="tabular_qt"):
        """
        Loads the transformer state from .npy files.
        """
        q_path = os.path.join(output_dir, f"{prefix}_quantiles.npy")
        r_path = os.path.join(output_dir, f"{prefix}_references.npy")

        if os.path.exists(q_path) and os.path.exists(r_path):
            self.qt.quantiles_ = np.load(q_path)
            self.qt.references_ = np.load(r_path)
            self.qt.n_quantiles_ = self.qt.quantiles_.shape[0]
            self.fitted = True
        else:
            raise FileNotFoundError(
                f"Cached TabularGaussianizer files not found in {output_dir} with prefix {prefix}"
            )


class EmbeddingReducer:
    """
    Applies PCA to reduce dimensionality of embeddings while retaining specified variance.
    """

    def __init__(self, n_components=config.PCA_VARIANCE, random_state=config.SEED):
        self.pca = PCA(n_components=n_components, random_state=random_state)
        self.fitted = False

    def fit(self, X):
        """
        Fits PCA on X.
        """
        self.pca.fit(X)
        self.fitted = True
        return self

    def transform(self, X):
        """
        Projects X onto the principal components.
        """
        if not self.fitted and not hasattr(self.pca, "components_"):
            raise RuntimeError(
                "EmbeddingReducer must be fitted before calling transform."
            )

        return self.pca.transform(X)

    def save(self, output_dir, prefix):
        """
        Saves PCA components and mean to .npy files.
        """
        if not hasattr(self.pca, "components_"):
            raise RuntimeError("Cannot save unfitted EmbeddingReducer.")

        os.makedirs(output_dir, exist_ok=True)
        np.save(
            os.path.join(output_dir, f"{prefix}_components.npy"), self.pca.components_
        )
        np.save(os.path.join(output_dir, f"{prefix}_mean.npy"), self.pca.mean_)
        # Save explained variance for completeness
        np.save(
            os.path.join(output_dir, f"{prefix}_explained_variance.npy"),
            self.pca.explained_variance_,
        )

    def load(self, output_dir, prefix):
        """
        Loads PCA state from .npy files.
        """
        c_path = os.path.join(output_dir, f"{prefix}_components.npy")
        m_path = os.path.join(output_dir, f"{prefix}_mean.npy")

        if os.path.exists(c_path) and os.path.exists(m_path):
            self.pca.components_ = np.load(c_path)
            self.pca.mean_ = np.load(m_path)

            # Restore derived attributes required for transform
            self.pca.n_components_ = self.pca.components_.shape[0]
            self.pca.n_features_in_ = self.pca.mean_.shape[0]

            # Load explained variance if available
            v_path = os.path.join(output_dir, f"{prefix}_explained_variance.npy")
            if os.path.exists(v_path):
                self.pca.explained_variance_ = np.load(v_path)

            self.fitted = True
        else:
            raise FileNotFoundError(
                f"Cached EmbeddingReducer files not found in {output_dir} with prefix {prefix}"
            )
