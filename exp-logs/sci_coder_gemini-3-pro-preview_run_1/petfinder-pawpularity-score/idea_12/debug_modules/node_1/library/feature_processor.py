import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from library.config import Config


class ExpertPreprocessor:
    """
    Handles feature engineering and transformation for Level-0 experts.

    Implements two distinct pipelines:
    1. Linear/Kernel Pipeline (Ridge, SVR):
       Concatenates Embeddings and Metadata, then applies StandardScaler.
    2. Tree-based Pipeline (ExtraTrees, LightGBM):
       Applies PCA to Embeddings (dimensionality reduction), then concatenates with raw Metadata.
    """

    def __init__(self, pca_components=Config.PCA_COMPONENTS, random_state=Config.SEED):
        """
        Args:
            pca_components (int): Number of components for PCA.
            random_state (int): Seed for reproducibility.
        """
        self.pca_components = pca_components
        self.random_state = random_state
        self.linear_scaler = None
        self.tree_pca = None

    def fit_linear(self, embeddings, metadata):
        """
        Fits the StandardScaler for Linear/Kernel experts.

        Args:
            embeddings (np.ndarray): Shape (N, D)
            metadata (np.ndarray): Shape (N, M)
        """
        # Concatenate [Embeddings, Metadata]
        X = np.hstack([embeddings, metadata])

        self.linear_scaler = StandardScaler()
        self.linear_scaler.fit(X)
        return self

    def transform_linear(self, embeddings, metadata):
        """
        Transforms data for Linear/Kernel experts using the fitted scaler.

        Args:
            embeddings (np.ndarray): Shape (N, D)
            metadata (np.ndarray): Shape (N, M)

        Returns:
            np.ndarray: Scaled features of shape (N, D + M)
        """
        if self.linear_scaler is None:
            raise RuntimeError("ExpertPreprocessor (Linear) has not been fitted.")

        X = np.hstack([embeddings, metadata])
        return self.linear_scaler.transform(X)

    def fit_tree(self, embeddings):
        """
        Fits PCA for Tree-based experts.
        Note: PCA is applied ONLY to the embeddings, not the metadata.

        Args:
            embeddings (np.ndarray): Shape (N, D)
        """
        n_samples, n_features = embeddings.shape
        # Ensure n_components is valid given the data shape
        n_comp = min(self.pca_components, n_samples, n_features)

        self.tree_pca = PCA(n_components=n_comp, random_state=self.random_state)
        self.tree_pca.fit(embeddings)
        return self

    def transform_tree(self, embeddings, metadata):
        """
        Transforms data for Tree-based experts.
        Applies PCA to embeddings and concatenates with raw metadata.

        Args:
            embeddings (np.ndarray): Shape (N, D)
            metadata (np.ndarray): Shape (N, M)

        Returns:
            np.ndarray: Features of shape (N, n_components + M)
        """
        if self.tree_pca is None:
            raise RuntimeError("ExpertPreprocessor (Tree) has not been fitted.")

        # Apply PCA to embeddings
        embeddings_pca = self.tree_pca.transform(embeddings)

        # Concatenate with raw metadata
        return np.hstack([embeddings_pca, metadata])


def create_interaction_matrix(expert_preds, metadata):
    """
    Constructs the input matrix for the Level-1 Interaction-Aware Meta-Learner.

    The design matrix consists of:
    1. Raw Expert Predictions (P)
    2. Raw Metadata (M)
    3. Interaction Terms (P x M): The element-wise product of every expert prediction
       with every metadata flag.

    Args:
        expert_preds (np.ndarray): Level-0 Expert predictions. Shape (N, n_experts).
        metadata (np.ndarray): Binary metadata features. Shape (N, n_meta).

    Returns:
        np.ndarray: The constructed feature matrix.
                    Shape (N, n_experts + n_meta + (n_experts * n_meta)).
    """
    # Ensure expert_preds is 2D
    if expert_preds.ndim == 1:
        expert_preds = expert_preds.reshape(-1, 1)

    n_samples = expert_preds.shape[0]

    # 1. Prepare for broadcasting to calculate interactions
    # P shape: (N, n_experts, 1)
    P_expanded = expert_preds[:, :, np.newaxis]

    # M shape: (N, 1, n_meta)
    M_expanded = metadata[:, np.newaxis, :]

    # 2. Calculate Interactions
    # Result shape: (N, n_experts, n_meta)
    interactions = P_expanded * M_expanded

    # Flatten interactions to (N, n_experts * n_meta)
    interactions_flat = interactions.reshape(n_samples, -1)

    # 3. Concatenate all components: [P, M, P*M]
    X_level1 = np.hstack([expert_preds, metadata, interactions_flat])

    return X_level1
