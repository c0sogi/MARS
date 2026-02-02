import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer, QuantileTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression


class JointPCATransformer(BaseEstimator, TransformerMixin):
    """
    Custom Transformer for Joint-Backbone Principal Component Ensemble (JBPCE).

    This transformer implements the core fusion logic of the strategy:
    1. Splits the concatenated input into Anchor (MiniLM) and Auxiliary (MPNet) parts.
    2. Independently L2 normalizes each part to ensure equal variance contribution.
    3. Concatenates the normalized vectors.
    4. Applies PCA to the joint space to extract shared and complementary signals.
    5. L2 normalizes the final projected features.
    """

    def __init__(self, n_components=100, split_index=384, random_state=42):
        """
        Args:
            n_components (int): Number of principal components to keep.
            split_index (int): The index at which to split the input features.
                               Should correspond to the dimension of the first backbone (Anchor).
            random_state (int): Seed for reproducibility.
        """
        self.n_components = n_components
        self.split_index = split_index
        self.random_state = random_state
        self.pca = None
        self.final_norm = Normalizer()

    def fit(self, X, y=None):
        """
        Fits the Joint PCA on the provided data.

        Args:
            X (np.ndarray): Input array of shape (n_samples, dim_a + dim_b).
            y (ignored): Not used.
        """
        # Validate input shape
        if X.shape[1] <= self.split_index:
            raise ValueError(
                f"Input dimension {X.shape[1]} is smaller than or equal to split index {self.split_index}"
            )

        # 1. Split into Backbone A and Backbone B
        X_a = X[:, : self.split_index]
        X_b = X[:, self.split_index :]

        # 2. Independent L2 Normalization
        # Add epsilon to avoid division by zero
        norm_a = np.linalg.norm(X_a, axis=1, keepdims=True)
        norm_a[norm_a == 0] = 1e-10
        X_a_norm = X_a / norm_a

        norm_b = np.linalg.norm(X_b, axis=1, keepdims=True)
        norm_b[norm_b == 0] = 1e-10
        X_b_norm = X_b / norm_b

        # 3. Concatenate normalized embeddings
        X_combined = np.hstack([X_a_norm, X_b_norm])

        # 4. Fit PCA on the joint space
        self.pca = PCA(n_components=self.n_components, random_state=self.random_state)
        self.pca.fit(X_combined)

        return self

    def transform(self, X):
        """
        Applies the transformation pipeline to new data.
        """
        if self.pca is None:
            raise RuntimeError("Transformer has not been fitted yet.")

        # 1. Split
        X_a = X[:, : self.split_index]
        X_b = X[:, self.split_index :]

        # 2. Independent L2 Normalization
        norm_a = np.linalg.norm(X_a, axis=1, keepdims=True)
        norm_a[norm_a == 0] = 1e-10
        X_a_norm = X_a / norm_a

        norm_b = np.linalg.norm(X_b, axis=1, keepdims=True)
        norm_b[norm_b == 0] = 1e-10
        X_b_norm = X_b / norm_b

        # 3. Concatenate
        X_combined = np.hstack([X_a_norm, X_b_norm])

        # 4. Apply PCA projection
        X_projected = self.pca.transform(X_combined)

        # 5. Final L2 Normalization
        X_final = self.final_norm.transform(X_projected)

        return X_final


def build_jbpce_pipeline(
    emb_dim_a=384,
    emb_dim_b=768,
    pca_components=100,
    lr_C=1.0,
    lr_class_weight="balanced",
    n_estimators=20,
    max_samples=1.0,
    bootstrap=True,
    random_state=42,
):
    """
    Constructs the full JBPCE pipeline.

    Structure:
    - Preprocessor (ColumnTransformer):
        - Branch 1 (Embeddings): JointPCATransformer (Split -> Norm -> Concat -> PCA -> Norm)
        - Branch 2 (Metadata): QuantileTransformer (RankGauss)
    - Classifier:
        - BaggingClassifier wrapping LogisticRegression

    Args:
        emb_dim_a (int): Dimension of Backbone A (MiniLM).
        emb_dim_b (int): Dimension of Backbone B (MPNet).
        pca_components (int): Number of PCA components to retain.
        lr_C (float): Inverse regularization strength for Logistic Regression.
        lr_class_weight (str or None): Class weights for Logistic Regression.
        n_estimators (int): Number of base estimators in the Bagging ensemble.
        max_samples (float): Fraction of samples to draw for each base estimator.
        bootstrap (bool): Whether samples are drawn with replacement.
        random_state (int): Seed for reproducibility.

    Returns:
        sklearn.pipeline.Pipeline: The constructed pipeline.
    """

    total_emb_dim = emb_dim_a + emb_dim_b

    # Define the custom transformer for the embedding block
    joint_pca_transformer = JointPCATransformer(
        n_components=pca_components, split_index=emb_dim_a, random_state=random_state
    )

    # Define the transformer for the metadata block
    # We use RankGauss (QuantileTransformer with normal output) to handle outliers
    meta_transformer = QuantileTransformer(
        output_distribution="normal", random_state=random_state
    )

    # Combine branches using ColumnTransformer
    # We assume the input matrix is [Embeddings (0 to total_emb_dim) | Metadata (total_emb_dim to end)]
    preprocessor = ColumnTransformer(
        transformers=[
            ("emb_joint_pca", joint_pca_transformer, slice(0, total_emb_dim)),
            ("meta_rankgauss", meta_transformer, slice(total_emb_dim, None)),
        ]
    )

    # Define the classifier
    # We use 'liblinear' solver as it is robust for high-dimensional data
    base_lr = LogisticRegression(
        solver="liblinear",
        C=lr_C,
        class_weight=lr_class_weight,
        random_state=random_state,
    )

    bagging_clf = BaggingClassifier(
        estimator=base_lr,
        n_estimators=n_estimators,
        max_samples=max_samples,
        bootstrap=bootstrap,
        random_state=random_state,
        n_jobs=1,  # Set to 1 to avoid nested parallelism issues during GridSearch
    )

    # Assemble the full pipeline
    pipeline = Pipeline([("preprocessor", preprocessor), ("clf", bagging_clf)])

    return pipeline
