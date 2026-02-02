import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from library.config import Config
from library.utils import clip_probabilities


def create_hybrid_pipeline(
    dino_dim=1024, conv_dim=1536, tab_dim=192, pca_variance=Config.PCA_VARIANCE
):
    """
    Constructs the Selective-Topology Scikit-Learn pipeline.

    Architecture:
    1. Independent Subspace Reduction (Visual Streams):
       - DINOv2 features -> PCA (preserve linear topology)
       - ConvNeXt features -> PCA (preserve linear topology)
    2. Tabular Gaussianization:
       - Tabular features -> QuantileTransformer (Normal distribution)
    3. Global Variance Alignment:
       - Concatenated features -> StandardScaler
    4. Classifier:
       - LDA with Ledoit-Wolf shrinkage

    Args:
        dino_dim (int): Number of dimensions for DINOv2 features.
        conv_dim (int): Number of dimensions for ConvNeXt features.
        tab_dim (int): Number of dimensions for tabular features.
        pca_variance (float): Variance retention for PCA.

    Returns:
        sklearn.pipeline.Pipeline: The constructed pipeline.
    """

    # Calculate slice indices
    # Input X structure: [DINO_Features | ConvNeXt_Features | Tabular_Features]
    dino_end = dino_dim
    conv_end = dino_dim + conv_dim
    total_dim = conv_end + tab_dim

    # Define transformations for specific feature blocks
    # Note: slices are [start, stop)
    transformers = [
        # Stream 1: Global Geometry (DINOv2) - Linear Reduction
        (
            "dino_pca",
            PCA(n_components=pca_variance, svd_solver="full"),
            slice(0, dino_end),
        ),
        # Stream 2: Local Texture (ConvNeXt) - Linear Reduction
        (
            "conv_pca",
            PCA(n_components=pca_variance, svd_solver="full"),
            slice(dino_end, conv_end),
        ),
        # Stream 3: Handcrafted Features (Tabular) - Non-Linear Gaussianization
        (
            "tab_qt",
            QuantileTransformer(output_distribution="normal", random_state=Config.SEED),
            slice(conv_end, total_dim),
        ),
    ]

    # Build the ColumnTransformer
    # n_jobs=1 is used to prevent potential conflicts with outer parallelism if any
    preprocessor = ColumnTransformer(transformers=transformers, n_jobs=1, verbose=False)

    # Construct the full pipeline
    # StandardScaler ensures Ledoit-Wolf shrinkage applies uniformly across modalities
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("scaler", StandardScaler()),
            ("classifier", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        ]
    )

    return pipeline


class LeafClassifier:
    """
    Wrapper class for the Leaf Species Identification model.
    Manages the lifecycle of the Selective-Topology pipeline.
    """

    def __init__(self):
        self.pipeline = None
        # Default dimensions based on Config models
        # DINOv2 Large: 1024
        # ConvNeXt Large: 1536
        # Tabular: 192
        self.dino_dim = 1024
        self.conv_dim = 1536
        self.tab_dim = 192

    def fit(self, X, y):
        """
        Fits the model pipeline to the training data.

        Args:
            X (np.ndarray): Input features of shape (N, D_total).
                            Expected structure: [DINO | ConvNeXt | Tabular]
            y (np.ndarray): Target labels.

        Returns:
            self: The fitted estimator.
        """
        n_features = X.shape[1]

        # Dynamic dimension validation and adjustment
        expected_dim = self.dino_dim + self.conv_dim + self.tab_dim

        if n_features == expected_dim:
            # Dimensions match expectations, use specific split
            self.pipeline = create_hybrid_pipeline(
                dino_dim=self.dino_dim, conv_dim=self.conv_dim, tab_dim=self.tab_dim
            )
        else:
            # Fallback: If dimensions differ (e.g. different models used),
            # treat visual features as a single block and tabular as the last 192.
            print(
                f"Warning: Input dimension {n_features} does not match expected {expected_dim}."
            )
            print("Adjusting pipeline to treat visual features as a single block.")

            actual_tab_dim = self.tab_dim
            actual_vis_dim = n_features - actual_tab_dim

            # Construct a modified pipeline with a single visual PCA
            transformers = [
                (
                    "vis_pca",
                    PCA(n_components=Config.PCA_VARIANCE, svd_solver="full"),
                    slice(0, actual_vis_dim),
                ),
                (
                    "tab_qt",
                    QuantileTransformer(
                        output_distribution="normal", random_state=Config.SEED
                    ),
                    slice(actual_vis_dim, n_features),
                ),
            ]

            preprocessor = ColumnTransformer(transformers=transformers, n_jobs=1)

            self.pipeline = Pipeline(
                [
                    ("preprocessor", preprocessor),
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
                    ),
                ]
            )

        # Fit the pipeline
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Args:
            X (np.ndarray): Input features.

        Returns:
            np.ndarray: Predicted class labels.
        """
        if self.pipeline is None:
            raise RuntimeError("Model has not been fitted yet.")
        return self.pipeline.predict(X)

    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X.
        Applies clipping to avoid log-loss extremes.

        Args:
            X (np.ndarray): Input features.

        Returns:
            np.ndarray: Predicted probabilities of shape (N, n_classes).
        """
        if self.pipeline is None:
            raise RuntimeError("Model has not been fitted yet.")

        probs = self.pipeline.predict_proba(X)

        # Apply clipping as per task metric requirements
        # max(min(p, 1-10^-15), 10^-15)
        clipped_probs = clip_probabilities(probs)

        return clipped_probs
