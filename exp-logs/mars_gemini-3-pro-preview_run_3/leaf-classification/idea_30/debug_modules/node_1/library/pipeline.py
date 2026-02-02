from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from library.config import Config


class ModelFactory:
    """
    Factory class for creating the Selective-Topology Orthogonal Manifold-Densified LDA pipeline.
    """

    def __init__(self):
        pass

    def create_pipeline(self, feature_indices):
        """
        Constructs the scikit-learn pipeline with selective feature topology.

        Args:
            feature_indices (dict): Dictionary containing start/end indices for each feature group.
                                    Expected keys: 'dino', 'conv', 'tabular'.
                                    Format: {'dino': (start, end), ...}

        Returns:
            sklearn.pipeline.Pipeline: The constructed pipeline.
        """
        # 1. Parse Feature Indices
        # The input X matrix is a concatenation of [DINO | CONV | TABULAR]
        # We need slice objects to direct specific columns to specific transformers.
        dino_start, dino_end = feature_indices["dino"]
        conv_start, conv_end = feature_indices["conv"]
        tab_start, tab_end = feature_indices["tabular"]

        dino_slice = slice(dino_start, dino_end)
        conv_slice = slice(conv_start, conv_end)
        tab_slice = slice(tab_start, tab_end)

        # 2. Define Stream-Specific Transformers

        # Visual Stream A: DINOv2 (Global Geometry)
        # Strategy: Independent Subspace Reduction (Linear)
        # We use PCA to reduce dimensions while preserving 99% variance.
        # Crucially, we do NOT apply non-linear transformations here to preserve linear separability.
        pca_dino = PCA(n_components=Config.PCA_VARIANCE, random_state=Config.SEED)

        # Visual Stream B: ConvNeXt (Local Texture)
        # Strategy: Independent Subspace Reduction (Linear)
        pca_conv = PCA(n_components=Config.PCA_VARIANCE, random_state=Config.SEED)

        # Tabular Stream: Handcrafted Features
        # Strategy: Tabular Gaussianization (Non-Linear)
        # These features (histograms) have arbitrary distributions. We force them to be Gaussian
        # to satisfy the normality assumption of LDA.
        qt_tab = QuantileTransformer(
            output_distribution="normal", random_state=Config.SEED
        )

        # 3. Construct ColumnTransformer
        # Applies the specific transformers to their respective column slices.
        preprocessor = ColumnTransformer(
            transformers=[
                ("pca_dino", pca_dino, dino_slice),
                ("pca_conv", pca_conv, conv_slice),
                ("qt_tab", qt_tab, tab_slice),
            ],
            # We don't want prefixes added to feature names if we were to inspect them
            verbose_feature_names_out=False,
        )

        # 4. Construct Main Pipeline
        pipeline = Pipeline(
            [
                # Step 1: Selective Feature Topology (Preprocessing)
                ("preprocessor", preprocessor),
                # Step 2: Global Variance Alignment
                # Standardize the concatenated vector [PCA-Visual, PCA-Visual, QT-Tabular].
                # This ensures Ledoit-Wolf shrinkage applies uniformly across modalities.
                ("scaler", StandardScaler()),
                # Step 3: Classifier
                # Linear Discriminant Analysis with Ledoit-Wolf shrinkage.
                # Robust for High-Dimension Low-Sample Size (HDLSS) regimes.
                (
                    "classifier",
                    LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
                ),
            ]
        )

        return pipeline
