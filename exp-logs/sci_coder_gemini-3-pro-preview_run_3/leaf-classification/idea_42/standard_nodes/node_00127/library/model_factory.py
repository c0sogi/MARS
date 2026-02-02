import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def create_pipeline(dino_dim=1024, conv_dim=1536, tab_dim=192):
    """
    Constructs the Selective-Topology LDA Pipeline.

    The pipeline expects the input X to be a horizontally concatenated matrix of:
    [DINO_Features | ConvNeXt_Features | Tabular_Features]

    Architecture:
    1. Independent Subspace Reduction (Visual Streams):
       - DINOv2: PCA (99% Variance, No Whitening) -> Preserves linear geometry.
       - ConvNeXt: PCA (99% Variance, No Whitening) -> Preserves linear geometry.

    2. Tabular Gaussianization (Tabular Stream):
       - Handcrafted Features: QuantileTransformer (Normal dist) -> Enforces Gaussian assumption.

    3. Global Alignment:
       - StandardScaler -> Aligns variances of the heterogeneous streams for effective shrinkage.

    4. Classifier:
       - LDA with Ledoit-Wolf Shrinkage -> Robust estimation in high-dim/low-sample regime.

    Args:
        dino_dim (int): Dimension of DINOv2 features (default 1024 for ViT-L).
        conv_dim (int): Dimension of ConvNeXt features (default 1536 for Large).
        tab_dim (int): Dimension of tabular features (default 192).

    Returns:
        sklearn.pipeline.Pipeline: The constructed model pipeline.
    """

    # Define column indices for slicing the concatenated input
    # Input structure: [0...dino_dim-1 | dino_dim...dino+conv-1 | dino+conv...end]
    dino_end = dino_dim
    conv_end = dino_dim + conv_dim
    total_dim = dino_dim + conv_dim + tab_dim

    # Slices
    dino_slice = slice(0, dino_end)
    conv_slice = slice(dino_end, conv_end)
    tab_slice = slice(conv_end, total_dim)

    # 1. & 2. Feature-Specific Preprocessing
    # We use ColumnTransformer to apply different transforms to different feature blocks.
    preprocessor = ColumnTransformer(
        transformers=[
            # Visual Stream 1: DINOv2
            # PCA retaining 99% variance. Whiten=False to preserve linear topology.
            ("dino_pca", PCA(n_components=0.99, whiten=False), dino_slice),
            # Visual Stream 2: ConvNeXt
            # PCA retaining 99% variance. Whiten=False to preserve linear topology.
            ("conv_pca", PCA(n_components=0.99, whiten=False), conv_slice),
            # Tabular Stream: Handcrafted Features
            # Non-linear transformation to Gaussian distribution.
            (
                "tab_gauss",
                QuantileTransformer(output_distribution="normal", random_state=42),
                tab_slice,
            ),
        ],
        remainder="drop",  # Should not be necessary if dimensions match, but good for safety
    )

    # 3. & 4. Global Alignment and Classification
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            # Global Standardization: Crucial for Ledoit-Wolf shrinkage to penalize features uniformly
            ("scaler", StandardScaler()),
            # Classifier: LDA with automatic (Ledoit-Wolf) shrinkage
            # solver='lsqr' supports shrinkage
            ("classifier", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        ]
    )

    return pipeline
