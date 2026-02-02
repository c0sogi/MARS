import library.config as cfg
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class SelectiveFeaturePipeline:
    """
    Factory class for creating the specific sklearn pipeline used by the
    Orthogonal-Expert Ensemble.

    This pipeline implements 'Selective Feature Topology':
    1. Visual features (DINOv2, ConvNeXt) undergo linear reduction (PCA) to
       preserve their learned manifold structure.
    2. Tabular features undergo non-linear Gaussianization (QuantileTransformer)
       to align with LDA assumptions.
    3. The fused vector is classified by an LDA expert.
    """

    def __init__(self, dino_dim=1024, conv_dim=1536, tab_dim=192):
        """
        Initialize with feature dimensions to calculate slicing indices.

        Args:
            dino_dim (int): Dimension of DINOv2 features (default 1024 for ViT-L).
            conv_dim (int): Dimension of ConvNeXt features (default 1536 for Large).
            tab_dim (int): Dimension of tabular features (default 192).
        """
        self.dino_dim = dino_dim
        self.conv_dim = conv_dim
        self.tab_dim = tab_dim

        # Calculate slice boundaries
        self.dino_end = self.dino_dim
        self.conv_end = self.dino_end + self.conv_dim
        self.tab_end = self.conv_end + self.tab_dim

    def create_expert_pipeline(self):
        """
        Constructs and returns the un-fitted sklearn Pipeline.

        Returns:
            sklearn.pipeline.Pipeline: The configured model pipeline.
        """
        # Define slices for the ColumnTransformer
        # Input X is expected to be [DINO | ConvNeXt | Tabular]
        dino_slice = slice(0, self.dino_end)
        conv_slice = slice(self.dino_end, self.conv_end)
        tab_slice = slice(self.conv_end, self.tab_end)

        # 1. Visual Streams: Independent PCA
        # We use svd_solver='full' to strictly respect the float n_components (variance ratio)
        dino_transformer = PCA(
            n_components=cfg.PCA_VARIANCE, svd_solver="full", random_state=cfg.SEED
        )

        conv_transformer = PCA(
            n_components=cfg.PCA_VARIANCE, svd_solver="full", random_state=cfg.SEED
        )

        # 2. Tabular Stream: Quantile Transformer
        # Gaussianize features for LDA
        tab_transformer = QuantileTransformer(
            output_distribution=cfg.QT_OUTPUT_DIST, random_state=cfg.SEED
        )

        # 3. Preprocessor Construction
        preprocessor = ColumnTransformer(
            transformers=[
                ("dino_pca", dino_transformer, dino_slice),
                ("conv_pca", conv_transformer, conv_slice),
                ("tab_qt", tab_transformer, tab_slice),
            ],
            verbose_feature_names_out=False,
        )

        # 4. Classifier: LDA with Shrinkage
        # LSQR solver supports shrinkage, which is crucial for high-dim low-sample settings
        clf = LinearDiscriminantAnalysis(
            solver=cfg.LDA_SOLVER, shrinkage=cfg.LDA_SHRINKAGE
        )

        # 5. Final Pipeline
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])

        return pipeline
