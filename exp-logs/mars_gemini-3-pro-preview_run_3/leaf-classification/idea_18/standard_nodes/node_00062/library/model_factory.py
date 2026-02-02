import sklearn
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from library.config import Config


class ModelFactory:
    """
    Factory class to construct the machine learning pipeline for the
    Hyper-Densified Independent-Component LDA strategy.
    """

    @staticmethod
    def build_lda_pipeline():
        """
        Constructs the scikit-learn pipeline.

        The pipeline consists of three main stages:
        1. Independent Subspace Reduction:
           - Splits the concatenated input vector into DINO, ConvNeXt, and Tabular streams.
           - Applies PCA (retaining 99% variance) to DINO and ConvNeXt streams independently.
           - Passes Tabular features through without reduction.
        2. Global Gaussianization:
           - Applies a QuantileTransformer to warp the joint feature distribution to a Gaussian (Normal).
           - This ensures the data satisfies the normality assumption of LDA.
        3. Classifier:
           - Applies Linear Discriminant Analysis with Ledoit-Wolf shrinkage (solver='lsqr', shrinkage='auto').
           - This provides a robust covariance estimate in the high-dimensional, few-shot regime.

        Input Structure Expected:
           Concatenated vector: [DINO_Features (1024) | ConvNeXt_Features (1536) | Tabular_Features (192)]

        Returns:
            sklearn.pipeline.Pipeline: The compiled model pipeline.
        """

        # ==========================================
        # 1. Define Feature Dimensions
        # ==========================================
        # These match the output dimensions from library.feature_extractor
        DIM_DINO = 1024
        DIM_CONV = 1536
        DIM_TAB = Config.NUM_TABULAR_FEATURES

        # ==========================================
        # 2. Define Slices for ColumnTransformer
        # ==========================================
        # Stream 1: DINOv2 (Indices 0 to 1024)
        slice_dino = slice(0, DIM_DINO)

        # Stream 2: ConvNeXt (Indices 1024 to 2560)
        slice_conv = slice(DIM_DINO, DIM_DINO + DIM_CONV)

        # Stream 3: Tabular (Indices 2560 to 2752)
        slice_tab = slice(DIM_DINO + DIM_CONV, DIM_DINO + DIM_CONV + DIM_TAB)

        # ==========================================
        # 3. Construct Pipeline
        # ==========================================

        # Stage 1: Independent Subspace Reduction
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "dino_pca",
                    PCA(n_components=Config.PCA_VARIANCE, random_state=Config.SEED),
                    slice_dino,
                ),
                (
                    "conv_pca",
                    PCA(n_components=Config.PCA_VARIANCE, random_state=Config.SEED),
                    slice_conv,
                ),
                ("tab_pass", "passthrough", slice_tab),
            ],
            verbose_feature_names_out=False,
        )

        # Stage 2 & 3: Gaussianization and Classification
        pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "gaussianizer",
                    QuantileTransformer(
                        output_distribution="normal", random_state=Config.SEED
                    ),
                ),
                (
                    "classifier",
                    LinearDiscriminantAnalysis(
                        solver="lsqr",
                        shrinkage="auto",  # Activates Ledoit-Wolf shrinkage
                    ),
                ),
            ]
        )

        return pipeline
