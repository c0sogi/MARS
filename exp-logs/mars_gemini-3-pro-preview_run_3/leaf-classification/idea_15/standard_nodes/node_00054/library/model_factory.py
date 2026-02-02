import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from library.config import Config


class ModelFactory:
    """
    Factory class responsible for constructing the Hyper-Densified Global-Gaussianized LDA pipeline.
    This pipeline integrates Independent Subspace Reduction, Global Feature Gaussianization,
    and a robust Linear Discriminant Analysis classifier.
    """

    @staticmethod
    def create_pipeline(dino_dim, convnext_dim, tabular_dim):
        """
        Constructs the Scikit-Learn pipeline.

        Topology:
        1. Input: Concatenated vector [DINOv2 | ConvNeXt | Tabular]
        2. Independent Subspace Reduction (ColumnTransformer):
           - DINOv2 columns -> PCA (99% variance)
           - ConvNeXt columns -> PCA (99% variance)
           - Tabular columns -> Passthrough
        3. Global Feature Gaussianization:
           - Apply QuantileTransformer (Normal dist) to the combined feature set.
        4. Classifier:
           - LDA with Ledoit-Wolf shrinkage.

        Args:
            dino_dim (int): Number of features in the DINOv2 stream.
            convnext_dim (int): Number of features in the ConvNeXt stream.
            tabular_dim (int): Number of features in the Tabular stream.

        Returns:
            sklearn.pipeline.Pipeline: The constructed and un-fitted pipeline.
        """
        # 1. Define Column Slices based on input dimensions
        # The input X is assumed to be the concatenation of [dino, convnext, tabular]
        dino_slice = slice(0, dino_dim)

        conv_start = dino_dim
        conv_end = dino_dim + convnext_dim
        conv_slice = slice(conv_start, conv_end)

        tab_start = conv_end
        tab_end = conv_end + tabular_dim
        tab_slice = slice(tab_start, tab_end)

        # 2. Step 1: Independent Subspace Reduction via ColumnTransformer
        # We apply PCA independently to the visual streams to preserve their distinct
        # manifold structures while reducing dimensionality based on variance.
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "pca_dino",
                    PCA(
                        n_components=Config.PCA_VARIANCE,
                        svd_solver="full",
                        random_state=Config.SEED,
                    ),
                    dino_slice,
                ),
                (
                    "pca_convnext",
                    PCA(
                        n_components=Config.PCA_VARIANCE,
                        svd_solver="full",
                        random_state=Config.SEED,
                    ),
                    conv_slice,
                ),
                ("passthrough_tabular", "passthrough", tab_slice),
            ],
            verbose_feature_names_out=False,
        )

        # 3. Step 2: Global Feature Gaussianization
        # LDA assumes class-conditional Gaussian densities. We strictly enforce this
        # by warping the entire feature space (visual PCAs + tabular) to a normal distribution.
        gaussianizer = QuantileTransformer(
            output_distribution=Config.QUANTILE_OUTPUT_DIST, random_state=Config.SEED
        )

        # 4. Step 3: Classifier
        # Linear Discriminant Analysis with Ledoit-Wolf shrinkage.
        # solver='lsqr' is required to support shrinkage.
        # shrinkage='auto' implements the Ledoit-Wolf lemma for optimal covariance estimation.
        classifier = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")

        # 5. Assemble Pipeline
        pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("gaussianizer", gaussianizer),
                ("classifier", classifier),
            ]
        )

        return pipeline
