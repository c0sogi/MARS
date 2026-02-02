import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    PowerTransformer,
    QuantileTransformer,
    PolynomialFeatures,
)
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from library.config import Config


class PipelineFactory:
    """
    Factory class to construct Scikit-Learn pipelines for the Discriminative-Manifold
    Generative Ensemble (DMGE).

    Provides static methods to build the five specific expert topologies (A-E).
    """

    @staticmethod
    def get_lda_classifier(shrinkage):
        """
        Helper to create the final LDA classifier with specific shrinkage.

        Args:
            shrinkage (float or str): The shrinkage parameter (e.g., 0.01, 'auto').

        Returns:
            LinearDiscriminantAnalysis: The configured classifier.
        """
        return LinearDiscriminantAnalysis(solver=Config.LDA_SOLVER, shrinkage=shrinkage)

    @staticmethod
    def build_topology_a(shrinkage):
        """
        Topology A: Marginal Statistical Anchors.

        Pipeline:
            1. PowerTransformer (Yeo-Johnson)
            2. LDA Classifier (Fixed Shrinkage)

        Args:
            shrinkage (float): Shrinkage intensity (e.g., 0.001, 0.01).
        """
        steps = [
            ("pt", PowerTransformer(method="yeo-johnson")),
            ("clf", PipelineFactory.get_lda_classifier(shrinkage)),
        ]
        return Pipeline(steps)

    @staticmethod
    def build_topology_b(shrinkage):
        """
        Topology B: Rotational Statistical Experts.

        Pipeline:
            1. PowerTransformer (Yeo-Johnson)
            2. PCA (No whitening, preserves variance scale)
            3. PowerTransformer (Yeo-Johnson)
            4. LDA Classifier (Fixed Shrinkage)

        Args:
            shrinkage (float): Shrinkage intensity.
        """
        steps = [
            ("pt1", PowerTransformer(method="yeo-johnson")),
            (
                "pca",
                PCA(
                    whiten=Config.TOPOLOGY_B_PCA_WHITEN, random_state=Config.RANDOM_SEED
                ),
            ),
            ("pt2", PowerTransformer(method="yeo-johnson")),
            ("clf", PipelineFactory.get_lda_classifier(shrinkage)),
        ]
        return Pipeline(steps)

    @staticmethod
    def build_topology_c(shrinkage):
        """
        Topology C: Discriminative-Interaction Experts.

        Pipeline:
            1. PowerTransformer (Yeo-Johnson)
            2. LDA Transformer (Supervised Projection to k=25 components)
            3. PolynomialFeatures (Degree 2, Interaction terms)
            4. PowerTransformer (Yeo-Johnson)
            5. LDA Classifier (Fixed Shrinkage)

        Args:
            shrinkage (float): Shrinkage intensity for the final classifier.
        """
        steps = [
            ("pt1", PowerTransformer(method="yeo-johnson")),
            # Supervised projection to discriminative subspace
            (
                "lda_proj",
                LinearDiscriminantAnalysis(
                    solver="svd",
                    n_components=Config.TOPOLOGY_C_LDA_COMPONENTS,
                ),
            ),
            # Quadratic expansion of discriminative features
            (
                "poly",
                PolynomialFeatures(
                    degree=Config.TOPOLOGY_C_POLY_DEGREE, include_bias=False
                ),
            ),
            ("pt2", PowerTransformer(method="yeo-johnson")),
            ("clf", PipelineFactory.get_lda_classifier(shrinkage)),
        ]
        return Pipeline(steps)

    @staticmethod
    def build_topology_d():
        """
        Topology D: Polynomial Physical Experts.

        Intended for Morphometric Features (11 dims).

        Pipeline:
            1. PowerTransformer (Yeo-Johnson)
            2. PolynomialFeatures (Degree 2)
            3. PowerTransformer (Yeo-Johnson)
            4. LDA Classifier (Auto Shrinkage / Ledoit-Wolf)
        """
        steps = [
            ("pt1", PowerTransformer(method="yeo-johnson")),
            (
                "poly",
                PolynomialFeatures(
                    degree=Config.TOPOLOGY_D_POLY_DEGREE, include_bias=False
                ),
            ),
            ("pt2", PowerTransformer(method="yeo-johnson")),
            # Use 'auto' (Ledoit-Wolf) for robust estimation on physical constraints
            ("clf", PipelineFactory.get_lda_classifier(shrinkage="auto")),
        ]
        return Pipeline(steps)

    @staticmethod
    def build_topology_e():
        """
        Topology E: Robust Distributional Experts.

        Pipeline:
            1. QuantileTransformer (Normal output)
            2. LDA Classifier (Auto Shrinkage / Ledoit-Wolf)
        """
        steps = [
            (
                "qt",
                QuantileTransformer(
                    output_distribution=Config.TOPOLOGY_E_OUTPUT_DIST,
                    n_quantiles=Config.TOPOLOGY_E_N_QUANTILES,
                    random_state=Config.RANDOM_SEED,
                ),
            ),
            # Use 'auto' shrinkage as a proxy for robust covariance estimation
            ("clf", PipelineFactory.get_lda_classifier(shrinkage="auto")),
        ]
        return Pipeline(steps)
