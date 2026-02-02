import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from library.config import (
    TOPOLOGIES,
    DISCRIMINATIVE_COMPONENTS,
    POLY_DEGREE,
    DTYPE,
    RANDOM_SEED,
)


class PipelineFactory:
    """
    Factory class to construct Scikit-Learn pipelines for the DSPGE strategy.
    Implements Topologies A, B, C, and D as defined in the configuration.
    """

    @staticmethod
    def _get_lda_classifier(shrinkage):
        """
        Creates the final LDA classifier step.
        Uses 'lsqr' solver which supports shrinkage (regularization).
        """
        # Solver 'lsqr' supports shrinkage and is generally robust.
        # 'auto' shrinkage is supported by sklearn's LDA.
        return LinearDiscriminantAnalysis(
            solver="lsqr",
            shrinkage=shrinkage,
            store_covariance=True,  # Useful for debugging or potential generative extensions
        )

    @staticmethod
    def build_topology_a(shrinkage):
        """
        Topology A: Marginal Statistical Anchors
        Pipeline: Global -> PowerTransformer -> LDA
        """
        steps = [
            ("pt", PowerTransformer(method="yeo-johnson", standardize=True)),
            ("lda_clf", PipelineFactory._get_lda_classifier(shrinkage)),
        ]
        return Pipeline(steps)

    @staticmethod
    def build_topology_b(shrinkage):
        """
        Topology B: Rotational Statistical Experts
        Pipeline: Global -> PowerTransformer -> PCA(no_whiten) -> PowerTransformer -> LDA
        """
        steps = [
            ("pt_1", PowerTransformer(method="yeo-johnson", standardize=True)),
            # PCA for rotation/alignment. Keep all components (n_components=None).
            # whiten=False is crucial per strategy to avoid noise amplification.
            ("pca", PCA(n_components=None, whiten=False, random_state=RANDOM_SEED)),
            ("pt_2", PowerTransformer(method="yeo-johnson", standardize=True)),
            ("lda_clf", PipelineFactory._get_lda_classifier(shrinkage)),
        ]
        return Pipeline(steps)

    @staticmethod
    def build_topology_c(shrinkage):
        """
        Topology C: Discriminative-Subspace Experts
        Pipeline: Global -> PowerTransformer -> LDA_Transform(n=15) -> Poly(2) -> PT -> LDA
        """
        # Note: The first LDA is a transformer (dimensionality reduction), not a classifier.
        # We use solver='eigen' or 'svd' for transformation usually, but 'lsqr' also works for transform.
        # We don't necessarily need shrinkage for the projection step if N > D, but
        # since we are projecting *global* features (192) which is < N (712),
        # standard LDA is stable. However, adding mild shrinkage to the projector
        # can help if features are collinear. We'll use the same shrinkage or a fixed small one.
        # The strategy implies the projection is to find discriminative axes.

        # We use a fixed small shrinkage for the projector to ensure stability
        # without over-regularizing the subspace creation, or 'auto'.
        # Let's use 'auto' for the projector to be safe, or the passed shrinkage.
        # Using the passed shrinkage ensures consistency with the expert's "hypothesis".

        lda_projector = LinearDiscriminantAnalysis(
            solver="lsqr", shrinkage=shrinkage, n_components=DISCRIMINATIVE_COMPONENTS
        )

        steps = [
            ("pt_1", PowerTransformer(method="yeo-johnson", standardize=True)),
            ("lda_proj", lda_projector),
            (
                "poly",
                PolynomialFeatures(
                    degree=POLY_DEGREE, include_bias=False, interaction_only=False
                ),
            ),
            ("pt_2", PowerTransformer(method="yeo-johnson", standardize=True)),
            ("lda_clf", PipelineFactory._get_lda_classifier(shrinkage)),
        ]
        return Pipeline(steps)

    @staticmethod
    def build_topology_d(shrinkage):
        """
        Topology D: Polynomial Physical Experts
        Pipeline: Morphometrics -> PowerTransformer -> Poly(2) -> PT -> LDA
        """
        steps = [
            ("pt_1", PowerTransformer(method="yeo-johnson", standardize=True)),
            (
                "poly",
                PolynomialFeatures(
                    degree=POLY_DEGREE, include_bias=False, interaction_only=False
                ),
            ),
            ("pt_2", PowerTransformer(method="yeo-johnson", standardize=True)),
            ("lda_clf", PipelineFactory._get_lda_classifier(shrinkage)),
        ]
        return Pipeline(steps)

    @staticmethod
    def create_pipeline(topology_key, shrinkage):
        """
        Universal entry point to create a pipeline based on topology key and shrinkage.

        Args:
            topology_key (str): 'A', 'B', 'C', or 'D'.
            shrinkage (float or str): Shrinkage parameter for LDA ('auto' or float 0-1).

        Returns:
            sklearn.pipeline.Pipeline: The constructed pipeline.
        """
        if topology_key not in TOPOLOGIES:
            raise ValueError(f"Unknown topology key: {topology_key}")

        if topology_key == "A":
            return PipelineFactory.build_topology_a(shrinkage)
        elif topology_key == "B":
            return PipelineFactory.build_topology_b(shrinkage)
        elif topology_key == "C":
            return PipelineFactory.build_topology_c(shrinkage)
        elif topology_key == "D":
            return PipelineFactory.build_topology_d(shrinkage)
        else:
            raise ValueError(f"Topology {topology_key} implementation not found.")
