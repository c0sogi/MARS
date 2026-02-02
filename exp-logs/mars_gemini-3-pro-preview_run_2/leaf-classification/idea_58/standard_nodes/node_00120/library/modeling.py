import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    PowerTransformer,
    QuantileTransformer,
    PolynomialFeatures,
)
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from library import config


class LDAReductionTransformer(BaseEstimator, TransformerMixin):
    """
    A wrapper around LinearDiscriminantAnalysis to be used explicitly as a
    dimensionality reduction transformer within a Pipeline.

    This ensures that the LDA step is treated as a transformation (projection)
    rather than a final classifier, allowing subsequent steps like PolynomialFeatures.
    """

    def __init__(self, n_components=None):
        self.n_components = n_components
        self.lda = None

    def fit(self, X, y=None):
        """
        Fits the LDA model. Requires target labels 'y'.
        """
        if y is None:
            raise ValueError(
                "LDAReductionTransformer requires target labels (y) to fit."
            )

        # We use the default solver (svd) for dimensionality reduction as it is
        # generally robust for projection purposes.
        self.lda = LinearDiscriminantAnalysis(n_components=self.n_components)
        self.lda.fit(X, y)
        return self

    def transform(self, X):
        """
        Projects X onto the most discriminative directions.
        """
        if self.lda is None:
            raise RuntimeError("LDAReductionTransformer has not been fitted.")
        return self.lda.transform(X)


class TopologyFactory:
    """
    Factory class to construct preprocessing topologies (Pipelines) based on
    configuration names defined in the strategy.
    """

    @staticmethod
    def get_topology(topology_name):
        """
        Returns a sklearn Pipeline object representing the requested topology.

        Args:
            topology_name (str): The identifier for the topology
                                 (e.g., 'marginal', 'rotational', 'interaction').

        Returns:
            sklearn.pipeline.Pipeline: The preprocessing pipeline.
        """
        steps = []

        if topology_name == config.TOPOLOGY_MARGINAL:
            # Baseline: Stabilize variance feature-wise
            steps.append(("power", PowerTransformer(method="yeo-johnson")))

        elif topology_name == config.TOPOLOGY_ROTATIONAL:
            # Aligns subspace with principal axes without whitening, then restabilizes
            steps.append(("power_1", PowerTransformer(method="yeo-johnson")))
            steps.append(
                ("pca", PCA(whiten=False))
            )  # Keeps all components, just rotates
            steps.append(("power_2", PowerTransformer(method="yeo-johnson")))

        elif topology_name == config.TOPOLOGY_ROBUST:
            # Rank-based normalization for skewed distributions
            steps.append(
                (
                    "quantile",
                    QuantileTransformer(
                        output_distribution=config.QUANTILE_OUTPUT_DIST,
                        n_quantiles=config.QUANTILE_N_QUANTILES,
                        random_state=config.RANDOM_SEED,
                    ),
                )
            )

        elif topology_name == config.TOPOLOGY_INTERACTION:
            # Discriminative Bottleneck -> Quadratic Interactions
            steps.append(("power_1", PowerTransformer(method="yeo-johnson")))
            steps.append(
                (
                    "lda_reduce",
                    LDAReductionTransformer(
                        n_components=config.INTERACTION_LDA_COMPONENTS
                    ),
                )
            )
            steps.append(
                (
                    "poly",
                    PolynomialFeatures(degree=config.POLY_DEGREE, include_bias=False),
                )
            )
            steps.append(("power_2", PowerTransformer(method="yeo-johnson")))

        elif topology_name == config.TOPOLOGY_PHYSICAL_POLY:
            # Simple polynomial expansion for physical scalars
            steps.append(
                (
                    "poly",
                    PolynomialFeatures(degree=config.POLY_DEGREE, include_bias=False),
                )
            )

        else:
            raise ValueError(f"Unknown topology name: {topology_name}")

        return Pipeline(steps)


def get_base_estimator(shrinkage):
    """
    Returns the Linear Discriminant Analysis estimator with the specified shrinkage.

    Args:
        shrinkage (float): The regularization parameter.

    Returns:
        LinearDiscriminantAnalysis: The configured estimator.
    """
    return LinearDiscriminantAnalysis(solver=config.LDA_SOLVER, shrinkage=shrinkage)


class ExpertPipeline(BaseEstimator, ClassifierMixin):
    """
    A unified Expert model that combines a specific Preprocessing Topology
    with a Shrinkage-Regularized LDA Classifier.
    """

    def __init__(self, topology, shrinkage):
        self.topology = topology
        self.shrinkage = shrinkage
        self.pipeline = None

    def fit(self, X, y):
        """
        Constructs and fits the internal pipeline.
        """
        # 1. Build Preprocessing Topology
        preprocessor = TopologyFactory.get_topology(self.topology)

        # 2. Build Estimator
        estimator = get_base_estimator(self.shrinkage)

        # 3. Assemble Full Pipeline
        self.pipeline = Pipeline(
            [("preprocessor", preprocessor), ("classifier", estimator)]
        )

        # 4. Fit
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        if self.pipeline is None:
            raise RuntimeError("ExpertPipeline has not been fitted.")
        return self.pipeline.predict(X)

    def predict_proba(self, X):
        if self.pipeline is None:
            raise RuntimeError("ExpertPipeline has not been fitted.")
        return self.pipeline.predict_proba(X)
