import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
from sklearn.decomposition import PCA
from library.config import Config


def get_topology(topology_name: str) -> Pipeline:
    """
    Factory function to create the specific Gaussianization topology pipeline.

    Args:
        topology_name (str): The name of the topology. Must be one of:
                             'marginal', 'iterative', 'constrained'.

    Returns:
        sklearn.pipeline.Pipeline: The constructed preprocessing pipeline.

    Raises:
        ValueError: If the topology_name is not recognized.
    """
    if topology_name == "marginal":
        # Topology A: Marginal Parametric Anchors
        # Mechanism: Gaussianizes each feature independently using Yeo-Johnson.
        # Role: Preserves the robust baseline.
        return Pipeline(
            [
                (
                    "pt",
                    PowerTransformer(method="yeo-johnson", standardize=True),
                ),
            ]
        )

    elif topology_name == "iterative":
        # Topology B: Iterative Parametric Experts
        # Mechanism:
        # 1. Stabilize: Initial Power Transform to marginals.
        # 2. Rotate: PCA (without whitening) to align with principal axes.
        # 3. Re-Gaussianize: Second Power Transform on Principal Components.
        # Rationale: Approximates Multivariate Normality without the noise amplification of whitening.
        return Pipeline(
            [
                (
                    "pt_1",
                    PowerTransformer(method="yeo-johnson", standardize=True),
                ),
                (
                    "pca",
                    PCA(whiten=Config.PCA_WHITEN),
                ),
                (
                    "pt_2",
                    PowerTransformer(method="yeo-johnson", standardize=True),
                ),
            ]
        )

    elif topology_name == "constrained":
        # Topology C: Constrained Non-Parametric Experts
        # Mechanism: Quantile Transform with a strictly limited number of quantiles.
        # Rationale: Handles skewed features while preventing overfitting via low-rank constraint.
        return Pipeline(
            [
                (
                    "qt",
                    QuantileTransformer(
                        output_distribution=Config.QUANTILE_OUTPUT_DIST,
                        n_quantiles=Config.QUANTILE_N_QUANTILES,
                        random_state=Config.RANDOM_SEED,
                    ),
                ),
            ]
        )

    else:
        raise ValueError(
            f"Unknown topology: '{topology_name}'. "
            f"Available options are: {Config.TOPOLOGIES}"
        )
