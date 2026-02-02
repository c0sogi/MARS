import numpy as np
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
from library.config import POWER_PARAMS, QUANTILE_PARAMS


def get_power_transformer():
    """
    Returns a PowerTransformer configured with Yeo-Johnson method and standardization.
    Used for the 'Statistical Anchors' and 'Polynomial-Physical' experts.

    Configuration (from library.config):
    - method: 'yeo-johnson'
    - standardize: True

    Returns:
        sklearn.preprocessing.PowerTransformer
    """
    # Initialize transformer with params from config
    transformer = PowerTransformer(**POWER_PARAMS)
    return transformer


def get_quantile_transformer():
    """
    Returns a QuantileTransformer configured for Gaussianization.
    Used for the 'Robust Distributional' experts.

    Configuration (from library.config):
    - n_quantiles: 50 (Constrained to ~7% of N to prevent overfitting)
    - output_distribution: 'normal'
    - random_state: Fixed for reproducibility

    Returns:
        sklearn.preprocessing.QuantileTransformer
    """
    # Initialize transformer with params from config
    transformer = QuantileTransformer(**QUANTILE_PARAMS)
    return transformer


def get_preprocessor(name):
    """
    Dispatcher function to retrieve the correct preprocessor based on the
    configuration string identifier.

    Args:
        name (str): 'power' or 'quantile'.

    Returns:
        Transformer instance (sklearn compatible).

    Raises:
        ValueError: If the name is not recognized.
    """
    if name == "power":
        return get_power_transformer()
    elif name == "quantile":
        return get_quantile_transformer()
    else:
        raise ValueError(
            f"Unknown preprocessor identifier: '{name}'. Expected 'power' or 'quantile'."
        )
