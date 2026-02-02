from sklearn.preprocessing import PowerTransformer, QuantileTransformer
from library.config import Config


def get_preprocessor(resolution_type):
    """
    Factory function that returns the specific scikit-learn transformer corresponding
    to the requested Gaussian Resolution type defined in the Config.

    The 'Gaussian Resolution Spectrum' consists of:
    1. Parametric Basis (Yeo-Johnson):
       - Assumes data can be stabilized via power law.
       - High bias, low variance.
       - Best for global stabilization.

    2. Robust Non-Parametric Basis (Quantile, n=20):
       - Coarse resolution.
       - Handles outliers and skew but strictly limits flexibility to prevent overfitting.
       - Uses ~3% of N (given N~700) as breakpoints.

    3. Flexible Non-Parametric Basis (Quantile, n=100):
       - Fine resolution.
       - Captures local density variations.
       - Uses ~14% of N as breakpoints.

    Args:
        resolution_type (str): One of the BASIS_* constants from Config.

    Returns:
        TransformerMixin: An initialized scikit-learn transformer (PowerTransformer or QuantileTransformer).

    Raises:
        ValueError: If an unknown resolution_type is provided.
    """
    if resolution_type == Config.BASIS_PARAMETRIC:
        # Parametric Gaussianization
        # Standardize=True ensures zero mean and unit variance after transformation
        return PowerTransformer(method="yeo-johnson", standardize=True)

    elif resolution_type == Config.BASIS_ROBUST:
        # Robust Non-Parametric Gaussianization (Coarse)
        # Low n_quantiles forces a smoother mapping, resisting outlier noise
        return QuantileTransformer(
            output_distribution="normal",
            n_quantiles=Config.N_QUANTILES_ROBUST,
            random_state=Config.RANDOM_SEED,
        )

    elif resolution_type == Config.BASIS_FLEXIBLE:
        # Flexible Non-Parametric Gaussianization (Fine)
        # Higher n_quantiles allows capturing multi-modal densities
        return QuantileTransformer(
            output_distribution="normal",
            n_quantiles=Config.N_QUANTILES_FLEXIBLE,
            random_state=Config.RANDOM_SEED,
        )

    else:
        raise ValueError(
            f"Unknown resolution type: {resolution_type}. "
            f"Expected one of: {[Config.BASIS_PARAMETRIC, Config.BASIS_ROBUST, Config.BASIS_FLEXIBLE]}"
        )
