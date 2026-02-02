from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler
from library import config


def get_pipeline(
    pt_method=config.PT_METHOD,
    pt_standardize=config.PT_STANDARDIZE,
    scaler_with_mean=config.SCALER_WITH_MEAN,
    scaler_with_std=config.SCALER_WITH_STD,
):
    """
    Constructs and returns the preprocessing pipeline required for Gaussianizing
    the data for LDA.

    The pipeline consists of:
    1. PowerTransformer: To make feature distributions more Gaussian-like.
    2. StandardScaler: To enforce zero mean and unit variance.

    Args:
        pt_method (str): The power transform method (e.g., 'yeo-johnson').
        pt_standardize (bool): Whether to apply zero-mean, unit-variance normalization
                               in the PowerTransformer. We typically set this to False
                               and use a separate StandardScaler.
        scaler_with_mean (bool): If True, center the data before scaling.
        scaler_with_std (bool): If True, scale the data to unit variance.

    Returns:
        sklearn.pipeline.Pipeline: The configured preprocessing pipeline.
    """

    # Define the steps for the pipeline
    steps = [
        (
            "power_transformer",
            PowerTransformer(method=pt_method, standardize=pt_standardize),
        ),
        (
            "scaler",
            StandardScaler(with_mean=scaler_with_mean, with_std=scaler_with_std),
        ),
    ]

    # Create the pipeline object
    pipeline = Pipeline(steps)

    return pipeline
