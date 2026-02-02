import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import PowerTransformer, StandardScaler
from library import config


def get_preprocessing_pipeline(
    variance_threshold=config.VARIANCE_THRESHOLD,
    use_yeo_johnson=config.USE_YEO_JOHNSON,
    standardize=config.STANDARDIZE,
):
    """
    Constructs the sanitized preprocessing pipeline based on the 'Sanitized Integral-Geometric
    High-Precision OAS Discriminant' strategy.

    The pipeline enforces a strict order of operations to ensure numerical stability:
    1. Sanitization (VarianceThreshold): Removes constant or near-constant features (e.g.,
       empty histogram bins, null geometric features) to prevent variance explosion
       or division-by-zero errors in subsequent scaling steps.
    2. Transformation (PowerTransformer): Applies Yeo-Johnson transformation to
       Gaussianize the feature distributions. We use standardize=False here to
       decouple the transformation from the scaling.
    3. Scaling (StandardScaler): Centers and scales the features to unit variance,
       preparing the data for the OAS covariance estimator.

    Args:
        variance_threshold (float): Threshold for removing low-variance features.
                                    Features with variance <= threshold are discarded.
        use_yeo_johnson (bool): Whether to apply Yeo-Johnson power transformation.
        standardize (bool): Whether to apply Standard Scaling.

    Returns:
        sklearn.pipeline.Pipeline: The constructed preprocessing pipeline ready for fitting.
    """
    steps = []

    # 1. Sanitization: Variance Thresholding
    # This is the "Sanitized" part of the strategy. It must happen first.
    if variance_threshold is not None:
        steps.append(("sanitization", VarianceThreshold(threshold=variance_threshold)))

    # 2. Transformation: Yeo-Johnson
    # Gaussianizes features to better satisfy the assumptions of the OAS estimator.
    if use_yeo_johnson:
        steps.append(
            ("yeo_johnson", PowerTransformer(method="yeo-johnson", standardize=False))
        )

    # 3. Scaling: Standard Scaler
    # Ensures all features contribute equally to the covariance matrix calculation.
    if standardize:
        steps.append(("scaler", StandardScaler()))

    pipeline = Pipeline(steps)

    return pipeline
