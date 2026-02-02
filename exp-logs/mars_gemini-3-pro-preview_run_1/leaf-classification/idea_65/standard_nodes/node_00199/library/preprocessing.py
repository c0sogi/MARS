import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import PowerTransformer, StandardScaler
from library import config, data


class SanitizedPipeline(BaseEstimator, TransformerMixin):
    """
    A high-precision preprocessing pipeline that enforces a variance sanitization
    barrier before applying numerical stabilization and scaling.

    Pipeline Steps:
    1. VarianceThreshold: Removes constant features (threshold=0) to prevent
       numerical explosion during scaling.
    2. PowerTransformer: Applies Yeo-Johnson transformation (standardize=False)
       to stabilize variance and make features more Gaussian-like.
    3. StandardScaler: Centers and scales features to unit variance.

    All operations are strictly enforced to use float64 precision.
    """

    def __init__(self, variance_threshold: float = 0.0):
        """
        Args:
            variance_threshold (float): Threshold for feature variance.
                                        Features with variance <= threshold are removed.
                                        Defaults to 0.0 (remove constant features).
        """
        self.variance_threshold = variance_threshold
        self.pipeline = None

    def fit(self, X, y=None):
        """
        Fits the pipeline to the data.

        Args:
            X (array-like): Training data. Converted to float64.
            y (array-like, optional): Target values. Ignored.

        Returns:
            self: Returns the instance itself.
        """
        # Enforce high precision
        X = np.array(X, dtype=config.FLOAT_PRECISION)

        # Define the sequence of transformations
        self.pipeline = Pipeline(
            [
                ("sanitizer", VarianceThreshold(threshold=self.variance_threshold)),
                (
                    "stabilizer",
                    PowerTransformer(method="yeo-johnson", standardize=False),
                ),
                ("scaler", StandardScaler()),
            ]
        )

        self.pipeline.fit(X, y)
        return self

    def transform(self, X):
        """
        Transforms the data using the fitted pipeline.

        Args:
            X (array-like): Data to transform. Converted to float64.

        Returns:
            np.ndarray: Transformed data in float64 precision.
        """
        if self.pipeline is None:
            raise RuntimeError("The pipeline has not been fitted yet.")

        # Enforce high precision
        X = np.array(X, dtype=config.FLOAT_PRECISION)

        return self.pipeline.transform(X)


def load_data(debug: bool = False, load_cached_data: bool = True):
    """
    Loads the preprocessed dataset using the library's data loader.

    The underlying loader automatically applies the logic defined in
    SanitizedPipeline (VarianceThreshold -> Yeo-Johnson -> StandardScaler)
    and handles caching to ./working/idea_65/.

    Args:
        debug (bool): If True, loads a small subset of data for debugging.
        load_cached_data (bool): If True, attempts to load from cache.

    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, test_ids, classes)
            - X_train, X_val, X_test: Preprocessed float64 feature arrays.
            - y_train, y_val: Encoded integer labels.
            - test_ids: Array of image IDs for the test set.
            - classes: Array of original class names.
    """
    loader = data.LeafDataLoader()
    return loader.load_data(load_cached_data=load_cached_data, debug=debug)
