import os
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
    QuantileTransformer,
    PolynomialFeatures,
)
from library.config import Config, get_metadata_features
from library.utils import set_seed


class InteractionMetadataProcessor:
    """
    Implements the metadata processing branch of the Projected Semantic-Interaction Ensemble.

    This class handles:
    1. Retrieval of raw numerical metadata.
    2. Non-linear transformation using QuantileTransformer (RankGauss).
    3. Explicit feature interaction generation using PolynomialFeatures.
    4. Standardization of the resulting feature space.
    """

    def __init__(self):
        self.pipeline_path = os.path.join(Config.WORKING_DIR, "meta_pipeline.joblib")
        self.pipeline = None
        set_seed(Config.RANDOM_SEED)

    def fit(
        self,
        df_train: pd.DataFrame,
        load_cached_data: bool = True,
        split_name: str = "train",
    ):
        """
        Fits the interaction-aware pipeline on the training data and saves it.

        Args:
            df_train (pd.DataFrame): Training data.
            load_cached_data (bool): Whether to use cached raw features.
            split_name (str): The split name to use for caching (default: "train").

        Returns:
            self
        """
        # Ensure working directory exists
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

        # Get raw numerical features
        X_raw = get_metadata_features(
            df_train, split=split_name, load_cached_data=load_cached_data
        )

        # Define the pipeline:
        # 1. QuantileTransformer: Gaussianize features to handle outliers and skew.
        # 2. PolynomialFeatures: Create interactions (A, B, A*B) to allow linear model to see conditional patterns.
        # 3. StandardScaler: Scale the resulting expanded space (interactions of normals are not necessarily normal/scaled).
        self.pipeline = Pipeline(
            [
                (
                    "quantile",
                    QuantileTransformer(
                        output_distribution="normal", random_state=Config.RANDOM_SEED
                    ),
                ),
                (
                    "poly",
                    PolynomialFeatures(
                        degree=2, interaction_only=True, include_bias=False
                    ),
                ),
                ("scaler", StandardScaler()),
            ]
        )

        # Fit the pipeline
        self.pipeline.fit(X_raw)

        # Save the fitted pipeline
        joblib.dump(self.pipeline, self.pipeline_path)

        return self

    def transform(self, df: pd.DataFrame, split: str, load_cached_data: bool = True):
        """
        Transforms metadata into the interaction-aware feature space.

        Args:
            df (pd.DataFrame): Data containing metadata columns.
            split (str): Dataset split ('train', 'val', 'test') for cache management.
            load_cached_data (bool): Whether to use cached processed features.

        Returns:
            np.ndarray: The processed feature matrix.
        """
        # Define cache path for the transformed output
        cache_path = os.path.join(Config.WORKING_DIR, f"{split}_meta_transformed.npy")

        # Check cache
        if load_cached_data and os.path.exists(cache_path):
            try:
                X_transformed = np.load(cache_path)
                return X_transformed
            except Exception:
                # If load fails, proceed to re-compute
                pass

        # Load pipeline if not currently in memory
        if self.pipeline is None:
            if os.path.exists(self.pipeline_path):
                self.pipeline = joblib.load(self.pipeline_path)
            else:
                raise FileNotFoundError(
                    f"Pipeline model not found at {self.pipeline_path}. "
                    "Please call fit() on training data first."
                )

        # Get raw numerical features
        X_raw = get_metadata_features(
            df, split=split, load_cached_data=load_cached_data
        )

        # Apply transformation
        X_transformed = self.pipeline.transform(X_raw)

        # Save to cache
        np.save(cache_path, X_transformed)

        return X_transformed

    def fit_transform(self, df_train: pd.DataFrame, load_cached_data: bool = True):
        """
        Fits the pipeline on training data and returns the transformed training features.

        Args:
            df_train (pd.DataFrame): Training data.
            load_cached_data (bool): Whether to use cached raw features.

        Returns:
            np.ndarray: Transformed training features.
        """
        self.fit(df_train, load_cached_data=load_cached_data)
        # We force re-computation/caching of the transformed train data
        # by calling transform immediately after fit.
        return self.transform(
            df_train, split="train", load_cached_data=load_cached_data
        )
