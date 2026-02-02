import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from library.config import GRID_ROUNDING, K_FOLDS_TARGET_ENCODING, CACHE_DIR, SEED


class SpatialTargetEncoder:
    """
    Encodes spatial coordinates into grid-based statistics (Mean Fare).
    Implements K-Fold Target Encoding to prevent data leakage during training.
    """

    def __init__(self, grid_rounding=GRID_ROUNDING, k_folds=K_FOLDS_TARGET_ENCODING):
        self.grid_rounding = grid_rounding
        self.k_folds = k_folds
        self.global_means = None
        self.global_mean_scalar = None
        self.group_cols = ["p_lat", "p_lon", "d_lat", "d_lon"]

    def _get_bin_columns(self, df):
        """
        Discretizes continuous coordinates into grid bins.
        """
        bins = pd.DataFrame(index=df.index)
        bins["p_lat"] = df["pickup_latitude"].round(self.grid_rounding)
        bins["p_lon"] = df["pickup_longitude"].round(self.grid_rounding)
        bins["d_lat"] = df["dropoff_latitude"].round(self.grid_rounding)
        bins["d_lon"] = df["dropoff_longitude"].round(self.grid_rounding)
        return bins

    def fit(self, df):
        """
        Computes global statistics on the full dataset.
        Used for inference on the test set.
        """
        bins = self._get_bin_columns(df)
        tmp = bins.copy()
        tmp["fare_amount"] = df["fare_amount"]

        # Compute mean for every route
        # This creates a Series with MultiIndex
        self.global_means = tmp.groupby(self.group_cols)["fare_amount"].mean()
        self.global_mean_scalar = df["fare_amount"].mean()
        return self

    def transform(self, df):
        """
        Applies learned global statistics to new data (test set).
        """
        if self.global_means is None:
            raise ValueError("Encoder must be fitted before calling transform.")

        bins = self._get_bin_columns(df)

        # Map global means to the bins
        # join is efficient; global_means index matches group_cols
        mapped = bins.join(
            self.global_means.rename("route_avg_fare"), on=self.group_cols, how="left"
        )

        # Fill missing routes with global scalar mean
        mapped["route_avg_fare"] = mapped["route_avg_fare"].fillna(
            self.global_mean_scalar
        )

        return mapped["route_avg_fare"].values

    def fit_transform_cv(self, df):
        """
        Performs K-Fold Target Encoding to generate features for the training set without leakage.
        Uses an optimized subtraction method for efficiency on large datasets.
        """
        # 1. Prepare Bins and Global Stats
        bins = self._get_bin_columns(df)
        tmp = bins.copy()
        tmp["fare_amount"] = df["fare_amount"]

        # Calculate Global Sums and Counts
        # We group by the bins
        global_stats = tmp.groupby(self.group_cols)["fare_amount"].agg(["sum", "count"])
        global_scalar_mean = df["fare_amount"].mean()

        # Result container
        result = np.full(len(df), np.nan, dtype=np.float32)

        # 2. K-Fold Loop
        kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=SEED)

        print(f"Starting {self.k_folds}-Fold Target Encoding on {len(df)} rows...")

        for fold_idx, (train_indices, val_indices) in enumerate(kf.split(df)):
            # Get stats for the validation fold
            val_tmp = tmp.iloc[val_indices]
            val_stats = val_tmp.groupby(self.group_cols)["fare_amount"].agg(
                ["sum", "count"]
            )

            # Join with global stats to perform subtraction
            # We only need to compute 'rest' stats for bins present in this validation fold
            merged = val_stats.join(global_stats, lsuffix="_val", rsuffix="_global")

            # Calculate 'Rest' stats (Global - Val)
            rest_sum = merged["sum_global"] - merged["sum_val"]
            rest_count = merged["count_global"] - merged["count_val"]

            # Calculate Mean
            # rest_count can be 0 if a bin appears ONLY in this fold.
            # Division by zero will result in inf or nan
            with np.errstate(divide="ignore", invalid="ignore"):
                rest_mean = rest_sum / rest_count

            # Create a mapping series for this fold
            fold_mapping = rest_mean.rename("route_avg_fare")

            # Map back to the validation rows
            # val_tmp has the bin columns; we join on them
            mapped_values = val_tmp.join(fold_mapping, on=self.group_cols, how="left")[
                "route_avg_fare"
            ]

            # Handle NaNs and Infs (where rest_count <= 0)
            # If rest_count is 0, we have no history for this route in the training set (the rest of the folds).
            # Fallback to global scalar mean.
            vals = mapped_values.values
            mask_invalid = ~np.isfinite(vals)
            vals[mask_invalid] = global_scalar_mean

            # Assign to result array
            result[val_indices] = vals

        return result


def generate_global_features(train_df, test_df, load_cached_data=True):
    """
    Orchestrates the generation and caching of global spatial features.

    Args:
        train_df: Full training dataframe (or large subset).
        test_df: Test dataframe.
        load_cached_data: Boolean, whether to attempt loading from cache.

    Returns:
        Tuple of (train_features_df, test_features_df).
        Each dataframe contains ['key', 'route_avg_fare'].
    """
    # Ensure cache directory exists
    os.makedirs(CACHE_DIR, exist_ok=True)

    train_cache_path = os.path.join(CACHE_DIR, "train_route_features.parquet")
    test_cache_path = os.path.join(CACHE_DIR, "test_route_features.parquet")

    # Try loading from cache
    if (
        load_cached_data
        and os.path.exists(train_cache_path)
        and os.path.exists(test_cache_path)
    ):
        print("Loading cached global features from parquet...")
        try:
            train_feats = pd.read_parquet(train_cache_path)
            test_feats = pd.read_parquet(test_cache_path)
            # Basic validation to ensure cache matches current data size
            if len(train_feats) == len(train_df) and len(test_feats) == len(test_df):
                return train_feats, test_feats
            else:
                print("Cached features size mismatch. Recomputing...")
        except Exception as e:
            print(f"Error loading cache: {e}. Recomputing...")

    print("Computing global spatial features...")
    encoder = SpatialTargetEncoder()

    # 1. Generate Training Features (CV)
    # This ensures no leakage for the training set
    print("Generating training features (K-Fold CV)...")
    train_route_vals = encoder.fit_transform_cv(train_df)
    train_feats = pd.DataFrame(
        {"key": train_df["key"].values, "route_avg_fare": train_route_vals}
    )

    # 2. Generate Test Features (Full Fit)
    # We fit on the entire training set to get the best stats for test
    print("Fitting encoder on full training set for inference...")
    encoder.fit(train_df)

    print("Generating test features...")
    test_route_vals = encoder.transform(test_df)
    test_feats = pd.DataFrame(
        {"key": test_df["key"].values, "route_avg_fare": test_route_vals}
    )

    # 3. Cache Results
    print(f"Saving generated features to {CACHE_DIR}...")
    train_feats.to_parquet(train_cache_path, index=False)
    test_feats.to_parquet(test_cache_path, index=False)

    return train_feats, test_feats
