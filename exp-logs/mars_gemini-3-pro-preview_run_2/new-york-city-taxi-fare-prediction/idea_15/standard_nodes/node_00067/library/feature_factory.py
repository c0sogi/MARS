import os
import gc
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from library.config import Config
from library.geometry_utils import DistanceCalculator, GridIndexer
from library.data_loader import TaxiDataLoader


class FactorizedEncoder:
    """
    Implements Factorized Spatiotemporal Feature Engineering.
    Generates Spatial Base priors and Temporal Rate priors from a Wisdom Set,
    and applies them to Learner/Test sets with leakage prevention.
    """

    def __init__(self, n_splits=5):
        self.n_splits = n_splits
        self.global_stats = {}
        self.global_averages = {}

    def _generate_features(self, df):
        """Generates basic keys and metrics required for stats aggregation."""
        # Ensure coordinates are clamped to NYC box to prevent grid key explosion
        p_lat, p_lon = GridIndexer.clamp_coordinates(
            df["pickup_latitude"].values, df["pickup_longitude"].values
        )
        d_lat, d_lon = GridIndexer.clamp_coordinates(
            df["dropoff_latitude"].values, df["dropoff_longitude"].values
        )

        # Update df with clamped values for feature usage
        df["pickup_latitude"] = p_lat
        df["pickup_longitude"] = p_lon
        df["dropoff_latitude"] = d_lat
        df["dropoff_longitude"] = d_lon

        # Distance Calculation
        if "dist_km" not in df.columns:
            df["dist_km"] = DistanceCalculator.haversine(p_lat, p_lon, d_lat, d_lon)

        # Time features
        if not pd.api.types.is_datetime64_any_dtype(df["pickup_datetime"]):
            df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], utc=True)

        df["hour"] = df["pickup_datetime"].dt.hour
        df["weekday"] = df["pickup_datetime"].dt.dayofweek
        df["year"] = df["pickup_datetime"].dt.year

        # Spatial Keys (L5, L6, L7)
        # We need pickup and dropoff keys combined for Route stats
        p_l7 = GridIndexer.get_grid_key(p_lat, p_lon, "L7")
        d_l7 = GridIndexer.get_grid_key(d_lat, d_lon, "L7")
        df["route_L7"] = p_l7 + "_" + d_l7

        p_l6 = GridIndexer.get_grid_key(p_lat, p_lon, "L6")
        d_l6 = GridIndexer.get_grid_key(d_lat, d_lon, "L6")
        df["route_L6"] = p_l6 + "_" + d_l6

        p_l5 = GridIndexer.get_grid_key(p_lat, p_lon, "L5")
        d_l5 = GridIndexer.get_grid_key(d_lat, d_lon, "L5")
        df["route_L5"] = p_l5 + "_" + d_l5

        # Temporal Key: Pickup Neighborhood (L5) + Hour + Weekday
        # Construct as string for easy mapping
        df["temp_key"] = (
            p_l5 + "_" + df["hour"].astype(str) + "_" + df["weekday"].astype(str)
        )

        return df

    def fit_wisdom(self, wisdom_df):
        """
        Calculates Global Statistics from the strict Wisdom Set.
        """
        print("Generating Wisdom Stats...")
        df = self._generate_features(wisdom_df.copy())

        # Calculate Unit Rate (Fare per Km)
        # Avoid division by zero; strict filter handles most, but safe guard
        df["unit_rate"] = df["fare_amount"] / df["dist_km"].replace(0, np.nan)

        # 1. Spatial Base Stats (Routes)
        # We store Sum and Count for vector subtraction
        for level in ["L7", "L6", "L5"]:
            col = f"route_{level}"
            # GroupBy is expensive on 44M rows, but necessary.
            # We assume 220GB RAM is sufficient.
            grp = df.groupby(col)["fare_amount"].agg(["sum", "count"])
            self.global_stats[f"spatial_{level}"] = grp

        # 2. Temporal Rate Stats
        # Aggregating unit_rate
        grp_temp = df.groupby("temp_key")["unit_rate"].agg(["sum", "count"])
        self.global_stats["temporal"] = grp_temp

        # 3. Global Averages (Fallbacks)
        self.global_averages["fare"] = df["fare_amount"].mean()
        self.global_averages["rate"] = df["unit_rate"].mean()

        print("Wisdom Stats Generated.")
        # Clean up to free memory
        del df
        gc.collect()

    def transform_learner(self, learner_df):
        """
        Applies stats to Learner set using K-Fold Vectorized Subtraction.
        """
        print("Transforming Learner Set with Vectorized Subtraction...")
        df = self._generate_features(learner_df.copy())

        # Initialize feature columns with NaNs
        feature_cols = ["mean_fare_L7", "mean_fare_L6", "mean_fare_L5", "pred_rate"]
        for col in feature_cols:
            df[col] = np.nan

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=Config.SEED)

        # Iterate folds to compute "Rest of World" stats
        for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
            # val_idx is the current "fold" we are transforming.
            # We want to subtract this fold's contribution from the Global stats.

            fold_df = df.iloc[val_idx]

            # --- Spatial Features ---
            for level in ["L7", "L6", "L5"]:
                key_col = f"route_{level}"
                global_stat = self.global_stats[f"spatial_{level}"]

                # Get Global values for the keys in this fold
                g_sum = fold_df[key_col].map(global_stat["sum"]).fillna(0)
                g_cnt = fold_df[key_col].map(global_stat["count"]).fillna(0)

                # Calculate Fold values (Local contribution)
                fold_grp = fold_df.groupby(key_col)["fare_amount"].agg(["sum", "count"])
                f_sum = fold_df[key_col].map(fold_grp["sum"]).fillna(0)
                f_cnt = fold_df[key_col].map(fold_grp["count"]).fillna(0)

                # Subtract to get "Rest of World"
                rest_sum = g_sum - f_sum
                rest_cnt = g_cnt - f_cnt

                # Calculate Mean
                # Handle division by zero (where rest_cnt <= 0)
                with np.errstate(divide="ignore", invalid="ignore"):
                    val = rest_sum / rest_cnt

                # Sanitize: If rest_cnt <= 0, the result is invalid (Inf or Leakage).
                # Force to NaN so fallback logic takes over.
                if isinstance(val, (pd.Series, np.ndarray)):
                    val[rest_cnt <= 0] = np.nan

                # Assign to the dataframe slice
                df.loc[val_idx, f"mean_fare_{level}"] = val

            # --- Temporal Feature ---
            key_col = "temp_key"
            global_stat = self.global_stats["temporal"]

            # Calculate unit rate for fold rows to aggregate
            fold_rates = fold_df["fare_amount"] / fold_df["dist_km"].replace(0, np.nan)

            # Temporary DF for grouping
            tmp = pd.DataFrame({"k": fold_df[key_col], "r": fold_rates})
            fold_grp = tmp.groupby("k")["r"].agg(["sum", "count"])

            g_sum = fold_df[key_col].map(global_stat["sum"]).fillna(0)
            g_cnt = fold_df[key_col].map(global_stat["count"]).fillna(0)

            f_sum = fold_df[key_col].map(fold_grp["sum"]).fillna(0)
            f_cnt = fold_df[key_col].map(fold_grp["count"]).fillna(0)

            rest_sum = g_sum - f_sum
            rest_cnt = g_cnt - f_cnt

            with np.errstate(divide="ignore", invalid="ignore"):
                rate_val = rest_sum / rest_cnt

            # Sanitize invalid counts
            if isinstance(rate_val, (pd.Series, np.ndarray)):
                rate_val[rest_cnt <= 0] = np.nan

            df.loc[val_idx, "pred_rate"] = rate_val

        # Apply Cascading Fallback (Imputation)
        df = self._apply_fallbacks(df)

        return self._finalize_columns(df)

    def transform_static(self, df_in):
        """
        Applies global stats directly (for Test/Val sets).
        """
        print("Transforming Static Set...")
        df = self._generate_features(df_in.copy())

        # Spatial
        for level in ["L7", "L6", "L5"]:
            key_col = f"route_{level}"
            stats = self.global_stats[f"spatial_{level}"]

            # Map Sum and Count
            s = df[key_col].map(stats["sum"])
            c = df[key_col].map(stats["count"])

            # Mean
            df[f"mean_fare_{level}"] = s / c

        # Temporal
        stats = self.global_stats["temporal"]
        s = df["temp_key"].map(stats["sum"])
        c = df["temp_key"].map(stats["count"])
        df["pred_rate"] = s / c

        # Apply Cascading Fallback
        df = self._apply_fallbacks(df)

        return self._finalize_columns(df)

    def _apply_fallbacks(self, df):
        """Fills NaNs using a cascade: L7 -> L6 -> L5 -> Global Mean."""

        # Sanitize Infinite values to NaN so fillna can handle them.
        # This prevents XGBoost errors with Inf values.
        df = df.replace([np.inf, -np.inf], np.nan)

        # Spatial Fallback
        df["mean_fare_L7"] = df["mean_fare_L7"].fillna(df["mean_fare_L6"])
        df["mean_fare_L7"] = df["mean_fare_L7"].fillna(df["mean_fare_L5"])
        df["mean_fare_L7"] = df["mean_fare_L7"].fillna(self.global_averages["fare"])

        df["mean_fare_L6"] = df["mean_fare_L6"].fillna(df["mean_fare_L5"])
        df["mean_fare_L6"] = df["mean_fare_L6"].fillna(self.global_averages["fare"])

        df["mean_fare_L5"] = df["mean_fare_L5"].fillna(self.global_averages["fare"])

        # Temporal Fallback
        df["pred_rate"] = df["pred_rate"].fillna(self.global_averages["rate"])

        # Compute Expected Fare from Rate
        df["temporal_fare"] = df["pred_rate"] * df["dist_km"]

        # Final check for any remaining Infs (e.g. from temporal_fare calc if dist is Inf)
        df = df.replace([np.inf, -np.inf], np.nan)
        # Fill remaining NaNs (if any) with global mean fare
        df["temporal_fare"] = df["temporal_fare"].fillna(self.global_averages["fare"])

        return df

    def _finalize_columns(self, df):
        """Selects final feature set for the model."""
        cols = [
            "pickup_longitude",
            "pickup_latitude",
            "dropoff_longitude",
            "dropoff_latitude",
            "passenger_count",
            "dist_km",
            "hour",
            "year",
            "mean_fare_L7",
            "mean_fare_L6",
            "mean_fare_L5",
            "temporal_fare",
        ]
        return df[cols]


def process_data(load_cached_data=True, debug=False):
    """
    Main entry point for data processing.
    Handles caching, loading raw data, and executing the factorized encoding.
    """
    # Paths for cached artifacts
    path_train = Config.CACHE_PROCESSED_TRAIN
    path_val = Config.CACHE_PROCESSED_VAL
    path_test = Config.CACHE_PROCESSED_TEST

    # 1. Try Loading Cache
    if (
        load_cached_data
        and os.path.exists(path_train)
        and os.path.exists(path_val)
        and os.path.exists(path_test)
    ):
        print("Loading processed data from cache...")
        X_train = pd.read_parquet(path_train)
        X_val = pd.read_parquet(path_val)
        X_test = pd.read_parquet(path_test)

        # Separate targets and keys which were saved with the data
        y_train = X_train.pop("target")
        y_val = X_val.pop("target")
        test_keys = X_test.pop("key")

        return X_train, y_train, X_val, y_val, X_test, test_keys

    # 2. Process from Scratch
    print("Cache not found or reload forced. Processing from raw data...")

    # Load Raw Data
    loader = TaxiDataLoader()
    learner_df, wisdom_df, val_df, test_df = loader.load_datasets(
        load_cached_data=load_cached_data, debug=debug
    )

    # Initialize and Fit Encoder
    encoder = FactorizedEncoder(n_splits=5)
    encoder.fit_wisdom(wisdom_df)

    # Transform Datasets
    print("Processing Train (Learner)...")
    # Reset index to ensure KFold positional indices align with DataFrame labels
    learner_df = learner_df.reset_index(drop=True)
    X_train = encoder.transform_learner(learner_df)
    y_train = learner_df["fare_amount"]

    print("Processing Validation...")
    X_val = encoder.transform_static(val_df)
    y_val = val_df["fare_amount"]

    print("Processing Test...")
    X_test = encoder.transform_static(test_df)
    test_keys = test_df["key"]

    # 3. Save to Cache
    print("Saving processed data to cache...")
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Attach targets/keys temporarily for saving
    X_train_save = X_train.copy()
    X_train_save["target"] = y_train

    X_val_save = X_val.copy()
    X_val_save["target"] = y_val

    X_test_save = X_test.copy()
    X_test_save["key"] = test_keys

    X_train_save.to_parquet(path_train)
    X_val_save.to_parquet(path_val)
    X_test_save.to_parquet(path_test)

    return X_train, y_train, X_val, y_val, X_test, test_keys
