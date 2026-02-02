import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from library.config import Config
from library.utils import (
    haversine_array,
    manhattan_array,
    rotate_coordinates,
    encode_geohash,
)


class GeohashTargetEncoder:
    """
    Implements Hierarchical Distributional Fingerprinting with Intersection-Filtered
    Vectorized Subtraction to generate robust spatial priors without data leakage.
    """

    def __init__(self, precisions=Config.GEOHASH_PRECISIONS):
        self.precisions = precisions
        self.global_stats = {}  # {precision: DataFrame}
        self.global_prior = {}  # {mean, std}

    def fit(self, wisdom_df):
        """
        Compute global statistics (Wisdom) from the strictly filtered background dataset.
        """
        print("Fitting GeohashTargetEncoder on Wisdom Set...")

        # Calculate global scalar prior for fallback
        self.global_prior["mean"] = wisdom_df["fare_amount"].mean()
        self.global_prior["std"] = wisdom_df["fare_amount"].std()

        # Work on a copy to avoid side effects
        df = wisdom_df.copy()
        df["fare_sq"] = df["fare_amount"] ** 2

        for p in self.precisions:
            # Encode
            gh_col = f"gh_{p}"
            df[gh_col] = encode_geohash(
                df["pickup_latitude"], df["pickup_longitude"], p
            )

            # Aggregate: Sum, SumSq, Count
            stats = df.groupby(gh_col).agg({"fare_amount": "sum", "fare_sq": "sum"})
            stats["count"] = df.groupby(gh_col).size()

            # Rename for clarity
            stats = stats.rename(
                columns={"fare_amount": "sum_fare", "fare_sq": "sum_sq"}
            )

            self.global_stats[p] = stats

        print("Encoder fitting complete.")
        return self

    def _compute_moments(self, sums, sum_sqs, counts):
        """
        Helper to compute Mean and Std from aggregates.
        """
        # Mean = Sum / Count
        means = np.divide(
            sums,
            counts,
            out=np.full_like(sums, self.global_prior["mean"]),
            where=counts > 0,
        )

        # Var = (SumSq / N) - Mean^2
        # Std = sqrt(Var)
        term1 = np.divide(
            sum_sqs,
            counts,
            out=np.full_like(
                sum_sqs, self.global_prior["std"] ** 2 + self.global_prior["mean"] ** 2
            ),
            where=counts > 0,
        )
        vars_ = term1 - means**2
        vars_ = np.maximum(vars_, 0)  # Clip negative due to float precision
        stds = np.sqrt(vars_)

        return means, stds

    def transform_inference(self, df):
        """
        Apply global fingerprints to validation/test sets (No subtraction).
        """
        df_out = df.copy()

        for p in self.precisions:
            # Encode
            ghs = encode_geohash(
                df_out["pickup_latitude"], df_out["pickup_longitude"], p
            )

            # Map global stats
            stats = self.global_stats[p]
            temp_df = pd.DataFrame({"gh": ghs}, index=df_out.index)
            merged = temp_df.merge(stats, left_on="gh", right_index=True, how="left")

            # Fill NaNs (unseen geohashes) with 0
            merged["sum_fare"] = merged["sum_fare"].fillna(0.0)
            merged["sum_sq"] = merged["sum_sq"].fillna(0.0)
            merged["count"] = merged["count"].fillna(0.0)

            # Compute Moments
            means, stds = self._compute_moments(
                merged["sum_fare"].values,
                merged["sum_sq"].values,
                merged["count"].values,
            )

            # Assign features
            df_out[f"geo_L{p}_mean"] = means
            df_out[f"geo_L{p}_std"] = stds
            df_out[f"geo_L{p}_count"] = merged["count"].values

        return df_out

    def transform_train(self, learner_df, n_folds=Config.NUM_FOLDS):
        """
        Apply fingerprints to Learner set using Intersection-Filtered Vectorized Subtraction.
        """
        print(f"Transforming Learner Set with {n_folds}-Fold Subtraction...")
        df_out = learner_df.copy()

        # Initialize feature columns
        for p in self.precisions:
            df_out[f"geo_L{p}_mean"] = np.nan
            df_out[f"geo_L{p}_std"] = np.nan
            df_out[f"geo_L{p}_count"] = np.nan

        # 1. Identify rows in Learner that meet Strict Wisdom Criteria
        # We must replicate the logic to know which rows contributed to the Global Stats
        dists = haversine_array(
            df_out["pickup_latitude"].values,
            df_out["pickup_longitude"].values,
            df_out["dropoff_latitude"].values,
            df_out["dropoff_longitude"].values,
        )

        mask_fare_range = (df_out["fare_amount"] >= Config.WISDOM_MIN_FARE) & (
            df_out["fare_amount"] <= Config.WISDOM_MAX_FARE
        )
        mask_valid_dist = dists > 0.001

        mask_price_per_km = np.zeros(len(df_out), dtype=bool)
        mask_price_per_km[mask_valid_dist] = (
            df_out.loc[mask_valid_dist, "fare_amount"] / dists[mask_valid_dist]
        ) <= Config.WISDOM_MAX_FARE_PER_KM

        # This mask identifies rows that are "Wisdom Valid"
        is_wisdom = mask_fare_range & mask_valid_dist & mask_price_per_km

        # Pre-calculate Geohashes for all rows to save time in loop
        gh_maps = {}
        for p in self.precisions:
            gh_maps[p] = np.array(
                encode_geohash(df_out["pickup_latitude"], df_out["pickup_longitude"], p)
            )

        # Pre-calc squares
        fares = df_out["fare_amount"].values
        fares_sq = fares**2

        # K-Fold Loop
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=Config.SEED)

        for fold, (train_idx, val_idx) in enumerate(kf.split(df_out)):
            # val_idx represents the current fold we are generating features for

            # Identify which rows in this fold contribute to the leakage (Intersection)
            fold_wisdom_mask = is_wisdom.iloc[val_idx].values
            valid_indices = val_idx[fold_wisdom_mask]

            for p in self.precisions:
                global_stats = self.global_stats[p]

                # A. Compute Fold Correction Stats (Aggregates of Intersection)
                current_ghs = gh_maps[p][valid_indices]
                current_fares = fares[valid_indices]
                current_fares_sq = fares_sq[valid_indices]

                fold_correction = pd.DataFrame(
                    {"gh": current_ghs, "f": current_fares, "f2": current_fares_sq}
                )

                correction_agg = fold_correction.groupby("gh").agg(
                    {
                        "f": "sum",
                        "f2": "sum",
                    }
                )
                correction_agg["c"] = fold_correction.groupby("gh").size()

                # B. Map Global Stats to ALL rows in the fold
                fold_ghs = gh_maps[p][val_idx]
                fold_rows_df = pd.DataFrame({"gh": fold_ghs}, index=val_idx)

                # Join Global
                fold_rows_df = fold_rows_df.merge(
                    global_stats, left_on="gh", right_index=True, how="left"
                )
                fold_rows_df[["sum_fare", "sum_sq", "count"]] = fold_rows_df[
                    ["sum_fare", "sum_sq", "count"]
                ].fillna(0)

                # Join Correction
                fold_rows_df = fold_rows_df.merge(
                    correction_agg, left_on="gh", right_index=True, how="left"
                )
                fold_rows_df[["f", "f2", "c"]] = fold_rows_df[["f", "f2", "c"]].fillna(
                    0
                )

                # C. Perform Subtraction (Global - Correction)
                rest_sum = fold_rows_df["sum_fare"] - fold_rows_df["f"]
                rest_sq = fold_rows_df["sum_sq"] - fold_rows_df["f2"]
                rest_count = fold_rows_df["count"] - fold_rows_df["c"]

                # Clip to 0 to avoid numerical noise
                rest_count = np.maximum(rest_count, 0)
                rest_sum = np.maximum(rest_sum, 0)
                rest_sq = np.maximum(rest_sq, 0)

                # D. Compute Moments
                means, stds = self._compute_moments(
                    rest_sum.values, rest_sq.values, rest_count.values
                )

                # E. Assign back
                df_out.loc[val_idx, f"geo_L{p}_mean"] = means
                df_out.loc[val_idx, f"geo_L{p}_std"] = stds
                df_out.loc[val_idx, f"geo_L{p}_count"] = rest_count.values

        return df_out


def add_geometric_features(df):
    """
    Adds explicit geometric and temporal features to the dataframe.
    """
    print("Adding geometric features...")
    df = df.copy()

    # 1. Temporal Features
    # Ensure datetime format
    if not np.issubdtype(df["pickup_datetime"].dtype, np.datetime64):
        df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], utc=True)

    df["hour"] = df["pickup_datetime"].dt.hour
    df["year"] = df["pickup_datetime"].dt.year
    df["dayofweek"] = df["pickup_datetime"].dt.dayofweek

    # 2. Coordinate Arrays
    plat = df["pickup_latitude"].values
    plon = df["pickup_longitude"].values
    dlat = df["dropoff_latitude"].values
    dlon = df["dropoff_longitude"].values

    # 3. Distance Metrics
    df["dist_haversine"] = haversine_array(plat, plon, dlat, dlon)
    df["dist_manhattan"] = manhattan_array(plat, plon, dlat, dlon)

    # 4. Rotated Coordinates (29 degrees for NYC grid alignment)
    plat_rot, plon_rot = rotate_coordinates(plat, plon, 29)
    dlat_rot, dlon_rot = rotate_coordinates(dlat, dlon, 29)

    df["pickup_lat_rot"] = plat_rot
    df["pickup_lon_rot"] = plon_rot
    df["dropoff_lat_rot"] = dlat_rot
    df["dropoff_lon_rot"] = dlon_rot

    return df


def process_and_cache_data(
    wisdom_df, learner_df, val_df, test_df, load_cached_data=True
):
    """
    Orchestrates the feature engineering pipeline.
    Checks cache, runs encoding/feature generation, and saves results.
    """
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    f_train_path = os.path.join(cache_dir, "featurized_train.parquet")
    f_val_path = os.path.join(cache_dir, "featurized_val.parquet")
    f_test_path = os.path.join(cache_dir, "featurized_test.parquet")

    # 1. Try Loading from Cache
    if load_cached_data:
        if (
            os.path.exists(f_train_path)
            and os.path.exists(f_val_path)
            and os.path.exists(f_test_path)
        ):
            print("Loading featurized data from cache...")
            return (
                pd.read_parquet(f_train_path),
                pd.read_parquet(f_val_path),
                pd.read_parquet(f_test_path),
            )

    print("Cache missing or refresh requested. Generating features from scratch...")

    # 2. Fit Encoder on Wisdom Set
    encoder = GeohashTargetEncoder()
    encoder.fit(wisdom_df)

    # 3. Transform Learner (Training Mode with Subtraction)
    learner_feat = encoder.transform_train(learner_df)

    # 4. Transform Val & Test (Inference Mode)
    val_feat = encoder.transform_inference(val_df)
    test_feat = encoder.transform_inference(test_df)

    # 5. Add Geometric Features
    learner_feat = add_geometric_features(learner_feat)
    val_feat = add_geometric_features(val_feat)
    test_feat = add_geometric_features(test_feat)

    # 6. Cache Results
    print("Caching featurized data to working directory...")
    learner_feat.to_parquet(f_train_path, index=False)
    val_feat.to_parquet(f_val_path, index=False)
    test_feat.to_parquet(f_test_path, index=False)

    return learner_feat, val_feat, test_feat
