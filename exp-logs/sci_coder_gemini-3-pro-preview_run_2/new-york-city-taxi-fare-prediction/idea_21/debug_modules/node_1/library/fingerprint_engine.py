import os
import gc
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from library.config import CACHE_DIR, GEOHASH_LEVELS, STRICT_FILTER, SEED, NYC_BBOX
from library.geo_utils import vectorized_geohash

# Threshold for "Heavy Tail" probability feature
HIGH_FARE_THRESHOLD = 50.0


def _calculate_moments(sum_fare, sum_sq_fare, count, high_count):
    """
    Computes statistical moments (Mean, Std, Tail Prob) from aggregates.
    Handles division by zero safely.
    """
    # Initialize with NaNs
    mean = np.full_like(sum_fare, np.nan)
    std = np.full_like(sum_fare, np.nan)
    tail_prob = np.full_like(sum_fare, np.nan)

    # Mask for valid counts
    valid_mask = count > 0

    if not np.any(valid_mask):
        return mean, std, tail_prob

    # Compute Mean
    mean[valid_mask] = sum_fare[valid_mask] / count[valid_mask]

    # Compute Variance -> Std
    # Var = E[X^2] - (E[X])^2
    # Var = (SumSq / N) - (Sum / N)^2
    # Numerical stability: Clip variance to >= 0
    avg_sq = sum_sq_fare[valid_mask] / count[valid_mask]
    mean_sq = mean[valid_mask] ** 2
    variance = np.maximum(0, avg_sq - mean_sq)
    std[valid_mask] = np.sqrt(variance)

    # Compute Tail Probability
    tail_prob[valid_mask] = high_count[valid_mask] / count[valid_mask]

    return mean, std, tail_prob


def compute_global_stats(
    wisdom_df: pd.DataFrame, load_cached_data: bool = True
) -> dict:
    """
    Computes global sufficient statistics (Sum, SumSq, Count, HighCount)
    for the Wisdom Set at all Geohash levels.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Check if all levels are cached
    all_cached = True
    cached_stats = {}

    for level in GEOHASH_LEVELS:
        path = os.path.join(CACHE_DIR, f"global_stats_L{level}.parquet")
        if load_cached_data and os.path.exists(path):
            cached_stats[level] = pd.read_parquet(path)
        else:
            all_cached = False
            break

    if all_cached:
        print("Loading cached Global Wisdom Stats...")
        return cached_stats

    print("Computing Global Wisdom Stats from scratch...")

    # Pre-calculate common columns to save time
    # We need fare, fare^2, and is_high_fare indicator
    fares = wisdom_df["fare_amount"].values
    fares_sq = fares**2
    is_high = (fares > HIGH_FARE_THRESHOLD).astype(np.int32)

    lats = wisdom_df["pickup_latitude"].values
    lons = wisdom_df["pickup_longitude"].values

    stats_dict = {}

    for level in GEOHASH_LEVELS:
        print(f"  Aggregating Geohash Level {level}...")

        # 1. Compute Geohash
        gh_ids = vectorized_geohash(lats, lons, level)

        # 2. Create temporary dataframe for aggregation
        # Using pandas groupby is efficient enough for 55M rows if we limit columns
        temp_df = pd.DataFrame(
            {
                "gh": gh_ids,
                "sum_fare": fares,
                "sum_sq_fare": fares_sq,
                "high_count": is_high,
                "count": 1,
            }
        )

        # 3. Aggregate
        agg_df = temp_df.groupby("gh").sum().reset_index()

        # Optimize types
        agg_df["gh"] = agg_df["gh"].astype(np.int64)
        agg_df["count"] = agg_df["count"].astype(np.int32)
        agg_df["high_count"] = agg_df["high_count"].astype(np.int32)
        agg_df["sum_fare"] = agg_df["sum_fare"].astype(np.float32)
        agg_df["sum_sq_fare"] = agg_df["sum_sq_fare"].astype(np.float32)

        # Index by geohash for fast lookup
        agg_df = agg_df.set_index("gh")

        # Save
        save_path = os.path.join(CACHE_DIR, f"global_stats_L{level}.parquet")
        agg_df.to_parquet(save_path)

        stats_dict[level] = agg_df

        del temp_df, gh_ids
        gc.collect()

    return stats_dict


def get_oof_fingerprints(
    learner_df: pd.DataFrame,
    wisdom_stats: dict,
    n_splits: int = 5,
    load_cached_data: bool = True,
) -> pd.DataFrame:
    """
    Generates Distributional Fingerprints for the Learner Set using
    Conditional Vectorized Subtraction to prevent leakage.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, "featurized_train.parquet")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached OOF Fingerprints from {cache_path}...")
        return pd.read_parquet(cache_path)

    print(f"Generating OOF Fingerprints (K={n_splits})...")

    # Initialize result columns
    result_df = learner_df.copy()

    # Initialize feature columns with NaNs
    for level in GEOHASH_LEVELS:
        result_df[f"L{level}_mean"] = np.nan
        result_df[f"L{level}_std"] = np.nan
        result_df[f"L{level}_prob"] = np.nan

    # Pre-compute distance for strict filter check inside the loop
    # We need to identify which learner rows would have been in the wisdom set
    # Strict Filter: min_fare <= fare <= max_fare AND fare/km <= max_fare_per_km
    from library.geo_utils import compute_haversine

    dists = compute_haversine(
        learner_df["pickup_latitude"].values,
        learner_df["pickup_longitude"].values,
        learner_df["dropoff_latitude"].values,
        learner_df["dropoff_longitude"].values,
    )

    # Avoid div by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        fare_per_km = learner_df["fare_amount"].values / dists

    # Boolean mask: True if the row satisfies STRICT criteria
    is_strict_valid = (
        (learner_df["fare_amount"].values >= STRICT_FILTER["min_fare"])
        & (learner_df["fare_amount"].values <= STRICT_FILTER["max_fare"])
        & (fare_per_km <= STRICT_FILTER["max_fare_per_km"])
    )

    # K-Fold Split
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    for fold_idx, (_, val_idx) in enumerate(kf.split(learner_df)):
        # print(f"  Processing Fold {fold_idx + 1}/{n_splits}...")

        # Subset for this fold
        fold_lats = learner_df.iloc[val_idx]["pickup_latitude"].values
        fold_lons = learner_df.iloc[val_idx]["pickup_longitude"].values
        fold_fares = learner_df.iloc[val_idx]["fare_amount"].values

        # Identify which rows in this fold are "Strict" (contributed to Global Stats)
        fold_strict_mask = is_strict_valid[val_idx]

        # We only need to subtract stats for the strict rows
        strict_fares = fold_fares[fold_strict_mask]
        strict_fares_sq = strict_fares**2
        strict_high = (strict_fares > HIGH_FARE_THRESHOLD).astype(np.int32)

        # Iterate over levels
        for level in GEOHASH_LEVELS:
            # 1. Get Global Stats
            global_agg = wisdom_stats[level]

            # 2. Compute Geohashes for ALL rows in fold (to map results later)
            fold_gh = vectorized_geohash(fold_lats, fold_lons, level)

            # 3. Compute Geohashes for STRICT rows (to aggregate for subtraction)
            strict_lats = fold_lats[fold_strict_mask]
            strict_lons = fold_lons[fold_strict_mask]
            strict_gh = vectorized_geohash(strict_lats, strict_lons, level)

            # 4. Aggregate Fold Stats (Strict Only)
            temp_fold = pd.DataFrame(
                {
                    "gh": strict_gh,
                    "sum_fare": strict_fares,
                    "sum_sq_fare": strict_fares_sq,
                    "high_count": strict_high,
                    "count": 1,
                }
            )
            fold_agg = temp_fold.groupby("gh").sum()

            # 5. Perform Subtraction (Vectorized via Index Alignment)
            # We want: Global - Fold
            # Map global stats to the unique geohashes found in this fold
            unique_gh_in_fold = np.unique(fold_gh)

            # Extract relevant global stats
            # Reindex fills missing with NaN, but we want 0 if global doesn't exist (unlikely but possible)
            current_global = global_agg.reindex(unique_gh_in_fold).fillna(0)
            current_fold = fold_agg.reindex(unique_gh_in_fold).fillna(0)

            # Subtract
            loo_sum = current_global["sum_fare"] - current_fold["sum_fare"]
            loo_sum_sq = current_global["sum_sq_fare"] - current_fold["sum_sq_fare"]
            loo_count = current_global["count"] - current_fold["count"]
            loo_high = current_global["high_count"] - current_fold["high_count"]

            # 6. Calculate Moments
            m, s, p = _calculate_moments(
                loo_sum.values, loo_sum_sq.values, loo_count.values, loo_high.values
            )

            # 7. Map back to fold rows
            # Create a lookup series
            lookup_mean = pd.Series(m, index=unique_gh_in_fold)
            lookup_std = pd.Series(s, index=unique_gh_in_fold)
            lookup_prob = pd.Series(p, index=unique_gh_in_fold)

            # Assign
            # Using map is faster than merge for single columns
            # We map 'fold_gh' (all rows in fold) to the computed LOO stats
            result_df.loc[val_idx, f"L{level}_mean"] = (
                pd.Series(fold_gh).map(lookup_mean).values
            )
            result_df.loc[val_idx, f"L{level}_std"] = (
                pd.Series(fold_gh).map(lookup_std).values
            )
            result_df.loc[val_idx, f"L{level}_prob"] = (
                pd.Series(fold_gh).map(lookup_prob).values
            )

    print(f"Saving OOF Fingerprints to {cache_path}...")
    result_df.to_parquet(cache_path)
    return result_df


def get_test_fingerprints(
    test_df: pd.DataFrame,
    wisdom_stats: dict,
    load_cached_data: bool = True,
    prefix: str = "test",
) -> pd.DataFrame:
    """
    Generates Distributional Fingerprints for the Test/Validation Set.
    Directly maps Global Wisdom Stats (no subtraction needed).
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"featurized_{prefix}.parquet")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached {prefix} Fingerprints from {cache_path}...")
        return pd.read_parquet(cache_path)

    print(f"Generating {prefix} Fingerprints...")

    result_df = test_df.copy()

    lats = result_df["pickup_latitude"].values
    lons = result_df["pickup_longitude"].values

    for level in GEOHASH_LEVELS:
        # 1. Compute Geohash
        gh_ids = vectorized_geohash(lats, lons, level)

        # 2. Get Global Stats
        global_agg = wisdom_stats[level]

        # 3. Map Stats to Rows
        # We need to handle keys that might not exist in wisdom stats (new locations)
        # reindex will introduce NaNs for missing keys, which is correct (unknown prior)

        # To do this efficiently:
        # Get the stats for the geohashes present in test
        matched_stats = global_agg.reindex(gh_ids)

        # Extract arrays
        sum_f = matched_stats["sum_fare"].values
        sum_sq = matched_stats["sum_sq_fare"].values
        cnt = matched_stats["count"].values
        high_cnt = matched_stats["high_count"].values

        # 4. Calculate Moments
        m, s, p = _calculate_moments(sum_f, sum_sq, cnt, high_cnt)

        result_df[f"L{level}_mean"] = m
        result_df[f"L{level}_std"] = s
        result_df[f"L{level}_prob"] = p

    print(f"Saving {prefix} Fingerprints to {cache_path}...")
    result_df.to_parquet(cache_path)
    return result_df
