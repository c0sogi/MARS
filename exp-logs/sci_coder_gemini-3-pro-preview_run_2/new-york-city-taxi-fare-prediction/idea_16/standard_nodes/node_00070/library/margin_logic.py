import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from library.config import Config
from library.utils import encode_geohash, haversine_distance
from library.data_processor import filter_strict


def _calculate_raw_stats(df):
    """
    Computes raw sums and counts for L1 (Micro) and L2 (Meso) levels,
    plus global fare and distance totals.
    Used for both Global Stats generation and Fold Stats subtraction.
    """
    # Ensure distance is calculated for physics baseline (L3)
    dists = haversine_distance(
        df["pickup_latitude"].values,
        df["pickup_longitude"].values,
        df["dropoff_latitude"].values,
        df["dropoff_longitude"].values,
    )

    # We need fare_amount for sums
    fares = df["fare_amount"].values

    # Global Totals
    total_fare = np.sum(fares)
    total_dist = np.sum(dists)

    # --- L1 Stats (Micro) ---
    # Encode
    p_l1 = encode_geohash(
        df["pickup_latitude"],
        df["pickup_longitude"],
        precision=Config.GEOHASH_PRECISION_MICRO,
    )
    d_l1 = encode_geohash(
        df["dropoff_latitude"],
        df["dropoff_longitude"],
        precision=Config.GEOHASH_PRECISION_MICRO,
    )

    # Create a DataFrame for grouping
    # We use a combined key or just group by both. Grouping by both is safer.
    temp_df = pd.DataFrame({"p_hash": p_l1, "d_hash": d_l1, "fare": fares})

    # GroupBy
    l1_stats = temp_df.groupby(["p_hash", "d_hash"])["fare"].agg(["sum", "count"])

    # --- L2 Stats (Meso) ---
    p_l2 = encode_geohash(
        df["pickup_latitude"],
        df["pickup_longitude"],
        precision=Config.GEOHASH_PRECISION_MESO,
    )
    d_l2 = encode_geohash(
        df["dropoff_latitude"],
        df["dropoff_longitude"],
        precision=Config.GEOHASH_PRECISION_MESO,
    )

    temp_df["p_hash"] = p_l2
    temp_df["d_hash"] = d_l2

    l2_stats = temp_df.groupby(["p_hash", "d_hash"])["fare"].agg(["sum", "count"])

    return l1_stats, l2_stats, total_fare, total_dist


def compute_global_stats(wisdom_df, load_cached_data=True):
    """
    Computes (or loads) the Global Statistics from the Wisdom Set.
    Returns raw sums/counts to allow for vectorized subtraction later.
    """
    l1_path = os.path.join(Config.WORKING_DIR, "global_stats_l1.parquet")
    l2_path = os.path.join(Config.WORKING_DIR, "global_stats_l2.parquet")
    l3_path = os.path.join(Config.WORKING_DIR, "global_stats_l3.npy")

    if (
        load_cached_data
        and os.path.exists(l1_path)
        and os.path.exists(l2_path)
        and os.path.exists(l3_path)
    ):
        print("Loading cached Global Stats...")
        l1_stats = pd.read_parquet(l1_path)
        l2_stats = pd.read_parquet(l2_path)
        totals = np.load(l3_path)
        return l1_stats, l2_stats, totals[0], totals[1]

    print("Computing Global Stats from Wisdom Set...")
    l1_stats, l2_stats, total_fare, total_dist = _calculate_raw_stats(wisdom_df)

    # Save
    print("Saving Global Stats to cache...")
    l1_stats.to_parquet(l1_path)
    l2_stats.to_parquet(l2_path)
    np.save(l3_path, np.array([total_fare, total_dist]))

    return l1_stats, l2_stats, total_fare, total_dist


def _apply_waterfall(df, l1_means, l2_means, l3_rate):
    """
    Applies the Hierarchical Waterfall Logic to a dataframe.

    Args:
        df: DataFrame with coordinates.
        l1_means: DataFrame with index (p_hash, d_hash) and cols (mean, count).
        l2_means: DataFrame with index (p_hash, d_hash) and cols (mean, count).
        l3_rate: Float (Price per KM).

    Returns:
        np.array: The constructed margin.
    """
    # 1. Calculate L3 Baseline (Physics)
    dists = haversine_distance(
        df["pickup_latitude"].values,
        df["pickup_longitude"].values,
        df["dropoff_latitude"].values,
        df["dropoff_longitude"].values,
    )
    margin_l3 = dists * l3_rate

    # 2. Prepare Keys
    p_l1 = encode_geohash(
        df["pickup_latitude"],
        df["pickup_longitude"],
        precision=Config.GEOHASH_PRECISION_MICRO,
    )
    d_l1 = encode_geohash(
        df["dropoff_latitude"],
        df["dropoff_longitude"],
        precision=Config.GEOHASH_PRECISION_MICRO,
    )

    p_l2 = encode_geohash(
        df["pickup_latitude"],
        df["pickup_longitude"],
        precision=Config.GEOHASH_PRECISION_MESO,
    )
    d_l2 = encode_geohash(
        df["dropoff_latitude"],
        df["dropoff_longitude"],
        precision=Config.GEOHASH_PRECISION_MESO,
    )

    # 3. Map L1 Stats
    # Create temp df for joining
    keys_l1 = pd.DataFrame({"p_hash": p_l1, "d_hash": d_l1})
    merged_l1 = keys_l1.join(l1_means, on=["p_hash", "d_hash"], how="left")

    # Fill missing with 0 count
    l1_count = merged_l1["count"].fillna(0).values
    l1_val = merged_l1["mean"].fillna(0).values

    # 4. Map L2 Stats
    keys_l2 = pd.DataFrame({"p_hash": p_l2, "d_hash": d_l2})
    merged_l2 = keys_l2.join(l2_means, on=["p_hash", "d_hash"], how="left")

    l2_count = merged_l2["count"].fillna(0).values
    l2_val = merged_l2["mean"].fillna(0).values

    # 5. Waterfall Logic
    # If L1 count > thresh -> L1
    # Else If L2 count > thresh -> L2
    # Else -> L3

    thresh = Config.MARGIN_COUNT_THRESHOLD

    final_margin = np.where(
        l1_count > thresh, l1_val, np.where(l2_count > thresh, l2_val, margin_l3)
    )

    # Add input features for Soft Integration (optional, but good for the model)
    # We return the margin, but the calling function might want to add these as features.
    # For now, we just return the scalar margin.

    return final_margin


def construct_train_margins(learner_df, wisdom_df, load_cached_data=True):
    """
    Constructs margins for the training set using K-Fold Vectorized Subtraction.
    """
    cache_path = os.path.join(Config.WORKING_DIR, "learner_with_margins.parquet")

    if load_cached_data and os.path.exists(cache_path):
        print("Loading cached Learner Set with Margins...")
        return pd.read_parquet(cache_path)

    print("Constructing Train Margins (K-Fold Vectorized Subtraction)...")

    # 1. Get Global Stats (Raw Sums/Counts)
    g_l1, g_l2, g_fare, g_dist = compute_global_stats(
        wisdom_df, load_cached_data=load_cached_data
    )

    # 2. Setup K-Fold
    # Reset index to ensure iloc works cleanly
    learner_df = learner_df.reset_index(drop=True)
    kf = KFold(n_splits=5, shuffle=True, random_state=Config.SEED)

    # Initialize margin column
    learner_df["margin"] = np.nan

    # 3. Iterate Folds
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(learner_df)):
        # val_idx represents the current chunk we want to predict margins for.
        # We need to subtract the contribution of these rows from the global stats
        # IF they were part of the wisdom set.

        fold_rows = learner_df.iloc[val_idx]

        # Identify rows that would have passed strict filtering (and thus are in Global Stats)
        fold_strict = filter_strict(fold_rows)

        # Calculate stats for this strict subset
        # If fold_strict is empty, subtraction is zero
        if len(fold_strict) > 0:
            f_l1, f_l2, f_fare, f_dist = _calculate_raw_stats(fold_strict)

            # Vectorized Subtraction (Global - Fold)
            # align and subtract, fill_value=0 implies if key not in fold, subtract 0
            loo_l1 = g_l1.sub(f_l1, fill_value=0)
            loo_l2 = g_l2.sub(f_l2, fill_value=0)
            loo_fare = g_fare - f_fare
            loo_dist = g_dist - f_dist
        else:
            loo_l1 = g_l1.copy()
            loo_l2 = g_l2.copy()
            loo_fare = g_fare
            loo_dist = g_dist

        # Clip to ensure no negative counts/sums due to float precision (though ints should be fine)
        loo_l1 = loo_l1.clip(lower=0)
        loo_l2 = loo_l2.clip(lower=0)

        # Compute Means/Rates for this fold's LOO stats
        # Avoid division by zero
        loo_l1["mean"] = np.divide(
            loo_l1["sum"],
            loo_l1["count"],
            out=np.zeros_like(loo_l1["sum"]),
            where=loo_l1["count"] != 0,
        )
        loo_l2["mean"] = np.divide(
            loo_l2["sum"],
            loo_l2["count"],
            out=np.zeros_like(loo_l2["sum"]),
            where=loo_l2["count"] != 0,
        )

        loo_rate = loo_fare / loo_dist if loo_dist > 0 else 0.0

        # Apply Waterfall to the fold rows
        margins = _apply_waterfall(fold_rows, loo_l1, loo_l2, loo_rate)

        # Assign back
        learner_df.iloc[val_idx, learner_df.columns.get_loc("margin")] = margins

    print("Saving Learner Set with Margins to cache...")
    learner_df.to_parquet(cache_path, index=False)

    return learner_df


def construct_test_margins(test_df, wisdom_df, load_cached_data=True):
    """
    Constructs margins for the test set using full Global Stats.
    """
    cache_path = os.path.join(Config.WORKING_DIR, "test_with_margins.parquet")

    if load_cached_data and os.path.exists(cache_path):
        print("Loading cached Test Set with Margins...")
        return pd.read_parquet(cache_path)

    print("Constructing Test Margins...")

    # 1. Get Global Stats
    g_l1, g_l2, g_fare, g_dist = compute_global_stats(
        wisdom_df, load_cached_data=load_cached_data
    )

    # 2. Compute Means/Rates
    g_l1["mean"] = np.divide(
        g_l1["sum"],
        g_l1["count"],
        out=np.zeros_like(g_l1["sum"]),
        where=g_l1["count"] != 0,
    )
    g_l2["mean"] = np.divide(
        g_l2["sum"],
        g_l2["count"],
        out=np.zeros_like(g_l2["sum"]),
        where=g_l2["count"] != 0,
    )

    global_rate = g_fare / g_dist if g_dist > 0 else 0.0

    # 3. Apply Waterfall
    test_df = test_df.copy()
    margins = _apply_waterfall(test_df, g_l1, g_l2, global_rate)

    # Handle NaNs in Margins (caused by missing coordinates in test/val set)
    # Calculate Global Mean Fare as fallback
    total_count = g_l1["count"].sum()
    global_mean = g_fare / total_count if total_count > 0 else 0.0

    # Fill NaNs with Global Mean
    margins = np.nan_to_num(margins, nan=global_mean)

    test_df["margin"] = margins

    print("Saving Test Set with Margins to cache...")
    test_df.to_parquet(cache_path, index=False)

    return test_df
