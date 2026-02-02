import os
import pandas as pd
import numpy as np
from library import config


def engineer_ratios(df):
    """
    Generates ratio-based features to capture relative user engagement
    and reputation signals.
    """
    # Epsilon to prevent division by zero
    epsilon = 1e-6

    # 1. Upvote Ratio at Request
    # Data provides:
    #   A = upvotes_minus_downvotes (Net)
    #   B = upvotes_plus_downvotes (Total)
    #   Upvotes = (B + A) / 2
    #   Ratio = Upvotes / B

    if (
        "requester_upvotes_plus_downvotes_at_request" in df.columns
        and "requester_upvotes_minus_downvotes_at_request" in df.columns
    ):

        total_votes = df["requester_upvotes_plus_downvotes_at_request"]
        net_votes = df["requester_upvotes_minus_downvotes_at_request"]

        # Calculate derived upvotes (approximate if odd, but floats handle it)
        upvotes = (total_votes + net_votes) / 2.0

        df["requester_upvote_ratio_at_request"] = upvotes / (total_votes + epsilon)

    # 2. Comment to Post Ratio (Global)
    if (
        "requester_number_of_comments_at_request" in df.columns
        and "requester_number_of_posts_at_request" in df.columns
    ):

        comments = df["requester_number_of_comments_at_request"]
        posts = df["requester_number_of_posts_at_request"]
        df["requester_global_comment_to_post_ratio"] = comments / (posts + epsilon)

    # 3. RAOP Engagement Ratio
    if (
        "requester_number_of_comments_in_raop_at_request" in df.columns
        and "requester_number_of_posts_on_raop_at_request" in df.columns
    ):

        raop_comments = df["requester_number_of_comments_in_raop_at_request"]
        raop_posts = df["requester_number_of_posts_on_raop_at_request"]
        df["requester_raop_comment_to_post_ratio"] = raop_comments / (
            raop_posts + epsilon
        )

    return df


def load_and_clean_data(load_cached_data=True):
    """
    Loads data, performs cleaning, feature engineering, and leakage removal.
    Implements caching to parquet files.
    """
    # Ensure cache directory exists
    os.makedirs(config.CACHE_DIR, exist_ok=True)

    # Define cache paths
    train_cache = os.path.join(config.CACHE_DIR, "train_processed.parquet")
    val_cache = os.path.join(config.CACHE_DIR, "val_processed.parquet")
    test_cache = os.path.join(config.CACHE_DIR, "test_processed.parquet")

    # Check cache
    if load_cached_data:
        if (
            os.path.exists(train_cache)
            and os.path.exists(val_cache)
            and os.path.exists(test_cache)
        ):
            print("Loading processed data from cache...")
            train_df = pd.read_parquet(train_cache)
            val_df = pd.read_parquet(val_cache)
            test_df = pd.read_parquet(test_cache)
            return train_df, val_df, test_df
        else:
            print("Cache miss. Processing data from scratch...")
    else:
        print("Ignoring cache. Processing data from scratch...")

    # Load raw metadata
    print("Loading raw CSVs...")
    train_df = pd.read_csv(config.TRAIN_DATA_PATH)
    val_df = pd.read_csv(config.VAL_DATA_PATH)
    test_df = pd.read_csv(config.TEST_DATA_PATH)

    # Apply Feature Engineering
    print("Engineering features...")
    train_df = engineer_ratios(train_df)
    val_df = engineer_ratios(val_df)
    test_df = engineer_ratios(test_df)

    # Identify columns to drop based on leakage suffixes
    # We scan train_df for candidates
    all_cols = train_df.columns.tolist()
    leakage_cols = [
        c
        for c in all_cols
        if any(c.endswith(suffix) for suffix in config.LEAKAGE_SUFFIXES)
    ]

    # Identify explicit exclusions
    excluded_cols = [c for c in config.EXCLUDED_COLS if c in all_cols]

    # Combine drop lists
    cols_to_drop = set(leakage_cols + excluded_cols)

    # Helper to select valid columns
    def select_valid_columns(df):
        # Start with all columns
        current_cols = set(df.columns)

        # Remove leakage/excluded
        valid_cols = current_cols - cols_to_drop

        # Select subset of DF
        subset = df[list(valid_cols)].copy()

        # Keep only Numeric and Boolean columns, plus ID and Text
        # We explicitly include ID and Text to ensure they aren't dropped by select_dtypes
        # We also include Target if present

        # 1. Identify numeric/bool
        numeric_subset = subset.select_dtypes(include=[np.number, bool])
        keep_cols = set(numeric_subset.columns)

        # 2. Add mandatory columns if they exist in the subset
        if config.ID_COL in subset.columns:
            keep_cols.add(config.ID_COL)
        if config.TEXT_COL in subset.columns:
            keep_cols.add(config.TEXT_COL)
        if config.TARGET_COL in subset.columns:
            keep_cols.add(config.TARGET_COL)

        return subset[list(keep_cols)]

    # Process splits
    train_proc = select_valid_columns(train_df)
    val_proc = select_valid_columns(val_df)
    test_proc = select_valid_columns(test_df)

    # Enforce Feature Intersection (Strict alignment)
    # Get feature sets (excluding ID, Target, Text)
    def get_feats(df):
        return set(df.columns) - {config.ID_COL, config.TARGET_COL, config.TEXT_COL}

    train_feats = get_feats(train_proc)
    val_feats = get_feats(val_proc)
    test_feats = get_feats(test_proc)

    # Intersection
    common_feats = list(train_feats.intersection(val_feats).intersection(test_feats))
    common_feats.sort()  # Deterministic order

    print(f"Selected {len(common_feats)} common numerical features.")

    # Reconstruct DataFrames with ordered columns
    def finalize_df(df, feats):
        # Base columns
        cols = [config.ID_COL, config.TEXT_COL] + feats
        # Add target if present
        if config.TARGET_COL in df.columns:
            cols.append(config.TARGET_COL)
        return df[cols].copy()

    train_final = finalize_df(train_proc, common_feats)
    val_final = finalize_df(val_proc, common_feats)
    test_final = finalize_df(test_proc, common_feats)

    # Imputation
    # Calculate median on TRAIN only
    print("Imputing missing values...")
    medians = train_final[common_feats].median()

    train_final[common_feats] = train_final[common_feats].fillna(medians)
    val_final[common_feats] = val_final[common_feats].fillna(medians)
    test_final[common_feats] = test_final[common_feats].fillna(medians)

    # Fill Text NaNs
    train_final[config.TEXT_COL] = train_final[config.TEXT_COL].fillna("")
    val_final[config.TEXT_COL] = val_final[config.TEXT_COL].fillna("")
    test_final[config.TEXT_COL] = test_final[config.TEXT_COL].fillna("")

    # Save to cache
    print("Saving to cache...")
    train_final.to_parquet(train_cache)
    val_final.to_parquet(val_cache)
    test_final.to_parquet(test_cache)

    return train_final, val_final, test_final
