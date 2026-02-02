import os
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import process_target_residuals


def process_gnss_data(metadata_df, load_cached_data=True):
    """
    Loads, quantizes, and aggregates raw GNSS data for the drives listed in metadata_df.

    Args:
        metadata_df (pd.DataFrame): DataFrame containing 'drive_id', 'phone_name', and 'gnss_path'.
        load_cached_data (bool): If True, tries to load processed parquet files from cache.

    Returns:
        pd.DataFrame: Aggregated features with keys ['drive_id', 'phone_name', 'UnixTimeMillis'].
    """
    # Identify unique trips (drive + phone)
    trips = metadata_df[["drive_id", "phone_name", "gnss_path"]].drop_duplicates()

    processed_dfs = []

    for _, row in trips.iterrows():
        drive_id = row["drive_id"]
        phone_name = row["phone_name"]
        gnss_rel_path = row["gnss_path"]

        # Define cache path for this specific trip
        cache_filename = f"{drive_id}_{phone_name}_features.parquet"
        cache_path = os.path.join(Config.WORKING_DIR, cache_filename)

        # Try to load from cache
        if load_cached_data and os.path.exists(cache_path):
            try:
                df_features = pd.read_parquet(cache_path)
                processed_dfs.append(df_features)
                continue
            except Exception as e:
                print(f"Failed to load cache {cache_path}: {e}. Recomputing...")

        # Compute from scratch
        raw_path = os.path.join(Config.INPUT_DIR, gnss_rel_path)
        if not os.path.exists(raw_path):
            print(f"Warning: GNSS file not found: {raw_path}")
            continue

        try:
            # Load raw data
            df_raw = pd.read_csv(raw_path, usecols=Config.GNSS_RAW_COLS)

            # Temporal Quantization: Round utcTimeMillis to nearest second (1000ms)
            # This aligns raw measurements with the 1Hz ground truth
            df_raw["UnixTimeMillis"] = (df_raw["utcTimeMillis"] + 500) // 1000 * 1000

            # Boundary-Aware Aggregation
            # Group by the quantized timestamp and apply statistical moments
            agg_dict = Config.AGGREGATION_MAP
            df_agg = df_raw.groupby("UnixTimeMillis").agg(agg_dict)

            # Flatten MultiIndex columns (e.g., ('Cn0DbHz', 'mean') -> 'Cn0DbHz_mean')
            df_agg.columns = [f"{col[0]}_{col[1]}" for col in df_agg.columns]
            df_agg = df_agg.reset_index()

            # Add metadata keys
            df_agg["drive_id"] = drive_id
            df_agg["phone_name"] = phone_name

            # Ensure all expected columns exist (fill missing with NaN if a stat wasn't computable)
            for feature in Config.INPUT_FEATURES:
                if feature not in df_agg.columns:
                    df_agg[feature] = np.nan

            # Filter to keep only relevant columns and keys
            cols_to_keep = [
                "drive_id",
                "phone_name",
                "UnixTimeMillis",
            ] + Config.INPUT_FEATURES
            df_agg = df_agg[cols_to_keep]

            # Save to cache
            df_agg.to_parquet(cache_path, index=False)

            processed_dfs.append(df_agg)

        except Exception as e:
            print(f"Error processing GNSS for {drive_id}-{phone_name}: {e}")
            continue

    if not processed_dfs:
        # Return empty dataframe with correct schema if no data found
        cols = ["drive_id", "phone_name", "UnixTimeMillis"] + Config.INPUT_FEATURES
        return pd.DataFrame(columns=cols)

    # Concatenate all processed trips
    full_features = pd.concat(processed_dfs, ignore_index=True)
    return full_features


def get_data(metadata_df, load_cached_data=True):
    """
    Constructs the final dataset for training or inference.
    Merges metadata, aggregated GNSS features, and (optionally) ground truth targets.

    Args:
        metadata_df (pd.DataFrame): Metadata defining the split (train/val/test).
        load_cached_data (bool): Whether to use cached intermediate files.

    Returns:
        pd.DataFrame: The complete dataset ready for the model.
    """
    # 1. Load and Process Features
    print(f"Loading GNSS features for {len(metadata_df)} samples...")
    df_features = process_gnss_data(metadata_df, load_cached_data=load_cached_data)

    # 2. Merge Features with Metadata
    # Use left join to preserve all metadata rows (timestamps requested for prediction)
    # This handles cases where GNSS data might be missing for a specific second
    df_merged = pd.merge(
        metadata_df,
        df_features,
        on=["drive_id", "phone_name", "UnixTimeMillis"],
        how="left",
    )

    # 3. Handle Targets (if Training/Validation data)
    # Check if ground truth columns exist in metadata
    has_gt = "LatitudeDegrees" in metadata_df.columns

    if has_gt:
        print("Loading target residuals...")
        # process_target_residuals handles its own caching
        # It returns residuals for ALL train drives found in the input directory
        # We merge it with our specific split
        df_targets = process_target_residuals(
            metadata_df, load_cached_data=load_cached_data
        )

        df_merged = pd.merge(
            df_merged,
            df_targets,
            on=["drive_id", "phone_name", "UnixTimeMillis"],
            how="inner",  # Inner join because we can only train on rows where we have valid GT and WLS
        )

    # 4. Final Cleanup
    # Fill missing feature values with 0 (standard for missing sensor data in this context)
    # This ensures the model receives a complete tensor
    feature_cols = Config.INPUT_FEATURES
    df_merged[feature_cols] = df_merged[feature_cols].fillna(0)

    print(f"Data loaded. Shape: {df_merged.shape}")
    return df_merged
