import pandas as pd
import numpy as np
import os
import gc
from library.config import INPUT_DIR, FEATURE_NAMES, WORKING_DIR
from library.utils import load_sensor_geometry


def process_batch(batch_df, geometry_df):
    """
    Process a single batch of events to extract features.

    Args:
        batch_df (pd.DataFrame): Raw pulse data for a batch.
        geometry_df (pd.DataFrame): Sensor geometry data.

    Returns:
        pd.DataFrame: Aggregated features indexed by event_id.
    """
    # Filter out auxiliary pulses (noise)
    batch_df = batch_df[~batch_df["auxiliary"]].copy()

    if batch_df.empty:
        return pd.DataFrame(columns=FEATURE_NAMES)

    # Merge sensor geometry
    # batch_df has sensor_id column, geometry_df has sensor_id index
    batch_df = batch_df.join(geometry_df, on="sensor_id", how="left")

    # Calculate charge-weighted coordinates components
    batch_df["wx"] = batch_df["x"] * batch_df["charge"]
    batch_df["wy"] = batch_df["y"] * batch_df["charge"]
    batch_df["wz"] = batch_df["z"] * batch_df["charge"]

    # Group by event_id (which is the index of batch_df)
    grp = batch_df.groupby(level=0)

    # 1. Aggregations for sums, counts, and std devs
    aggs = grp.agg(
        {
            "charge": ["sum", "count"],
            "wx": "sum",
            "wy": "sum",
            "wz": "sum",
            "x": "std",
            "y": "std",
            "z": "std",
        }
    )

    # Rename columns to match internal naming convention before final selection
    aggs.columns = [
        "total_charge",
        "n_pulses",
        "sum_wx",
        "sum_wy",
        "sum_wz",
        "spread_x",
        "spread_y",
        "spread_z",
    ]

    # 2. Time quantiles
    # unstack moves the quantiles (0.1, 0.5, 0.9) from rows to columns
    time_stats = grp["time"].quantile([0.1, 0.5, 0.9]).unstack()
    time_stats.columns = ["time_10", "time_50", "time_90"]

    # Combine aggregations
    features = pd.concat([aggs, time_stats], axis=1)

    # Calculate Centroids (Center of Gravity)
    # Handle potential division by zero (unlikely for valid events with charge)
    features["center_x"] = features["sum_wx"] / features["total_charge"]
    features["center_y"] = features["sum_wy"] / features["total_charge"]
    features["center_z"] = features["sum_wz"] / features["total_charge"]

    # Fill NaNs (e.g., standard deviation is NaN if n_pulses < 2)
    features = features.fillna(0)

    # Ensure correct column order and selection
    return features[FEATURE_NAMES]


def generate_features(meta_path, output_path, load_cached_data=True, debug_n_rows=None):
    """
    Main function to generate features for a dataset defined by metadata.

    Args:
        meta_path (str): Path to the metadata parquet file.
        output_path (str): Path to save/load the processed features.
        load_cached_data (bool): Whether to load from cache if available.
        debug_n_rows (int, optional): Limit number of rows for debugging.

    Returns:
        pd.DataFrame: DataFrame containing features and targets (if available).
    """
    # 1. Caching Check
    if load_cached_data and os.path.exists(output_path):
        print(f"Loading cached features from {output_path}...")
        return pd.read_parquet(output_path)

    print(f"Generating features for {meta_path}...")

    # 2. Load Metadata
    meta_df = pd.read_parquet(meta_path)

    # Debugging: Limit rows
    if debug_n_rows is not None:
        meta_df = meta_df.iloc[:debug_n_rows]
        print(f"Debug mode: Processing first {debug_n_rows} rows.")

    # 3. Load Geometry
    geometry_df = load_sensor_geometry()

    # 4. Process Batches
    # Identify unique batch files to process
    unique_files = meta_df["file_path"].unique()

    processed_chunks = []

    for i, rel_path in enumerate(unique_files):
        full_path = os.path.join(INPUT_DIR, rel_path)

        if not os.path.exists(full_path):
            print(f"Warning: File {full_path} not found. Skipping.")
            continue

        # Load batch
        batch_df = pd.read_parquet(full_path)

        # Get subset of metadata corresponding to this file
        batch_meta = meta_df[meta_df["file_path"] == rel_path]

        # Filter batch data to only include relevant events before processing
        # This ensures we don't process the whole file when only a subset is needed (Cite debug_lesson_3)
        batch_df = batch_df[batch_df.index.isin(batch_meta["event_id"])]

        # Extract features for all events in this batch
        batch_features = process_batch(batch_df, geometry_df)

        # Filter and Merge with Metadata
        # We only want events that are in our current metadata split (train/val/test)
        # And we want to attach targets (azimuth, zenith) if they exist

        # Determine columns to merge
        # Always merge on event_id. Targets are included if present.
        merge_cols = ["event_id"]
        if "azimuth" in batch_meta.columns and "zenith" in batch_meta.columns:
            merge_cols.extend(["azimuth", "zenith"])

        # Merge: Inner join acts as a filter for event_ids belonging to this split
        # batch_features is indexed by event_id
        merged_chunk = batch_features.merge(
            batch_meta[merge_cols], left_index=True, right_on="event_id", how="inner"
        )

        processed_chunks.append(merged_chunk)

        # Garbage collection to keep memory usage stable
        del batch_df, batch_features, merged_chunk
        if i % 10 == 0:
            gc.collect()

    # 5. Concatenate all chunks
    if not processed_chunks:
        # Should not happen unless input is empty or files are missing
        print("Warning: No data processed.")
        return pd.DataFrame()

    full_df = pd.concat(processed_chunks, ignore_index=True)

    # 6. Save to Cache
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    full_df.to_parquet(output_path, index=False)
    print(f"Features saved to {output_path}")

    return full_df
