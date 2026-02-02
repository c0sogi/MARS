import pandas as pd
import numpy as np
import os
import sys
import library.config as C
import library.utils as U


def load_and_merge_data(mode="train", load_cached_data=True, debug=False):
    """
    Loads metadata, tracking, and helmet data, merges them, and handles caching.

    Args:
        mode (str): 'train', 'validation', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.
        debug (bool): If True, samples the data for faster debugging.

    Returns:
        tuple: (df_merged, df_helmets)
            df_merged: DataFrame containing labels merged with tracking data for P1 and P2.
            df_helmets: DataFrame containing helmet data filtered for the relevant plays.
    """
    # 1. Determine File Paths based on Mode
    if mode == "train":
        meta_path = C.TRAIN_META_PATH
        track_path = C.TRACKING_PATH_TRAIN
        helm_path = C.HELMETS_PATH_TRAIN
    elif mode == "validation":
        # Validation set uses train source data but validation metadata
        meta_path = C.VAL_META_PATH
        track_path = C.TRACKING_PATH_TRAIN
        helm_path = C.HELMETS_PATH_TRAIN
    elif mode == "test":
        meta_path = C.TEST_META_PATH
        track_path = C.TRACKING_PATH_TEST
        helm_path = C.HELMETS_PATH_TEST
    else:
        raise ValueError(
            f"Invalid mode: {mode}. Must be 'train', 'validation', or 'test'."
        )

    # 2. Setup Caching
    cache_dir = C.WORKING_DIR
    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)

    # Include debug in filename to avoid cache pollution
    debug_suffix = "_debug" if debug else ""
    cache_name_merged = f"merged_data_{mode}{debug_suffix}.parquet"
    cache_name_helmets = f"helmets_{mode}{debug_suffix}.parquet"

    cache_path_merged = os.path.join(cache_dir, cache_name_merged)
    cache_path_helmets = os.path.join(cache_dir, cache_name_helmets)

    # 3. Try Loading from Cache
    if load_cached_data:
        if os.path.exists(cache_path_merged) and os.path.exists(cache_path_helmets):
            print(f"Loading cached data for {mode} (debug={debug})...")
            try:
                df_merged = pd.read_parquet(cache_path_merged)
                df_helmets = pd.read_parquet(cache_path_helmets)
                return df_merged, df_helmets
            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing...")
        else:
            print(f"Cache not found for {mode}. Processing from scratch...")

    # 4. Load Metadata
    print(f"Loading metadata from {meta_path}...")
    df_meta = pd.read_csv(meta_path)

    if debug:
        # Sample metadata
        print("Debug mode: Sampling metadata...")
        unique_plays = df_meta["game_play"].unique()
        sample_plays = np.random.choice(
            unique_plays, size=min(len(unique_plays), 5), replace=False
        )
        df_meta = df_meta[df_meta["game_play"].isin(sample_plays)].copy()

    # Get relevant game_plays to filter large datasets
    relevant_plays = df_meta["game_play"].unique()

    # 5. Load and Filter Tracking Data
    print(f"Loading tracking data from {track_path}...")
    # Tracking data can be large, read and filter
    df_tracking = pd.read_csv(track_path)
    df_tracking = df_tracking[df_tracking["game_play"].isin(relevant_plays)].copy()

    # 6. Load and Filter Helmet Data
    print(f"Loading helmet data from {helm_path}...")
    df_helmets = pd.read_csv(helm_path)
    df_helmets = df_helmets[df_helmets["game_play"].isin(relevant_plays)].copy()

    # 7. Pre-process IDs for Merging
    # Convert IDs to strings to handle 'G' and ensure consistent types
    df_meta["nfl_player_id_1"] = df_meta["nfl_player_id_1"].astype(str)
    df_meta["nfl_player_id_2"] = df_meta["nfl_player_id_2"].astype(str)

    # Ensure tracking ID is string. Some files might have it as int or float.
    # We strip potential decimals if it was read as float (e.g., "12345.0" -> "12345")
    if pd.api.types.is_float_dtype(df_tracking["nfl_player_id"]):
        df_tracking["nfl_player_id"] = (
            df_tracking["nfl_player_id"].fillna(-1).astype(int).astype(str)
        )
    else:
        df_tracking["nfl_player_id"] = df_tracking["nfl_player_id"].astype(str)

    # 8. Merge Tracking Data
    print("Merging tracking data...")

    # Prepare Tracking P1
    # Rename columns to avoid collisions and indicate player
    track_cols = [
        c
        for c in df_tracking.columns
        if c not in ["game_play", "step", "nfl_player_id"]
    ]

    df_track_p1 = df_tracking.copy()
    rename_map_p1 = {c: f"{c}_p1" for c in track_cols}
    rename_map_p1["nfl_player_id"] = "nfl_player_id_1"
    df_track_p1 = df_track_p1.rename(columns=rename_map_p1)

    # Prepare Tracking P2
    df_track_p2 = df_tracking.copy()
    rename_map_p2 = {c: f"{c}_p2" for c in track_cols}
    rename_map_p2["nfl_player_id"] = "nfl_player_id_2"
    df_track_p2 = df_track_p2.rename(columns=rename_map_p2)

    # Merge P1
    # Left join ensures we keep all labels
    df_merged = pd.merge(
        df_meta, df_track_p1, on=["game_play", "step", "nfl_player_id_1"], how="left"
    )

    # Merge P2
    # Note: Rows where nfl_player_id_2 == 'G' will have NaNs for P2 tracking columns
    df_merged = pd.merge(
        df_merged, df_track_p2, on=["game_play", "step", "nfl_player_id_2"], how="left"
    )

    # 9. Memory Optimization
    print("Optimizing memory usage...")
    df_merged = U.reduce_mem_usage(df_merged, verbose=False)
    df_helmets = U.reduce_mem_usage(df_helmets, verbose=False)

    # 10. Save to Cache
    print(f"Saving to cache: {cache_path_merged}...")
    df_merged.to_parquet(cache_path_merged, index=False)
    df_helmets.to_parquet(cache_path_helmets, index=False)

    return df_merged, df_helmets


def split_by_stream(df):
    """
    Splits the merged dataframe into Stream A (Player-Player) and Stream B (Player-Ground).

    Args:
        df (pd.DataFrame): The merged dataframe from load_and_merge_data.

    Returns:
        tuple: (df_stream_a, df_stream_b)
    """
    # Stream A: Interaction between two players
    # Condition: nfl_player_id_2 is NOT 'G'
    df_stream_a = df[df["nfl_player_id_2"] != "G"].copy()

    # Stream B: Impact with ground
    # Condition: nfl_player_id_2 IS 'G'
    df_stream_b = df[df["nfl_player_id_2"] == "G"].copy()

    return df_stream_a, df_stream_b
