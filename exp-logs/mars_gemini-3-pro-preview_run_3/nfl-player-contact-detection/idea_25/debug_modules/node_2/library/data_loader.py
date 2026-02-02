import pandas as pd
import numpy as np
import os
from library.config import (
    TRAIN_META_PATH,
    VAL_META_PATH,
    TEST_META_PATH,
    TRAIN_TRACKING_PATH,
    TEST_TRACKING_PATH,
    TRAIN_HELMETS_PATH,
    TEST_HELMETS_PATH,
    WORKING_DIR,
)
from library.utils import CacheManager

# Initialize Cache Manager
cache_manager = CacheManager(WORKING_DIR)


def load_metadata(mode="train", limit=None):
    """
    Loads the metadata (labels and video paths) for the specified mode.

    Args:
        mode (str): One of 'train', 'validation', 'test'.
        limit (int, optional): If provided, limits the number of rows for debugging.

    Returns:
        pd.DataFrame: The metadata dataframe.
    """
    if mode == "train":
        path = TRAIN_META_PATH
    elif mode == "validation":
        path = VAL_META_PATH
    elif mode == "test":
        path = TEST_META_PATH
    else:
        raise ValueError(f"Invalid mode: {mode}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found: {path}")

    # Load data
    df = pd.read_csv(path)

    # Apply limit if requested
    if limit is not None:
        df = df.head(limit).copy()

    return df


def load_tracking(mode="train", load_cached_data=True, game_plays=None):
    """
    Loads player tracking data. Handles mapping validation mode to train file.

    Args:
        mode (str): 'train', 'validation', or 'test'.
        load_cached_data (bool): Whether to use cached parquet files.
        game_plays (list/array, optional): If provided, filters data to these game_plays.

    Returns:
        pd.DataFrame: Tracking data.
    """
    # Determine source file based on mode
    if mode in ["train", "validation"]:
        source_path = TRAIN_TRACKING_PATH
        cache_key_prefix = "tracking_train"
    elif mode == "test":
        source_path = TEST_TRACKING_PATH
        cache_key_prefix = "tracking_test"
    else:
        raise ValueError(f"Invalid mode: {mode}")

    # Define cache filename
    # We cache the full file load to avoid re-parsing CSVs, filtering happens after
    cache_filename = cache_manager.get_hashed_filename(
        cache_key_prefix, {"source": source_path}, "parquet"
    )

    df_tracking = None

    # 1. Try Load from Cache
    if load_cached_data:
        df_tracking = cache_manager.load(cache_filename)

    # 2. Load from Source if needed
    if df_tracking is None:
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Tracking file not found: {source_path}")

        # Specify dtypes to save memory/time
        # datetime is loaded as object and can be converted later if needed
        df_tracking = pd.read_csv(source_path)

        # Save to cache
        cache_manager.save(cache_filename, df_tracking)

    # 3. Filter by game_plays if provided
    if game_plays is not None:
        # Ensure game_plays is a set for O(1) lookup
        gp_set = set(game_plays)
        df_tracking = df_tracking[df_tracking["game_play"].isin(gp_set)].copy()

    return df_tracking


def load_helmets(mode="train", load_cached_data=True, game_plays=None):
    """
    Loads baseline helmet data. Handles mapping validation mode to train file.

    Args:
        mode (str): 'train', 'validation', or 'test'.
        load_cached_data (bool): Whether to use cached parquet files.
        game_plays (list/array, optional): If provided, filters data to these game_plays.

    Returns:
        pd.DataFrame: Helmet data.
    """
    # Determine source file based on mode
    if mode in ["train", "validation"]:
        source_path = TRAIN_HELMETS_PATH
        cache_key_prefix = "helmets_train"
    elif mode == "test":
        source_path = TEST_HELMETS_PATH
        cache_key_prefix = "helmets_test"
    else:
        raise ValueError(f"Invalid mode: {mode}")

    cache_filename = cache_manager.get_hashed_filename(
        cache_key_prefix, {"source": source_path}, "parquet"
    )

    df_helmets = None

    # 1. Try Load from Cache
    if load_cached_data:
        df_helmets = cache_manager.load(cache_filename)

    # 2. Load from Source if needed
    if df_helmets is None:
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Helmets file not found: {source_path}")

        df_helmets = pd.read_csv(source_path)
        cache_manager.save(cache_filename, df_helmets)

    # 3. Filter by game_plays if provided
    if game_plays is not None:
        gp_set = set(game_plays)
        df_helmets = df_helmets[df_helmets["game_play"].isin(gp_set)].copy()

    return df_helmets


def merge_tracking_data(df_labels, df_tracking):
    """
    Merges tracking data onto the labels dataframe for both players involved.

    Args:
        df_labels (pd.DataFrame): Contains contact_id, game_play, step, nfl_player_id_1, nfl_player_id_2.
        df_tracking (pd.DataFrame): Tracking data.

    Returns:
        pd.DataFrame: Merged dataframe with suffixes _p1 and _p2.
    """
    # Prepare Labels for merging
    # Ensure IDs are numeric for merging.
    # nfl_player_id_2 can be 'G', which becomes NaN when coerced to numeric.
    # This is desired: Ground has no tracking data.

    df_merge = df_labels.copy()

    # Convert IDs to numeric (handling 'G' -> NaN)
    # We use temporary columns for merging to preserve original IDs if needed,
    # but usually replacing them is fine. Here we keep originals and make join keys.
    df_merge["join_id_1"] = pd.to_numeric(df_merge["nfl_player_id_1"], errors="coerce")
    df_merge["join_id_2"] = pd.to_numeric(df_merge["nfl_player_id_2"], errors="coerce")

    # Prepare Tracking
    # Ensure tracking ID is numeric
    # Note: We don't modify df_tracking in place
    track_cols = [
        c
        for c in df_tracking.columns
        if c not in ["game_play", "step", "nfl_player_id"]
    ]
    # We need the keys + features
    df_track_clean = df_tracking[
        ["game_play", "step", "nfl_player_id"] + track_cols
    ].copy()
    df_track_clean["nfl_player_id"] = pd.to_numeric(
        df_track_clean["nfl_player_id"], errors="coerce"
    )

    # --- Merge Player 1 ---
    # Rename tracking columns for P1
    df_track_p1 = df_track_clean.add_suffix("_p1")
    # Keys: game_play_p1, step_p1, nfl_player_id_p1
    # Map back to standard keys for the join
    df_track_p1 = df_track_p1.rename(
        columns={
            "game_play_p1": "game_play",
            "step_p1": "step",
            "nfl_player_id_p1": "join_id_1",
        }
    )

    df_merge = pd.merge(
        df_merge, df_track_p1, on=["game_play", "step", "join_id_1"], how="left"
    )

    # --- Merge Player 2 ---
    # Rename tracking columns for P2
    df_track_p2 = df_track_clean.add_suffix("_p2")
    df_track_p2 = df_track_p2.rename(
        columns={
            "game_play_p2": "game_play",
            "step_p2": "step",
            "nfl_player_id_p2": "join_id_2",
        }
    )

    df_merge = pd.merge(
        df_merge, df_track_p2, on=["game_play", "step", "join_id_2"], how="left"
    )

    # Clean up temporary join keys
    df_merge.drop(columns=["join_id_1", "join_id_2"], inplace=True)

    return df_merge
