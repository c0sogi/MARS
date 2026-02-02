import os
import numpy as np
import pandas as pd
import gc
from library.config import (
    INPUT_DIR,
    METADATA_DIR,
    WORKING_DIR,
    SEED,
    WINDOW_SIZE,
    KINEMATIC_COLS,
    VISUAL_COLS,
    VISUAL_META_COLS,
    CLAMP_MIN,
    CLAMP_MAX,
    GROUND_SPEED,
    GROUND_ACCEL,
    CATEGORICAL_COLS,
)
from library.utils import seed_everything

# Set seed for reproducibility
seed_everything(SEED)


def preprocess_tracking(tracking_df, game_plays):
    """
    Applies Entity-First processing to tracking data:
    1. Filters by relevant game_plays.
    2. Clamps numerical features for stability.
    3. Generates temporal window features (lags).
    """
    # Filter for relevant plays
    tracking_df = tracking_df[tracking_df["game_play"].isin(game_plays)].copy()

    # Sort for windowing
    tracking_df = tracking_df.sort_values(
        ["game_play", "nfl_player_id", "step"]
    ).reset_index(drop=True)

    # Clamp numerical features for stability
    # Only clamp specific kinematic columns that are derivatives or unbounded
    cols_to_clamp = ["speed", "acceleration", "sa"]
    for col in cols_to_clamp:
        if col in tracking_df.columns:
            tracking_df[col] = tracking_df[col].clip(CLAMP_MIN, CLAMP_MAX)

    # Generate Windowed Features
    # We want features from t-WINDOW_SIZE to t+WINDOW_SIZE
    # Efficient vectorized shifting

    # Base columns to window
    feature_cols = KINEMATIC_COLS

    # Result collection
    result_df = tracking_df[
        ["game_play", "nfl_player_id", "step"] + CATEGORICAL_COLS
    ].copy()

    # Group identifiers for boundary checking
    gp = tracking_df["game_play"].values
    pid = tracking_df["nfl_player_id"].values

    for lag in range(-WINDOW_SIZE, WINDOW_SIZE + 1):
        suffix = f"_lag_{lag}"

        # Shift the data
        # Positive lag means looking into the future (shift negative),
        # Negative lag means looking into the past (shift positive)
        # Wait: lag t-5 means 5 steps back. shift(5).
        # We want to align t with t-5. So at row t, we want value from t-5.
        shift_amount = lag

        shifted_df = tracking_df[feature_cols].shift(shift_amount)

        # Create mask for valid shifts (must stay within same player and play)
        shifted_gp = tracking_df["game_play"].shift(shift_amount)
        shifted_pid = tracking_df["nfl_player_id"].shift(shift_amount)

        mask = (gp == shifted_gp) & (pid == shifted_pid)

        # Apply mask and assign
        for col in feature_cols:
            col_name = f"{col}{suffix}"
            result_df[col_name] = np.where(mask, shifted_df[col], np.nan)

            # Fill NaNs (edges of play) with the current timestep's value (Edge Padding)
            # This is better than 0 for positions, and acceptable for speed/accel
            # to maintain continuity at the start/end of tracking.
            # However, for simplicity and robustness, we can fill with 0 or forward fill.
            # Given the "Entity-First" approach, edge padding is safer for positions.
            # We will fill remaining NaNs with the value at lag 0 (current step) later if needed,
            # or just fill with 0. Let's fill with 0 for now to keep it clean,
            # as the model handles 0s well with normalization.
            # Actually, for position, 0 is bad (middle of field).
            # We will forward/backward fill the raw tracking before shifting?
            # No, let's just fillna with 0 for derivatives and current value for position?
            # Simpler: Leave NaNs here, handle imputation after merge or rely on tree/NN robustness.
            # For NN, we must fill. Forward fill the result_df is tricky.
            # Let's fill with the unshifted value (lag 0) for that row.

            current_vals = tracking_df[col].values
            result_df[col_name] = result_df[col_name].fillna(pd.Series(current_vals))

    return result_df


def preprocess_helmets(helmets_df, game_plays):
    """
    Applies Max-Pooling Selection Strategy to helmet data.
    Selects the single best view (largest area) per player per step.
    """
    helmets_df = helmets_df[helmets_df["game_play"].isin(game_plays)].copy()

    # Calculate box area
    helmets_df["box_area"] = helmets_df["width"] * helmets_df["height"]

    # Columns to keep
    # We need to map frame to step.
    # Note: Helmets are by frame, Tracking is by step.
    # We don't have a direct frame-to-step map in helmets csv.
    # However, the task description says: "Sideline and Endzone video pairs are matched frame for frame".
    # And "step... incrementing by 1 every 0.1 seconds".
    # "The moment of snap occurs 5 seconds into the video." -> 300 frames.
    # So step 0 = frame 300. step 1 = frame 306 (approx 59.94Hz).
    # Formula: frame = 300 + step * 6 (approx).
    # Let's verify with tracking data or provided metadata.
    # The metadata/train.csv has 'step' and 'datetime'.
    # The baseline_helmets.csv has 'frame'.
    # We need to align them.
    # Approximate alignment: step * 6 + 300 approx.
    # However, exact alignment is better.
    # We will assume the provided 'step' in labels aligns with 'step' in tracking.
    # We need to link Helmets (Frame) -> Tracking (Step).
    # Since we don't have a provided mapping file in the inputs for this specific alignment
    # (other than calculating it), we will use the standard approximation:
    # frame = 300 + step * 6.
    # This is standard for this dataset (NFL Contact Detection).

    helmets_df["step"] = ((helmets_df["frame"] - 300) / 6).round().astype(int)

    # Sort by area descending to prioritize larger boxes
    helmets_df = helmets_df.sort_values("box_area", ascending=False)

    # Drop duplicates to keep only the largest box per (game_play, step, nfl_player_id)
    # This implements the Max-Pooling Strategy
    helmets_df = helmets_df.drop_duplicates(
        subset=["game_play", "step", "nfl_player_id"]
    )

    # Select features
    keep_cols = ["game_play", "step", "nfl_player_id"] + VISUAL_COLS + ["box_area"]
    return helmets_df[keep_cols]


def process_data(mode="train", load_cached_data=True):
    """
    Main data processing function with caching.

    Args:
        mode (str): 'train', 'validation', or 'test'.
        load_cached_data (bool): Whether to load from disk if available.

    Returns:
        pd.DataFrame: Processed feature matrix including target and ids.
    """
    cache_path = os.path.join(WORKING_DIR, f"{mode}_features.parquet")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached {mode} data from {cache_path}...")
        return pd.read_parquet(cache_path)

    print(f"Processing {mode} data from scratch...")

    # 1. Load Metadata (Labels/Query)
    meta_file = os.path.join(METADATA_DIR, f"{mode}.csv")
    df_meta = pd.read_csv(meta_file)

    # Extract unique game_plays to filter raw data
    unique_game_plays = df_meta["game_play"].unique()

    # 2. Load and Preprocess Tracking
    # Determine which tracking file to use
    tracking_file = (
        "train_player_tracking.csv"
        if mode in ["train", "validation"]
        else "test_player_tracking.csv"
    )
    df_tracking_raw = pd.read_csv(os.path.join(INPUT_DIR, tracking_file))

    # Preprocess (Entity-First: Windowing before merge)
    df_tracking_proc = preprocess_tracking(df_tracking_raw, unique_game_plays)

    # 3. Load and Preprocess Helmets
    helmet_file = (
        "train_baseline_helmets.csv"
        if mode in ["train", "validation"]
        else "test_baseline_helmets.csv"
    )
    df_helmets_raw = pd.read_csv(os.path.join(INPUT_DIR, helmet_file))

    df_helmets_proc = preprocess_helmets(df_helmets_raw, unique_game_plays)

    # 4. Merge Data
    # We merge P1 and P2 separately

    # --- Merge Player 1 ---
    # Tracking
    df_merged = df_meta.merge(
        df_tracking_proc.add_suffix("_1"),
        left_on=["game_play", "step", "nfl_player_id_1"],
        right_on=["game_play_1", "step_1", "nfl_player_id_1"],
        how="left",
    )
    # Helmets
    df_merged = df_merged.merge(
        df_helmets_proc.add_suffix("_1"),
        left_on=["game_play", "step", "nfl_player_id_1"],
        right_on=["game_play_1", "step_1", "nfl_player_id_1"],
        how="left",
    )

    # --- Merge Player 2 ---
    # Handle 'G' (Ground) in P2 ID.
    # Convert nfl_player_id_2 to numeric for merge, forcing 'G' to NaN
    df_merged["nfl_player_id_2_num"] = pd.to_numeric(
        df_merged["nfl_player_id_2"], errors="coerce"
    )

    # Tracking
    df_merged = df_merged.merge(
        df_tracking_proc.add_suffix("_2"),
        left_on=["game_play", "step", "nfl_player_id_2_num"],
        right_on=["game_play_2", "step_2", "nfl_player_id_2"],
        how="left",
    )
    # Helmets
    df_merged = df_merged.merge(
        df_helmets_proc.add_suffix("_2"),
        left_on=["game_play", "step", "nfl_player_id_2_num"],
        right_on=["game_play_2", "step_2", "nfl_player_id_2"],
        how="left",
    )

    # 5. Feature Engineering & Imputation

    # --- Visual Metadata ---
    # Create 'view_available' flags
    df_merged["view_available_1"] = df_merged["box_area_1"].notna().astype(int)
    df_merged["view_available_2"] = df_merged["box_area_2"].notna().astype(int)

    # Fill missing visual features with 0
    vis_cols_1 = [c + "_1" for c in VISUAL_COLS + ["box_area"]]
    vis_cols_2 = [c + "_2" for c in VISUAL_COLS + ["box_area"]]
    df_merged[vis_cols_1] = df_merged[vis_cols_1].fillna(0)
    df_merged[vis_cols_2] = df_merged[vis_cols_2].fillna(0)

    # --- Ground Imputation (Hybrid Physics) ---
    # Identify Ground rows
    is_ground = df_merged["nfl_player_id_2"] == "G"

    # Identify kinematic columns for P1 and P2
    # We need to match the windowed columns
    # Pattern: {col}_lag_{i}_{suffix}

    # Get list of all kinematic feature columns generated
    kin_cols_base = KINEMATIC_COLS
    kin_cols_p1 = []
    kin_cols_p2 = []

    for lag in range(-WINDOW_SIZE, WINDOW_SIZE + 1):
        suffix = f"_lag_{lag}"
        for col in kin_cols_base:
            kin_cols_p1.append(f"{col}{suffix}_1")
            kin_cols_p2.append(f"{col}{suffix}_2")

    # Apply Imputation
    # P2 Position = P1 Position (Distance = 0)
    # P2 Velocity/Accel/etc = 0

    # Position columns need to be copied from P1
    pos_cols = ["x_position", "y_position"]

    for lag in range(-WINDOW_SIZE, WINDOW_SIZE + 1):
        suffix = f"_lag_{lag}"

        # 1. Copy Position P1 -> P2 for Ground
        for p_col in pos_cols:
            c1 = f"{p_col}{suffix}_1"
            c2 = f"{p_col}{suffix}_2"
            if c1 in df_merged.columns and c2 in df_merged.columns:
                df_merged.loc[is_ground, c2] = df_merged.loc[is_ground, c1]

        # 2. Zero out Motion/Angle P2 for Ground
        # Motion columns: speed, acceleration, sa, distance
        # Angle columns: direction, orientation
        motion_angle_cols = [
            "speed",
            "acceleration",
            "sa",
            "distance",
            "direction",
            "orientation",
        ]

        for m_col in motion_angle_cols:
            c2 = f"{m_col}{suffix}_2"
            if c2 in df_merged.columns:
                df_merged.loc[is_ground, c2] = GROUND_SPEED  # 0.0

    # Fill remaining NaNs in tracking data (e.g. missing tracking for real players)
    # We fill with 0 to be safe for NN
    df_merged[kin_cols_p1] = df_merged[kin_cols_p1].fillna(0)
    df_merged[kin_cols_p2] = df_merged[kin_cols_p2].fillna(0)

    # Fill Categorical NaNs (for Embeddings)
    # We will use a placeholder string 'Missing' or 'Ground'
    cat_cols_p1 = [c + "_1" for c in CATEGORICAL_COLS]
    cat_cols_p2 = [c + "_2" for c in CATEGORICAL_COLS]

    df_merged[cat_cols_p1] = df_merged[cat_cols_p1].fillna("Missing")

    # For P2, if Ground, set to 'Ground'
    for c in CATEGORICAL_COLS:
        c2 = c + "_2"
        df_merged.loc[is_ground, c2] = "Ground"
        df_merged[c2] = df_merged[c2].fillna("Missing")

    # 6. Cleanup and Save
    # Drop intermediate merge columns
    drop_cols = [
        "game_play_1",
        "step_1",
        "game_play_2",
        "step_2",
        "nfl_player_id_2_num",
        "path_endzone",
        "path_sideline",
        "path_all29",  # Drop paths to save space, not needed for training
    ]
    df_merged = df_merged.drop(columns=[c for c in drop_cols if c in df_merged.columns])

    # Save to cache
    print(f"Saving {mode} data to {cache_path}...")
    df_merged.to_parquet(cache_path, index=False)

    return df_merged
