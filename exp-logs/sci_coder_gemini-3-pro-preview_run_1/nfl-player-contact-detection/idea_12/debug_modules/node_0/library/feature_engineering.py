import os
import numpy as np
import pandas as pd
import gc
from library.config import (
    WORKING_DIR,
    TRAIN_FEATURES_PATH,
    VAL_FEATURES_PATH,
    TEST_FEATURES_PATH,
    POLAR_GRID_SETTINGS,
    WINDOW_SIZE,
    SEED,
    GATING_DISTANCE,
)
from library.data_utils import (
    load_metadata_and_tracking,
    prepare_test_data,
    apply_geometric_gating,
)


def _compute_polar_kinematics(df):
    """
    Computes ego-centric polar coordinates and relative velocities.
    Transforms P2's state into P1's local coordinate system.
    """
    # 1. Handle Ground (P2='G') by filling NaNs with 0 for calculation safety
    # We will mask these later or rely on 'is_ground' feature
    # Create a mask for valid player-player interactions
    # Note: x_position_p2 is NaN for Ground

    # Fill NaNs temporarily for vectorized math
    df_filled = df.copy()
    fill_cols = ["x_position_p2", "y_position_p2", "speed_p2", "direction_p2"]
    for col in fill_cols:
        if col in df_filled.columns:
            df_filled[col] = df_filled[col].fillna(0)

    # 2. Convert Orientation/Direction to Radians (NFL: 0=Y-axis, 90=X-axis)
    # Math Angle (0=X-axis, CCW): theta_math = (90 - theta_nfl) * pi / 180
    def nfl_to_math_rad(deg):
        return np.deg2rad((90 - deg) % 360)

    # P1 Orientation (Frame of Reference)
    # If orientation is missing, assume direction, else 0
    ori_p1 = df_filled["orientation_p1"].fillna(df_filled["direction_p1"]).fillna(0)
    theta_p1 = nfl_to_math_rad(ori_p1)

    # P1 and P2 Velocities
    dir_p1_rad = nfl_to_math_rad(df_filled["direction_p1"].fillna(0))
    dir_p2_rad = nfl_to_math_rad(df_filled["direction_p2"].fillna(0))

    v1_x = df_filled["speed_p1"] * np.cos(dir_p1_rad)
    v1_y = df_filled["speed_p1"] * np.sin(dir_p1_rad)
    v2_x = df_filled["speed_p2"] * np.cos(dir_p2_rad)
    v2_y = df_filled["speed_p2"] * np.sin(dir_p2_rad)

    # Relative Position (Global)
    dx = df_filled["x_position_p2"] - df_filled["x_position_p1"]
    dy = df_filled["y_position_p2"] - df_filled["y_position_p1"]

    # Relative Velocity (Global)
    dvx = v2_x - v1_x
    dvy = v2_y - v1_y

    # 3. Rotate to P1's Ego-Centric Frame
    # Rotation matrix for -theta_p1 (align P1 orientation to X-axis or Y-axis?)
    # Let's align P1's orientation to the positive X-axis of the local grid.
    # New X = dx * cos(-theta) - dy * sin(-theta)
    # New Y = dx * sin(-theta) + dy * cos(-theta)
    # Note: We use -theta_p1 to rotate the world so P1 faces East (0 rad)

    cos_t = np.cos(-theta_p1)
    sin_t = np.sin(-theta_p1)

    local_x = dx * cos_t - dy * sin_t
    local_y = dx * sin_t + dy * cos_t

    local_vx = dvx * cos_t - dvy * sin_t
    local_vy = dvx * sin_t + dvy * cos_t

    # 4. Polar Coordinates
    polar_r = np.sqrt(local_x**2 + local_y**2)
    polar_theta = np.arctan2(local_y, local_x)  # -pi to pi

    # 5. Fluxes
    # Radial Flux: Projection of relative velocity onto the radius vector
    # Unit vector r: (cos(theta), sin(theta))
    # v_radial = v . r
    radial_flux = local_vx * np.cos(polar_theta) + local_vy * np.sin(polar_theta)

    # Tangential Flux: Projection onto tangent vector
    # Unit vector t: (-sin(theta), cos(theta))
    tangential_flux = -local_vx * np.sin(polar_theta) + local_vy * np.cos(polar_theta)

    # Assign to dataframe
    df["polar_r"] = polar_r
    df["polar_theta"] = polar_theta
    df["radial_flux"] = radial_flux
    df["tangential_flux"] = tangential_flux

    # 6. Polar Grid Binning (Occupancy/Interaction Grid)
    # We create features representing which bin P2 falls into.
    # Since we only have one P2 per row, this is effectively a spatial one-hot encoding
    # weighted by the interaction existence.

    # Sectors: 4 sectors (Front, Left, Back, Right)
    # Front: -45 to 45 deg (-pi/4 to pi/4)
    # Left: 45 to 135
    # Back: 135 to -135 (or 135 to 180 and -180 to -135)
    # Right: -135 to -45

    # Normalize theta to [0, 2pi) for easier binning
    theta_deg = np.degrees(polar_theta) % 360

    # Define sectors (centered at 0, 90, 180, 270)
    # Front (0): 315 to 45 (wrap around)
    # Left (90): 45 to 135
    # Back (180): 135 to 225
    # Right (270): 225 to 315

    sector_map = np.zeros(len(df), dtype=int)
    sector_map[(theta_deg >= 45) & (theta_deg < 135)] = 1  # Left
    sector_map[(theta_deg >= 135) & (theta_deg < 225)] = 2  # Back
    sector_map[(theta_deg >= 225) & (theta_deg < 315)] = 3  # Right
    # Default 0 is Front (315-360 and 0-45)

    # Radial Bands
    # Band 0: 0-1y, Band 1: 1-2y, Band 2: >2y
    band_map = np.zeros(len(df), dtype=int)
    band_map[polar_r >= 1.0] = 1
    band_map[polar_r >= 2.0] = 2

    # Create Grid Features
    # Format: grid_s{sector}_b{band}
    # We only set this to 1 (Occupancy) for the specific bin P2 is in.
    # For Ground (P2='G'), we should set these to 0.
    is_ground = df["nfl_player_id_2"] == "G"

    for s in range(4):
        for b in range(3):
            col_name = f"grid_s{s}_b{b}"
            # Active if sector matches AND band matches AND not ground
            mask = (sector_map == s) & (band_map == b) & (~is_ground)
            df[col_name] = mask.astype(int)

    # Explicitly zero out polar features for Ground rows to avoid noise
    ground_cols = ["polar_r", "polar_theta", "radial_flux", "tangential_flux"]
    df.loc[is_ground, ground_cols] = 0.0

    return df


def _add_temporal_features(df):
    """
    Generates flattened temporal windows using lag/lead shifts.
    Groups by GamePlay and Player Pair to ensure continuity.
    """
    # Define columns to lag
    # We want kinematics and polar features
    feature_cols = [
        "speed_p1",
        "acceleration_p1",
        "speed_p2",
        "acceleration_p2",
        "polar_r",
        "radial_flux",
        "tangential_flux",
    ]

    # Ensure we sort correctly
    # Note: 'nfl_player_id_2' is object (contains 'G').
    # Grouping by object column is fine in pandas.
    df = df.sort_values(by=["game_play", "nfl_player_id_1", "nfl_player_id_2", "step"])

    # Group object
    # We use a composite key for speed or just groupby
    # GroupBy is expensive on 3M rows with string keys.
    # Optimization: Convert game_play to category or int code if possible.
    # Here we rely on standard pandas groupby.
    grouper = df.groupby(["game_play", "nfl_player_id_1", "nfl_player_id_2"])

    # Generate lags
    # Window: +/- WINDOW_SIZE
    # We will create columns like 'speed_p1_lag_-5', 'speed_p1_lag_5'
    # Step size: We can skip steps to reduce dimensionality if needed,
    # but the prompt asks for "Flattened Temporal Windows".
    # We'll take a stride of 2 to keep feature count manageable: -10, -8, ..., 0, ..., 10

    shifts = range(-WINDOW_SIZE, WINDOW_SIZE + 1, 2)

    new_cols = {}

    for col in feature_cols:
        if col not in df.columns:
            continue

        # Extract series
        series = df[col]

        for s in shifts:
            if s == 0:
                continue  # Original column exists

            col_name = f"{col}_lag_{s}"
            # shift(s): positive s shifts down (lag), negative s shifts up (lead)
            # We want t-k. shift(k) gives value from t-k at t.
            new_cols[col_name] = grouper[col].shift(s)

    # Concatenate all new columns at once to avoid fragmentation
    df_lags = pd.DataFrame(new_cols, index=df.index)
    df = pd.concat([df, df_lags], axis=1)

    return df


def _process_dataset(df, is_train=True):
    """
    Main feature engineering pipeline.
    """
    # 1. Basic flags
    df["is_ground"] = (df["nfl_player_id_2"] == "G").astype(int)

    # 2. Polar Kinematics (Instantaneous)
    df = _compute_polar_kinematics(df)

    # 3. Temporal Features (History/Future)
    # Must be done BEFORE gating to preserve history of approaching players
    df = _add_temporal_features(df)

    # 4. Geometric Gating (Train/Val Only)
    if is_train:
        # Save count before
        n_before = len(df)
        df = apply_geometric_gating(df)
        n_after = len(df)
        # print(f"Gating reduced data from {n_before} to {n_after} rows.")

    # 5. Imputation
    # Lags introduce NaNs at the edges of plays.
    # Missing tracking introduces NaNs.
    # We fill with 0.
    df = df.fillna(0)

    # 6. Feature Selection
    # Drop metadata columns not needed for training
    drop_cols = [
        "game_play",
        "game_key",
        "play_id",
        "nfl_player_id_1",
        "nfl_player_id_2",
        "step",
        "datetime",
        "video_path_endzone",
        "video_path_sideline",
        "video_path_all29",
        "p2_join_key",
        "calculated_distance",
    ]
    # Also drop raw tracking positions if not needed (model should use relative/polar)
    pos_cols = ["x_position_p1", "y_position_p1", "x_position_p2", "y_position_p2"]

    # Identify feature columns (numeric)
    feature_cols = [
        c
        for c in df.columns
        if c not in drop_cols
        and c not in pos_cols
        and c != "contact_id"
        and c != "contact"
    ]

    # Ensure float32 for memory efficiency
    for c in feature_cols:
        df[c] = df[c].astype(np.float32)

    return df, feature_cols


def generate_dataset(split="train", load_cached_data=True, debug=False):
    """
    Generates X (features) and y (target) for the specified split.
    Handles caching.
    """
    # Determine paths
    if split == "train":
        cache_path = TRAIN_FEATURES_PATH
    elif split == "val":
        cache_path = VAL_FEATURES_PATH
    elif split == "test":
        cache_path = TEST_FEATURES_PATH
    else:
        raise ValueError(f"Invalid split: {split}")

    # Check cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading {split} features from cache: {cache_path}")
        df_final = pd.read_parquet(cache_path)

        # Separate X, y, meta
        if "contact" in df_final.columns:
            y = df_final["contact"].values
            X = df_final.drop(columns=["contact", "contact_id"])
        else:
            y = None
            X = df_final.drop(columns=["contact_id"])

        # Return contact_ids for submission mapping if test
        ids = df_final["contact_id"].values
        return X, y, ids

    # Compute from scratch
    print(f"Generating {split} features from scratch...")

    # 1. Load Merged Data
    if split == "test":
        df = prepare_test_data(load_cached_data=load_cached_data)
    else:
        df = load_metadata_and_tracking(split, load_cached_data=load_cached_data)

    # Debug sampling
    if debug and split != "test":
        df = df.sample(n=min(len(df), 50000), random_state=SEED).copy()

    # 2. Engineer Features
    # For test set, we do NOT gate (we need predictions for all rows)
    is_train_mode = split != "test"
    df_proc, feature_cols = _process_dataset(df, is_train=is_train_mode)

    # 3. Prepare Output
    # Keep contact_id for mapping
    output_cols = ["contact_id"] + feature_cols
    if "contact" in df_proc.columns:
        output_cols.append("contact")

    df_final = df_proc[output_cols]

    # 4. Save Cache
    print(f"Saving {split} features to cache: {cache_path}")
    df_final.to_parquet(cache_path, index=False)

    # 5. Return
    ids = df_final["contact_id"].values
    if "contact" in df_final.columns:
        y = df_final["contact"].values
        X = df_final.drop(columns=["contact", "contact_id"])
    else:
        y = None
        X = df_final.drop(columns=["contact_id"])

    return X, y, ids
