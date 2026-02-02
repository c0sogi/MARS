import os
import gc
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from library.config import Config

# ==========================================
# Helper Functions
# ==========================================


def create_entity_lags(tracking_df):
    """
    Generates temporal features (lags) on the player tracking dataframe.
    Returns a 'wide' dataframe where each row (entity at time t) contains
    features from t-W to t+W.
    """
    # Ensure data is sorted temporally for shifting
    tracking_df = tracking_df.sort_values(
        ["game_play", "nfl_player_id", "step"]
    ).reset_index(drop=True)

    # Base columns to lag
    feature_cols = Config.TRACKING_COLS

    # Group by entity (game_play + player)
    # Note: We don't use apply() as it's slow. We use global shifts and mask boundaries if needed,
    # but groupby().shift() is robust and reasonably fast for 1.2M rows.
    grouper = tracking_df.groupby(["game_play", "nfl_player_id"])

    # Dictionary to collect lagged series
    lagged_data = {}

    # Window range: [-WINDOW_SIZE, ..., 0, ..., +WINDOW_SIZE]
    lags = range(-Config.WINDOW_SIZE, Config.WINDOW_SIZE + 1)

    for lag in lags:
        # shift(lag):
        #   lag > 0 (e.g. 1):  Row t gets data from t-1. (Past)
        #   lag < 0 (e.g. -1): Row t gets data from t+1. (Future)
        #   lag = 0: Original data

        shifted_group = grouper[feature_cols].shift(lag)

        for col in feature_cols:
            # Naming convention: feature_lag{k}
            # lag 0 is suffix "_lag0" for consistency
            col_name = f"{col}_lag{lag}"
            lagged_data[col_name] = shifted_group[col]

    # Concatenate all lagged features with the key columns
    # Keys: game_play, nfl_player_id, step
    keys = tracking_df[["game_play", "nfl_player_id", "step"]]

    # Combine into wide dataframe
    wide_df = pd.concat([keys] + [lagged_data[k] for k in lagged_data], axis=1)

    return wide_df


def impute_ground_physics(df):
    """
    Enforces hybrid ground physics logic for rows where nfl_player_id_2 is 'G'.
    Logic:
      - Ground Position = Player 1 Position (Distance -> 0)
      - Ground Velocity/Accel = 0 (Preserves relative motion)
    """
    # Identify ground rows.
    # Note: After merge, nfl_player_id_2 might be NaN if it was 'G' in labels
    # and we merged on numeric IDs. We check the original string ID column from labels if available,
    # or rely on the fact that tracking merge failed for 'G'.
    # However, the robust way is to check the label column 'nfl_player_id_2'.

    if "nfl_player_id_2" not in df.columns:
        # If the column was lost or renamed, we assume rows with NaN in P2 tracking
        # but valid P1 tracking are Ground (or missing data).
        # But strictly, we should use the label column.
        return df

    # Check for 'G' in the ID column (it might be mixed type or string)
    is_ground = df["nfl_player_id_2"].astype(str) == "G"

    if not is_ground.any():
        return df

    lags = range(-Config.WINDOW_SIZE, Config.WINDOW_SIZE + 1)

    for lag in lags:
        suffix = f"_lag{lag}"

        # 1. Impute Position: P2 = P1
        # This ensures distance is 0, but allows the model to see P1's absolute location
        for axis in ["x_position", "y_position"]:
            p1_col = f"{axis}_1{suffix}"
            p2_col = f"{axis}_2{suffix}"

            if p1_col in df.columns and p2_col in df.columns:
                df.loc[is_ground, p2_col] = df.loc[is_ground, p1_col]

        # 2. Impute Kinematics: P2 = 0
        # Ground doesn't move.
        # Columns: speed, acceleration, sa, direction, orientation
        # We set them to 0.
        for feat in ["speed", "acceleration", "sa", "direction", "orientation"]:
            p2_col = f"{feat}_2{suffix}"
            if p2_col in df.columns:
                df.loc[is_ground, p2_col] = 0.0

    return df


def compute_relative_physics(df):
    """
    Calculates explicit relative physics features for every lag step.
    Includes: Log-Distance, Relative Speed, Relative Acceleration, Closing Speed.
    """
    lags = range(-Config.WINDOW_SIZE, Config.WINDOW_SIZE + 1)

    for lag in lags:
        suffix = f"_lag{lag}"

        # Retrieve P1 and P2 coordinates and kinematics
        x1 = df[f"x_position_1{suffix}"]
        y1 = df[f"y_position_1{suffix}"]
        x2 = df[f"x_position_2{suffix}"]
        y2 = df[f"y_position_2{suffix}"]

        s1 = df[f"speed_1{suffix}"]
        s2 = df[f"speed_2{suffix}"]

        a1 = df[f"acceleration_1{suffix}"]
        a2 = df[f"acceleration_2{suffix}"]

        # 1. Distance & Log-Distance
        dx = x1 - x2
        dy = y1 - y2
        dist = np.sqrt(dx**2 + dy**2)

        # Log1p expands resolution near 0 (contact)
        df[f"log_dist{suffix}"] = np.log1p(dist)

        # 2. Relative Scalar Kinematics
        df[f"rel_speed{suffix}"] = s1 - s2
        df[f"rel_accel{suffix}"] = a1 - a2

        # 3. Closing Speed
        # Project relative velocity vector onto position vector
        # v_rel = v1 - v2
        # r_rel = r1 - r2 (dx, dy)
        # closing_speed = - (v_rel . r_rel) / |r_rel|

        # Convert direction (degrees) to radians for vector components
        # Assuming 0 deg is North (Y), 90 deg is East (X) - standard mapping doesn't strictly matter
        # as long as P1 and P2 are consistent.
        dir1_rad = np.radians(df[f"direction_1{suffix}"].fillna(0))
        dir2_rad = np.radians(df[f"direction_2{suffix}"].fillna(0))

        vx1 = s1 * np.sin(dir1_rad)
        vy1 = s1 * np.cos(dir1_rad)
        vx2 = s2 * np.sin(dir2_rad)
        vy2 = s2 * np.cos(dir2_rad)

        dvx = vx1 - vx2
        dvy = vy1 - vy2

        # Dot product
        dot_prod = dvx * dx + dvy * dy

        # Clamped denominator for stability
        denom = np.maximum(dist, 1e-6)

        # Negative dot product because closing in means distance is decreasing
        df[f"closing_speed{suffix}"] = -(dot_prod / denom)

    return df


def prepare_wide_dataset(metadata_df, tracking_df, is_train=False):
    """
    Orchestrates the creation of the wide dataset:
    1. Create wide tracking features (lags).
    2. Merge onto metadata for P1 and P2.
    3. Impute Ground.
    4. Compute Relative Physics.
    5. Return features and targets.
    """
    # 1. Filter tracking to relevant game_plays to save memory
    relevant_gps = metadata_df["game_play"].unique()
    tracking_subset = tracking_df[tracking_df["game_play"].isin(relevant_gps)].copy()

    # 2. Create Entity Lags
    print("  Creating entity lags...")
    wide_tracking = create_entity_lags(tracking_subset)

    # 3. Merge P1
    print("  Merging Player 1 data...")
    # Ensure ID types match
    metadata_df["nfl_player_id_1"] = pd.to_numeric(
        metadata_df["nfl_player_id_1"], errors="coerce"
    )

    merged = metadata_df.merge(
        wide_tracking,
        left_on=["game_play", "nfl_player_id_1", "step"],
        right_on=["game_play", "nfl_player_id", "step"],
        how="left",
    )

    # Rename columns to _1 suffix
    # wide_tracking columns: game_play, nfl_player_id, step, [feats]
    # We want to rename [feats] to [feat]_1
    # Identify feature columns (those containing '_lag')
    feat_cols = [c for c in wide_tracking.columns if "_lag" in c]
    rename_map_1 = {
        c: f"{c[:-5]}_1{c[-5:]}" for c in feat_cols
    }  # insert _1 before _lag
    # Actually simpler: just append _1 to the feature name part?
    # current: speed_lag0. target: speed_1_lag0.
    # Logic: split by _lag, insert _1.

    rename_dict_1 = {}
    for c in feat_cols:
        parts = c.split("_lag")
        rename_dict_1[c] = f"{parts[0]}_1_lag{parts[1]}"

    merged = merged.rename(columns=rename_dict_1)

    # 4. Merge P2
    print("  Merging Player 2 data...")
    # Handle 'G' in P2 ID
    merged["nfl_player_id_2_num"] = pd.to_numeric(
        merged["nfl_player_id_2"], errors="coerce"
    )

    merged = merged.merge(
        wide_tracking,
        left_on=["game_play", "nfl_player_id_2_num", "step"],
        right_on=["game_play", "nfl_player_id", "step"],
        how="left",
        suffixes=("", "_p2_drop"),
    )

    # Drop duplicate keys from merge
    merged = merged.drop(
        columns=[
            "nfl_player_id",
            "nfl_player_id_2_num",
            "game_play_p2_drop",
            "step_p2_drop",
        ],
        errors="ignore",
    )

    # Rename P2 columns
    rename_dict_2 = {}
    for c in feat_cols:
        parts = c.split("_lag")
        rename_dict_2[c] = f"{parts[0]}_2_lag{parts[1]}"

    merged = merged.rename(columns=rename_dict_2)

    # Clean up memory
    del wide_tracking
    gc.collect()

    # 5. Impute Ground Physics
    print("  Imputing ground physics...")
    merged = impute_ground_physics(merged)

    # 6. Compute Relative Physics
    print("  Computing relative physics...")
    merged = compute_relative_physics(merged)

    # 7. Select Feature Columns
    # We want all lag columns: _1_lag, _2_lag, log_dist_lag, rel_speed_lag, etc.
    # We exclude metadata columns.
    all_cols = merged.columns
    feature_cols = [c for c in all_cols if "_lag" in c]

    # Sort columns to ensure consistent order
    feature_cols = sorted(feature_cols)

    X = merged[feature_cols].copy()

    # Handle NaNs (missing tracking data)
    # Fill with 0 is standard for this wide sparse representation, or mean.
    # Given we have ground imputation, remaining NaNs are likely missing sensor data.
    X = X.fillna(0.0)

    if is_train:
        y = merged["contact"].values
        return X, y
    else:
        # For test/val, we might need IDs to map back
        return X, merged


# ==========================================
# Main Interface Functions
# ==========================================


def generate_train_val_features(load_cached_data=True):
    """
    Generates or loads training and validation features.
    Fits and saves the scaler on the training set.
    """
    if (
        load_cached_data
        and os.path.exists(Config.TRAIN_FEATURES_PATH)
        and os.path.exists(Config.VAL_FEATURES_PATH)
    ):
        print("Loading cached train/val features...")
        df_train = pd.read_parquet(Config.TRAIN_FEATURES_PATH)
        df_val = pd.read_parquet(Config.VAL_FEATURES_PATH)

        # Separate X and y
        y_train = df_train["target"].values
        X_train = df_train.drop(columns=["target"]).values

        y_val = df_val["target"].values
        X_val = df_val.drop(columns=["target"]).values

        return X_train, y_train, X_val, y_val

    print("Generating train/val features from scratch...")

    # Load Metadata
    meta_train = pd.read_csv(os.path.join(Config.METADATA_DIR, "train.csv"))
    meta_val = pd.read_csv(os.path.join(Config.METADATA_DIR, "validation.csv"))

    # Load Tracking
    tracking = pd.read_csv(os.path.join(Config.INPUT_DIR, "train_player_tracking.csv"))

    # Process Train
    print("Processing Training Split...")
    X_train_df, y_train = prepare_wide_dataset(meta_train, tracking, is_train=True)

    # Fit Scaler
    print("Fitting Scaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_df)

    # Save Scaler
    joblib.dump(scaler, Config.SCALER_PATH)

    # Save Train Cache (Convert to DataFrame to save with parquet)
    # We add target back for saving
    cache_train = pd.DataFrame(X_train_scaled, columns=X_train_df.columns)
    cache_train["target"] = y_train
    cache_train.to_parquet(Config.TRAIN_FEATURES_PATH)

    # Clean up
    del X_train_df, meta_train, cache_train
    gc.collect()

    # Process Val
    print("Processing Validation Split...")
    X_val_df, y_val = prepare_wide_dataset(meta_val, tracking, is_train=True)

    # Transform Val
    X_val_scaled = scaler.transform(X_val_df)

    # Save Val Cache
    cache_val = pd.DataFrame(X_val_scaled, columns=X_val_df.columns)
    cache_val["target"] = y_val
    cache_val.to_parquet(Config.VAL_FEATURES_PATH)

    del X_val_df, meta_val, tracking, cache_val
    gc.collect()

    return X_train_scaled, y_train, X_val_scaled, y_val


def generate_test_features(load_cached_data=True):
    """
    Generates or loads test features.
    Uses the scaler saved during training.
    """
    if load_cached_data and os.path.exists(Config.TEST_FEATURES_PATH):
        print("Loading cached test features...")
        df_test = pd.read_parquet(Config.TEST_FEATURES_PATH)

        # Extract metadata columns needed for submission
        meta_cols = ["contact_id"]
        test_ids = df_test[meta_cols]
        X_test = df_test.drop(columns=meta_cols).values

        return X_test, test_ids

    print("Generating test features from scratch...")

    # Load Metadata
    meta_test = pd.read_csv(os.path.join(Config.METADATA_DIR, "test.csv"))

    # Load Tracking
    tracking = pd.read_csv(os.path.join(Config.INPUT_DIR, "test_player_tracking.csv"))

    # Process Test
    # Note: is_train=False returns (X_df, merged_df)
    X_test_df, merged_test = prepare_wide_dataset(meta_test, tracking, is_train=False)

    # Load Scaler
    if not os.path.exists(Config.SCALER_PATH):
        raise FileNotFoundError("Scaler not found! Run training first.")
    scaler = joblib.load(Config.SCALER_PATH)

    # Transform
    X_test_scaled = scaler.transform(X_test_df)

    # Save Cache
    # We include contact_id for reconstruction
    cache_test = pd.DataFrame(X_test_scaled, columns=X_test_df.columns)
    cache_test["contact_id"] = merged_test["contact_id"].values
    cache_test.to_parquet(Config.TEST_FEATURES_PATH)

    return X_test_scaled, cache_test[["contact_id"]]
