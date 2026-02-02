import os
import gc
import numpy as np
import pandas as pd
import library.config as config


def process_helmets(helmets_df):
    """
    Applies Max-Pooling Selection Strategy to helmet data.
    Maps frames to steps and selects the largest box per player/step.
    """
    # Map frame to step
    # Snap is at 5s * 59.94 fps = ~300 frames. Step is 10Hz (approx 6 frames).
    # step = (frame - 300) / 5.994
    helmets_df["step"] = ((helmets_df["frame"] - 300) / 5.994).round().astype(int)

    # Calculate area for max pooling
    helmets_df["area"] = helmets_df["width"] * helmets_df["height"]

    # Sort by area descending to pick largest
    helmets_df = helmets_df.sort_values("area", ascending=False)

    # Drop duplicates keeping first (largest area) per player per step
    cols_to_keep = ["game_play", "step", "nfl_player_id"] + config.VISUAL_FEATURES
    helmets_dedup = helmets_df.drop_duplicates(
        subset=["game_play", "step", "nfl_player_id"], keep="first"
    )

    return helmets_dedup[cols_to_keep]


def process_tracking(tracking_df):
    """
    Generates lags and windowed features for tracking data using an Entity-First approach.
    Creates a wide feature vector for window t-5 to t+5.
    """
    # Ensure sorted for correct shifting
    tracking_df = tracking_df.sort_values(["game_play", "nfl_player_id", "step"])

    # Features to lag
    features = config.KINEMATIC_FEATURES

    grouped = tracking_df.groupby(["game_play", "nfl_player_id"])

    lagged_dfs = []
    # Generate lags from -WINDOW_SIZE to +WINDOW_SIZE
    # lag < 0: past, lag > 0: future
    for lag in range(-config.WINDOW_SIZE, config.WINDOW_SIZE + 1):
        # shift(-lag) brings data from t+lag to t.
        # e.g., lag=-1 (past), shift(1) brings t-1 to t.
        shifted = grouped[features].shift(-lag)
        shifted.columns = [f"{col}_lag_{lag}" for col in features]
        lagged_dfs.append(shifted)

    # Concatenate all lags horizontally
    # We keep the key columns from the original sorted dataframe
    meta_cols = tracking_df[["game_play", "nfl_player_id", "step"]]
    tracking_wide = pd.concat([meta_cols] + lagged_dfs, axis=1)

    return tracking_wide


def prepare_data(
    labels_df, tracking_df, helmets_df, load_cached_data=True, split="train"
):
    """
    Main feature engineering pipeline.
    Merges labels with tracking and helmet data, handles ground imputation,
    and computes relative/stable kinematic features.
    """
    cache_file = os.path.join(config.WORKING_DIR, f"features_{split}.parquet")

    # 1. Caching Mechanism
    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading cached features from {cache_file}...")
        return pd.read_parquet(cache_file)

    print(f"Generating features for {split}...")

    # Filter source data to relevant game_plays to optimize memory
    gps = labels_df["game_play"].unique()
    tracking_df = tracking_df[tracking_df["game_play"].isin(gps)].copy()

    # 2. Process Tracking (Entity-First)
    track_wide = process_tracking(tracking_df)

    # 3. Process Helmets
    if helmets_df is not None:
        helmets_df = helmets_df[helmets_df["game_play"].isin(gps)].copy()
        helmets_proc = process_helmets(helmets_df)
    else:
        # Create empty dummy if no helmets (e.g. for robust testing)
        helmets_proc = pd.DataFrame(
            columns=["game_play", "step", "nfl_player_id"] + config.VISUAL_FEATURES
        )

    # 4. Merge Player 1 Data
    # Ensure ID consistency
    labels_df["nfl_player_id_1"] = pd.to_numeric(
        labels_df["nfl_player_id_1"], errors="coerce"
    )

    # Merge Tracking P1
    df = labels_df.merge(
        track_wide,
        left_on=["game_play", "step", "nfl_player_id_1"],
        right_on=["game_play", "step", "nfl_player_id"],
        how="left",
    )
    df = df.drop(columns=["nfl_player_id"])  # Drop redundant join key

    # Rename P1 tracking columns
    lag_cols = [c for c in track_wide.columns if "_lag_" in c]
    rename_dict_1 = {c: f"{c}_1" for c in lag_cols}
    df = df.rename(columns=rename_dict_1)

    # 5. Merge Player 2 Data (Handling Ground)
    # Create numeric ID for merge; 'G' becomes NaN
    df["p2_numeric"] = pd.to_numeric(df["nfl_player_id_2"], errors="coerce")

    # Merge Tracking P2
    df = df.merge(
        track_wide,
        left_on=["game_play", "step", "p2_numeric"],
        right_on=["game_play", "step", "nfl_player_id"],
        how="left",
        suffixes=("", "_p2"),
    )
    df = df.drop(columns=["nfl_player_id", "p2_numeric"])

    rename_dict_2 = {c: f"{c}_2" for c in lag_cols}
    df = df.rename(columns=rename_dict_2)

    # 6. Hybrid Ground Imputation
    # If nfl_player_id_2 == 'G', impute P2 features based on P1
    is_ground = df["nfl_player_id_2"] == "G"

    for lag in range(-config.WINDOW_SIZE, config.WINDOW_SIZE + 1):
        # Impute Position: Ground is at the same location as Player (Distance = 0)
        df.loc[is_ground, f"x_position_lag_{lag}_2"] = df.loc[
            is_ground, f"x_position_lag_{lag}_1"
        ]
        df.loc[is_ground, f"y_position_lag_{lag}_2"] = df.loc[
            is_ground, f"y_position_lag_{lag}_1"
        ]

        # Impute Dynamics: Ground has 0 velocity/acceleration
        for feat in ["speed", "acceleration", "sa", "direction", "orientation"]:
            df.loc[is_ground, f"{feat}_lag_{lag}_2"] = 0.0

    # 7. Merge Visual Features
    # P1
    df = df.merge(
        helmets_proc,
        left_on=["game_play", "step", "nfl_player_id_1"],
        right_on=["game_play", "step", "nfl_player_id"],
        how="left",
    )
    df = df.drop(columns=["nfl_player_id"])
    df = df.rename(columns={c: f"{c}_1" for c in config.VISUAL_FEATURES})

    # P2 (Ground will result in NaNs here, which is correct for now)
    df["p2_numeric"] = pd.to_numeric(df["nfl_player_id_2"], errors="coerce")
    df = df.merge(
        helmets_proc,
        left_on=["game_play", "step", "p2_numeric"],
        right_on=["game_play", "step", "nfl_player_id"],
        how="left",
    )
    df = df.drop(columns=["nfl_player_id", "p2_numeric"])
    df = df.rename(columns={c: f"{c}_2" for c in config.VISUAL_FEATURES})

    # Fill Visual NaNs with 0 (Missing helmet or Ground)
    vis_cols = [f"{c}_1" for c in config.VISUAL_FEATURES] + [
        f"{c}_2" for c in config.VISUAL_FEATURES
    ]
    df[vis_cols] = df[vis_cols].fillna(0)

    # 8. Compute Final Kinematic Features (Relative & Stabilized)
    def clamp(series, limit):
        return series.clip(limit[0], limit[1])

    feature_cols = []

    for lag in range(-config.WINDOW_SIZE, config.WINDOW_SIZE + 1):
        s1 = f"_lag_{lag}_1"
        s2 = f"_lag_{lag}_2"
        suffix = f"_lag_{lag}"

        # --- Relative Geometry ---
        x1, x2 = df[f"x_position{s1}"], df[f"x_position{s2}"]
        y1, y2 = df[f"y_position{s1}"], df[f"y_position{s2}"]

        dx = x1 - x2
        dy = y1 - y2
        dist = np.sqrt(dx**2 + dy**2)

        # Resolution Enhancement & Clamping
        dist = clamp(dist, config.CLAMP_CONFIG["distance"])
        df[f"log_dist{suffix}"] = np.log1p(dist)
        df[f"dx{suffix}"] = dx
        df[f"dy{suffix}"] = dy

        feature_cols.extend([f"log_dist{suffix}", f"dx{suffix}", f"dy{suffix}"])

        # --- Dynamics (Clamped) ---
        for feat in ["speed", "acceleration", "sa"]:
            limits = config.CLAMP_CONFIG.get(feat, (-50, 50))
            v1 = clamp(df[f"{feat}{s1}"], limits)
            v2 = clamp(df[f"{feat}{s2}"], limits)

            df[f"{feat}{s1}_c"] = v1
            df[f"{feat}{s2}_c"] = v2
            feature_cols.extend([f"{feat}{s1}_c", f"{feat}{s2}_c"])

        # --- Angular Continuity (Sin/Cos) ---
        for feat in ["direction", "orientation"]:
            # Convert degrees to radians and compute sin/cos
            # FillNA with 0 for safety (though imputed earlier)
            rad1 = np.radians(df[f"{feat}{s1}"].fillna(0))
            rad2 = np.radians(df[f"{feat}{s2}"].fillna(0))

            df[f"sin_{feat}{s1}"] = np.sin(rad1)
            df[f"cos_{feat}{s1}"] = np.cos(rad1)
            df[f"sin_{feat}{s2}"] = np.sin(rad2)
            df[f"cos_{feat}{s2}"] = np.cos(rad2)

            feature_cols.extend(
                [
                    f"sin_{feat}{s1}",
                    f"cos_{feat}{s1}",
                    f"sin_{feat}{s2}",
                    f"cos_{feat}{s2}",
                ]
            )

    # Add Visual Features
    feature_cols.extend(vis_cols)

    # 9. Finalize Dataset
    meta_cols = [
        "contact_id",
        "game_play",
        "step",
        "nfl_player_id_1",
        "nfl_player_id_2",
    ]
    if "contact" in df.columns:
        meta_cols.append("contact")

    # Select final columns and fill any remaining NaNs (e.g., missing tracking for P1) with 0
    final_df = df[meta_cols + feature_cols].copy()
    final_df[feature_cols] = final_df[feature_cols].fillna(0)

    # Save to cache
    print(f"Saving features to {cache_file}...")
    final_df.to_parquet(cache_file, index=False)

    # Cleanup
    del df, track_wide
    gc.collect()

    return final_df
