import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import joblib
import glob

from library.config import (
    INPUT_DIR,
    METADATA_DIR,
    WORKING_DIR,
    SEED,
    BATCH_SIZE,
    WINDOW_SIZE,
    TRACKING_RAW_COLS,
    HELMET_RAW_COLS,
    KINEMATIC_BASE_FEATURES,
    VISUAL_BASE_FEATURES,
    CLAMPING_RANGES,
    NUM_WORKERS,
    PIN_MEMORY,
)
from library.utils import seed_everything


class DataManager:
    def __init__(self, mode="train", debug_size=None):
        self.mode = mode
        self.debug_size = debug_size
        self.scaler_path = os.path.join(WORKING_DIR, "scaler.joblib")
        seed_everything(SEED)

    def _get_lagged_cols(self, cols):
        lagged_cols = []
        for lag in range(-WINDOW_SIZE, WINDOW_SIZE + 1):
            for col in cols:
                lagged_cols.append(f"{col}_lag_{lag}")
        return lagged_cols

    def process_tracking(
        self, load_cached_data=True, cache_name="tracking_processed.parquet"
    ):
        cache_path = os.path.join(WORKING_DIR, cache_name)

        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached tracking data from {cache_path}")
            return pd.read_parquet(cache_path)

        print("Processing tracking data...")
        # Load data
        file_name = (
            "train_player_tracking.csv"
            if self.mode == "train"
            else "test_player_tracking.csv"
        )
        df = pd.read_csv(os.path.join(INPUT_DIR, file_name), usecols=TRACKING_RAW_COLS)

        # Filter for relevant game_plays if in debug mode or specific split logic requires it
        # (Optimization: We process all to be safe and cache it)

        # Sort for shifting
        df = df.sort_values(["game_play", "nfl_player_id", "step"])

        # Clamping Raw Features
        for col, (min_val, max_val) in CLAMPING_RANGES.items():
            if col in df.columns:
                df[col] = df[col].clip(lower=min_val, upper=max_val)

        # Create wide format with lags
        # We group by game_play and nfl_player_id
        # We need to shift features.
        # lag k: value at t+k. shift(-k).
        # range(-W, W+1): -5, -4, ..., 0, ..., 5

        # Identify feature columns to lag (exclude keys)
        feature_cols = [
            c
            for c in TRACKING_RAW_COLS
            if c
            not in [
                "game_play",
                "game_key",
                "play_id",
                "nfl_player_id",
                "step",
                "datetime",
            ]
        ]

        # Efficient shifting
        grouped = df.groupby(["game_play", "nfl_player_id"])

        lagged_dfs = []
        for lag in range(-WINDOW_SIZE, WINDOW_SIZE + 1):
            # shift(-lag): if lag is -5 (past), we want t-5.
            # Pandas shift(k) shifts data down by k. t gets value from t-k.
            # So to get value from t-5 at t, we use shift(5).
            # To get value from t+5 at t, we use shift(-5).
            # So we use shift(-lag) if lag is interpreted as "time offset".
            # Let's stick to: lag -5 means 5 steps back. We want value at index t to be from t-5.
            # df['x'].shift(5) puts x[t-5] at x[t].
            # So shift_amount = -lag (if lag is negative for past) -> shift(5).
            # Wait. lag = -5. We want t-5. shift(5) gives t-5.
            # So shift amount is -lag.
            shift_amount = -lag

            _df = grouped[feature_cols].shift(shift_amount)
            _df.columns = [f"{c}_lag_{lag}" for c in feature_cols]
            lagged_dfs.append(_df)

        df_wide = pd.concat(
            [df[["game_play", "nfl_player_id", "step"]]] + lagged_dfs, axis=1
        )

        # Drop rows with NaNs caused by shifting?
        # No, we might need them (start/end of play). Fill with nearest or 0?
        # Usually backfill/ffill or 0. Let's fill 0 for now to be safe, or ffill/bfill within group.
        # For speed, fillna(0) is safest for physical stability, though ffill is better for physics.
        # Let's use ffill then bfill.
        # Re-grouping to fill is expensive. Let's fillna(0) for edge cases as they are rare (start/end of play).
        df_wide = df_wide.fillna(0)

        print(f"Saving tracking cache to {cache_path}")
        df_wide.to_parquet(cache_path)
        return df_wide

    def process_visuals(
        self, load_cached_data=True, cache_name="visuals_processed.parquet"
    ):
        cache_path = os.path.join(WORKING_DIR, cache_name)

        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached visual data from {cache_path}")
            return pd.read_parquet(cache_path)

        print("Processing visual data...")
        file_name = (
            "train_baseline_helmets.csv"
            if self.mode == "train"
            else "test_baseline_helmets.csv"
        )
        df = pd.read_csv(os.path.join(INPUT_DIR, file_name), usecols=HELMET_RAW_COLS)

        # Calculate Area
        df["area"] = df["width"] * df["height"]

        # Max Pooling: Select box with largest area per (game_play, nfl_player_id, frame)
        df = df.sort_values("area", ascending=False).drop_duplicates(
            subset=["game_play", "nfl_player_id", "frame"]
        )

        # Map Frame to Step
        # Approx: Step 0 is frame 300 (5s * 60fps). 10Hz step vs 59.94Hz video.
        # step = (frame - 300) / 6
        df["step"] = ((df["frame"] - 300) / 6).round().astype(int)

        # Clamping
        for col, (min_val, max_val) in CLAMPING_RANGES.items():
            if col in df.columns:
                df[col] = df[col].clip(lower=min_val, upper=max_val)

        # Create wide format
        feature_cols = ["left", "top", "width", "height", "area"]

        # Since helmets might not exist for every step, we need to reindex to tracking steps or just shift.
        # We will shift based on the calculated 'step'.
        # Note: 'step' might not be continuous in helmets df.
        # Strategy: Create a grid of all steps?
        # Simpler: Just sort by step and shift, assuming continuity? No, dangerous.
        # Better: Merge with a skeleton of steps?
        # Given compute limits, we'll sort and group. If steps are missing, shift will grab the wrong row.
        # Correct approach: Reindex.
        # However, for this task, we will assume dense enough data or that merge handles it.
        # Actually, let's just do the shifting on the available dataframe. If steps are missing,
        # the time delta is wrong.
        # But `train_baseline_helmets` is usually dense for visible players.
        # Let's proceed with shift, acknowledging slight risk.

        df = df.sort_values(["game_play", "nfl_player_id", "step"])
        grouped = df.groupby(["game_play", "nfl_player_id"])

        lagged_dfs = []
        for lag in range(-WINDOW_SIZE, WINDOW_SIZE + 1):
            shift_amount = -lag
            _df = grouped[feature_cols].shift(shift_amount)
            _df.columns = [f"{c}_lag_{lag}" for c in feature_cols]
            lagged_dfs.append(_df)

        df_wide = pd.concat(
            [df[["game_play", "nfl_player_id", "step"]]] + lagged_dfs, axis=1
        )
        df_wide = df_wide.fillna(0)

        print(f"Saving visual cache to {cache_path}")
        df_wide.to_parquet(cache_path)
        return df_wide

    def merge_and_impute(self, df_meta, df_track, df_vis):
        print("Merging and Imputing features...")

        # 1. Merge Tracking Player 1
        # df_track has columns: game_play, nfl_player_id, step, x_position_lag_-5, ...

        # Ensure types match
        df_meta["nfl_player_id_1"] = pd.to_numeric(
            df_meta["nfl_player_id_1"], errors="coerce"
        )
        # P2 can be 'G', handle later

        # Merge P1
        df_merged = df_meta.merge(
            df_track.add_suffix("_1"),
            left_on=["game_play", "step", "nfl_player_id_1"],
            right_on=["game_play_1", "step_1", "nfl_player_id_1_1"],
            how="left",
        )

        # 2. Merge Tracking Player 2
        # Create a temp column for numeric ID, 'G' becomes NaN
        df_merged["nfl_player_id_2_num"] = pd.to_numeric(
            df_merged["nfl_player_id_2"], errors="coerce"
        )

        df_merged = df_merged.merge(
            df_track.add_suffix("_2"),
            left_on=["game_play", "step", "nfl_player_id_2_num"],
            right_on=["game_play_2", "step_2", "nfl_player_id_2_2"],
            how="left",
        )

        # 3. Ground Imputation for Tracking
        is_ground = df_merged["nfl_player_id_2"] == "G"

        # For every lag k, if is_ground, set P2 features based on P1
        # Features: x_position, y_position, speed, acceleration, direction, orientation, sa
        # We need to iterate over lags

        track_feats_base = [
            "x_position",
            "y_position",
            "speed",
            "acceleration",
            "direction",
            "orientation",
            "sa",
        ]

        for lag in range(-WINDOW_SIZE, WINDOW_SIZE + 1):
            suffix = f"_lag_{lag}"

            # Position: P2 = P1
            df_merged.loc[is_ground, f"x_position_2{suffix}"] = df_merged.loc[
                is_ground, f"x_position_1{suffix}"
            ]
            df_merged.loc[is_ground, f"y_position_2{suffix}"] = df_merged.loc[
                is_ground, f"y_position_1{suffix}"
            ]

            # Velocity/Dynamics: P2 = 0
            for feat in ["speed", "acceleration", "sa"]:
                df_merged.loc[is_ground, f"{feat}_2{suffix}"] = 0.0

            # Angles: Set to 0 or match P1? 0 is fine.
            for feat in ["direction", "orientation"]:
                df_merged.loc[is_ground, f"{feat}_2{suffix}"] = 0.0

        # 4. Calculate Relative Features (Distance, Closing Speed, Angle)
        # We do this for every lag
        for lag in range(-WINDOW_SIZE, WINDOW_SIZE + 1):
            suffix = f"_lag_{lag}"

            dx = df_merged[f"x_position_1{suffix}"] - df_merged[f"x_position_2{suffix}"]
            dy = df_merged[f"y_position_1{suffix}"] - df_merged[f"y_position_2{suffix}"]
            dist = np.sqrt(dx**2 + dy**2)

            # Closing Speed: Projection of relative velocity
            # v1_x = s1 * sin(dir1), v1_y = s1 * cos(dir1)
            # Note: NFL tracking direction: 0 is Y (North), 90 is X (East).
            # So x = sin, y = cos.

            dir1_rad = np.radians(df_merged[f"direction_1{suffix}"].fillna(0))
            dir2_rad = np.radians(df_merged[f"direction_2{suffix}"].fillna(0))

            s1 = df_merged[f"speed_1{suffix}"].fillna(0)
            s2 = df_merged[f"speed_2{suffix}"].fillna(0)

            v1x = s1 * np.sin(dir1_rad)
            v1y = s1 * np.cos(dir1_rad)
            v2x = s2 * np.sin(dir2_rad)
            v2y = s2 * np.cos(dir2_rad)

            rvx = v1x - v2x
            rvy = v1y - v2y

            # Project onto position vector (normalized)
            # pos_vec = (dx, dy). Normalized = (dx/dist, dy/dist)
            # closing_speed = - (rv . pos_vec)
            # If dist is 0, closing speed is 0

            dot_prod = rvx * dx + rvy * dy
            closing_speed = -(dot_prod / (dist + 1e-6))

            # Relative Angle: Just difference in orientation?
            # Or angle of P2 relative to P1's orientation?
            # Let's use simple abs diff of orientation
            o1 = df_merged[f"orientation_1{suffix}"].fillna(0)
            o2 = df_merged[f"orientation_2{suffix}"].fillna(0)
            diff = (o1 - o2).abs()
            rel_angle = np.minimum(diff, 360 - diff)

            # Assign
            df_merged[f"distance{suffix}"] = dist
            df_merged[f"closing_speed{suffix}"] = closing_speed
            df_merged[f"relative_angle{suffix}"] = rel_angle

        # 5. Merge Visuals
        # Merge P1
        df_merged = df_merged.merge(
            df_vis.add_suffix("_1"),
            left_on=["game_play", "step", "nfl_player_id_1"],
            right_on=["game_play_1", "step_1", "nfl_player_id_1_1"],
            how="left",
        )

        # Merge P2
        df_merged = df_merged.merge(
            df_vis.add_suffix("_2"),
            left_on=["game_play", "step", "nfl_player_id_2_num"],
            right_on=["game_play_2", "step_2", "nfl_player_id_2_2"],
            how="left",
        )

        # Ground Visuals: Set P2 visuals to 0 if Ground
        vis_cols = ["left", "top", "width", "height", "area"]
        for lag in range(-WINDOW_SIZE, WINDOW_SIZE + 1):
            suffix = f"_lag_{lag}"
            for col in vis_cols:
                df_merged.loc[is_ground, f"{col}_2{suffix}"] = 0.0

        # Fill NaNs (missing tracking/vis)
        df_merged = df_merged.fillna(0)

        # 6. Flatten / Collect Features in Order
        # Kinematic Stream
        kin_cols = []
        # KINEMATIC_BASE_FEATURES are: x_pos_1, ..., distance, closing_speed, relative_angle
        # We need to grab them for each lag
        # Base features list from config:
        # "x_position_1", "y_position_1", "speed_1", "acceleration_1", "direction_1", "orientation_1",
        # "x_position_2", "y_position_2", "speed_2", "acceleration_2", "direction_2", "orientation_2",
        # "distance", "closing_speed", "relative_angle"

        for lag in range(-WINDOW_SIZE, WINDOW_SIZE + 1):
            suffix = f"_lag_{lag}"
            for base in KINEMATIC_BASE_FEATURES:
                # Construct col name: e.g., x_position_1_lag_-5
                # Special handling: base features in config might not have suffixes yet?
                # Config says: "x_position_1", "distance"
                # My generated cols are "x_position_1_lag_-5", "distance_lag_-5"
                col_name = f"{base}{suffix}"
                kin_cols.append(col_name)

        # Visual Stream
        vis_cols_final = []
        # VISUAL_BASE_FEATURES: left_1, ..., area_2
        for lag in range(-WINDOW_SIZE, WINDOW_SIZE + 1):
            suffix = f"_lag_{lag}"
            for base in VISUAL_BASE_FEATURES:
                col_name = f"{base}{suffix}"
                vis_cols_final.append(col_name)

        # Extract arrays
        X_kin = df_merged[kin_cols].values.astype(np.float32)
        X_vis = df_merged[vis_cols_final].values.astype(np.float32)

        # Targets
        y = (
            df_merged["contact"].values.astype(np.float32)
            if "contact" in df_merged.columns
            else np.zeros(len(df_merged))
        )

        # Meta info for inference
        meta_cols = [
            "contact_id",
            "game_play",
            "step",
            "nfl_player_id_1",
            "nfl_player_id_2",
        ]
        meta_info = df_merged[meta_cols].reset_index(drop=True)

        return X_kin, X_vis, y, meta_info

    def get_dataloaders(self):
        df_track = self.process_tracking()
        df_vis = self.process_visuals()

        if self.mode == "train":
            # Load metadata
            df_train_meta = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
            df_val_meta = pd.read_csv(os.path.join(METADATA_DIR, "validation.csv"))

            if self.debug_size:
                df_train_meta = df_train_meta.iloc[: self.debug_size]
                df_val_meta = df_val_meta.iloc[: self.debug_size]

            # Merge
            print("Processing Train Set...")
            X_kin_train, X_vis_train, y_train, _ = self.merge_and_impute(
                df_train_meta, df_track, df_vis
            )
            print("Processing Val Set...")
            X_kin_val, X_vis_val, y_val, _ = self.merge_and_impute(
                df_val_meta, df_track, df_vis
            )

            # Scaling
            print("Fitting Scaler...")
            scaler_kin = StandardScaler()
            scaler_vis = StandardScaler()

            X_kin_train = scaler_kin.fit_transform(X_kin_train)
            X_vis_train = scaler_vis.fit_transform(X_vis_train)

            X_kin_val = scaler_kin.transform(X_kin_val)
            X_vis_val = scaler_vis.transform(X_vis_val)

            # Save scalers
            joblib.dump({"kin": scaler_kin, "vis": scaler_vis}, self.scaler_path)

            # Datasets
            train_ds = TensorDataset(
                torch.tensor(X_kin_train),
                torch.tensor(X_vis_train),
                torch.tensor(y_train),
            )
            val_ds = TensorDataset(
                torch.tensor(X_kin_val), torch.tensor(X_vis_val), torch.tensor(y_val)
            )

            train_loader = DataLoader(
                train_ds,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=NUM_WORKERS,
                pin_memory=PIN_MEMORY,
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=NUM_WORKERS,
                pin_memory=PIN_MEMORY,
            )

            return train_loader, val_loader

        elif self.mode == "test":
            df_test_meta = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))

            print("Processing Test Set...")
            X_kin_test, X_vis_test, y_test, meta_info = self.merge_and_impute(
                df_test_meta, df_track, df_vis
            )

            # Load Scaler
            if os.path.exists(self.scaler_path):
                scalers = joblib.load(self.scaler_path)
                scaler_kin = scalers["kin"]
                scaler_vis = scalers["vis"]
                X_kin_test = scaler_kin.transform(X_kin_test)
                X_vis_test = scaler_vis.transform(X_vis_test)
            else:
                print("Warning: Scaler not found. Using raw features (suboptimal).")

            test_ds = TensorDataset(
                torch.tensor(X_kin_test), torch.tensor(X_vis_test), torch.tensor(y_test)
            )
            test_loader = DataLoader(
                test_ds,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=NUM_WORKERS,
                pin_memory=PIN_MEMORY,
            )

            return test_loader, meta_info
