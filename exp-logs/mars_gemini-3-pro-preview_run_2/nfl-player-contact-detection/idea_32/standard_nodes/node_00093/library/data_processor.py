import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from library.config import Config
from library.utils import seed_everything


class DataProcessor:
    def __init__(self):
        self.scalers = {}
        seed_everything(Config.SEED)

    def load_data(self, mode="train"):
        """
        Loads the raw datasets based on the mode (train/validation/test).
        """
        if mode == "train":
            df_meta = pd.read_csv(Config.TRAIN_META_PATH)
            track_path = Config.TRAIN_TRACKING_PATH
            helmet_path = Config.TRAIN_HELMETS_PATH
        elif mode == "validation":
            df_meta = pd.read_csv(Config.VAL_META_PATH)
            track_path = Config.TRAIN_TRACKING_PATH
            helmet_path = Config.TRAIN_HELMETS_PATH
        elif mode == "test":
            df_meta = pd.read_csv(Config.TEST_META_PATH)
            track_path = Config.TEST_TRACKING_PATH
            helmet_path = Config.TEST_HELMETS_PATH
        else:
            raise ValueError(f"Invalid mode: {mode}")

        df_tracking = pd.read_csv(track_path)
        df_helmets = pd.read_csv(helmet_path)

        # Ensure consistent types
        df_tracking["game_play"] = df_tracking["game_play"].astype(str)
        df_helmets["game_play"] = df_helmets["game_play"].astype(str)
        df_meta["game_play"] = df_meta["game_play"].astype(str)

        return df_meta, df_tracking, df_helmets

    def process_tracking(self, tracking):
        """
        Preprocesses tracking data:
        1. Computes derivatives (Jerk).
        2. Generates windowed features (lags) for entity-level columns.
        """
        # Filter to relevant columns
        cols = [
            "game_play",
            "nfl_player_id",
            "step",
            "x_position",
            "y_position",
            "speed",
            "acceleration",
            "orientation",
            "direction",
            "sa",
        ]
        tracking = tracking[cols].copy()

        # Sort for correct shifting
        tracking.sort_values(["game_play", "nfl_player_id", "step"], inplace=True)

        # Compute Jerk (derivative of acceleration)
        # dt = 0.1s. Jerk = diff(accel) / 0.1
        tracking["jerk"] = (
            tracking.groupby(["game_play", "nfl_player_id"])["acceleration"]
            .diff()
            .fillna(0)
            / 0.1
        )

        # Define base features to window
        base_feats = [
            "x_position",
            "y_position",
            "speed",
            "acceleration",
            "orientation",
            "direction",
            "sa",
            "jerk",
        ]

        # Generate Lags (Entity-First)
        # We use a loop to generate shifted columns.
        # Note: This increases width significantly but avoids N^2 merge complexity later.
        shifts = range(-Config.WINDOW_SIZE, Config.WINDOW_SIZE + 1)

        # Group object for shifting
        grp = tracking.groupby(["game_play", "nfl_player_id"])

        result_dfs = [tracking]

        for lag in shifts:
            if lag == 0:
                continue

            # Shift creates lags. Positive lag = previous step (t-k).
            # Negative lag = future step (t+k).
            # We want t-5 to t+5.
            # shift(1) gives value from t-1 at t.
            shifted = grp[base_feats].shift(lag)

            # Rename columns
            suffix = f"_lag_{lag}"
            shifted.columns = [c + suffix for c in shifted.columns]

            result_dfs.append(shifted)

        # Concatenate all lags
        tracking_processed = pd.concat(result_dfs, axis=1)

        # Fill NaNs created by shifting (edges of plays) with 0
        # For positions, 0 is technically wrong (0,0 is on field), but for relative features later it handles ok
        # or we could ffill. Given the strict physics, 0 for derivatives is safe.
        # For positions, we rely on the fact that contact usually doesn't happen at step 0 or max.
        tracking_processed = tracking_processed.fillna(0)

        return tracking_processed

    def process_helmets(self, helmets):
        """
        Preprocesses helmet data:
        1. Calculates box area.
        2. Applies Max-Pooling (selects best view per player/step).
        3. Maps video frames to simulation steps.
        """
        # Calculate Area
        helmets["area"] = helmets["width"] * helmets["height"]

        # Map Frame to Step
        # Step 0 = 5.0s = Frame 300 (approx). 0.1s = 5.994 frames.
        # step = round((frame - 300) / 5.994)
        helmets["step"] = ((helmets["frame"] - 300) / 5.994).round().astype(int)

        # Max Pooling: Sort by area descending, then drop duplicates on key
        helmets = helmets.sort_values("area", ascending=False)
        helmets = helmets.drop_duplicates(subset=["game_play", "step", "nfl_player_id"])

        # Select columns
        cols = [
            "game_play",
            "step",
            "nfl_player_id",
            "left",
            "top",
            "width",
            "height",
            "area",
        ]
        return helmets[cols]

    def engineer_features(self, df, tracking, helmets):
        """
        Merges data and calculates physics-informed pairwise features.
        """

        # --- Merge Player 1 ---
        # Helper to rename tracking columns
        def rename_track(df_t, p_suffix):
            rename_map = {}
            for c in df_t.columns:
                if c in ["game_play", "step", "nfl_player_id"]:
                    continue
                if "_lag_" in c:
                    # Insert player suffix before lag suffix
                    # e.g. x_position_lag_-5 -> x_position_1_lag_-5
                    base, lag_part = c.split("_lag_", 1)
                    rename_map[c] = f"{base}{p_suffix}_lag_{lag_part}"
                else:
                    rename_map[c] = c + p_suffix
            return df_t.rename(columns=rename_map)

        track_1 = rename_track(tracking, "_1")
        df = df.merge(
            track_1,
            left_on=["game_play", "step", "nfl_player_id_1"],
            right_on=["game_play", "step", "nfl_player_id"],
            how="left",
        )
        df.drop(columns=["nfl_player_id"], inplace=True)

        # --- Merge Player 2 (Handles Ground) ---
        # Convert nfl_player_id_2 to numeric for merge (force 'G' to NaN)
        df["p2_merge_id"] = pd.to_numeric(df["nfl_player_id_2"], errors="coerce")

        track_2 = rename_track(tracking, "_2")
        df = df.merge(
            track_2,
            left_on=["game_play", "step", "p2_merge_id"],
            right_on=["game_play", "step", "nfl_player_id"],
            how="left",
        )
        df.drop(columns=["nfl_player_id", "p2_merge_id"], inplace=True)

        # --- Impute Ground for P2 ---
        is_ground = df["nfl_player_id_2"] == "G"

        # Iterate over P2 columns to impute
        p2_cols = [
            c
            for c in df.columns
            if (c.endswith("_2") or "_2_lag_" in c) and c not in ["nfl_player_id_2"]
        ]

        for col in p2_cols:
            if "x_position" in col or "y_position" in col:
                # Impute Position: Ground is at P1's location (Distance = 0)
                p1_col = col.replace("_2", "_1")
                df.loc[is_ground, col] = df.loc[is_ground, p1_col]
            else:
                # Impute Motion/Dynamics: Ground is static (0)
                df.loc[is_ground, col] = 0.0

        # Fill remaining missing tracking data (e.g. missing sensors) with 0
        df = df.fillna(0)

        # --- Merge Helmets ---
        # P1 Helmets
        h1 = helmets.rename(
            columns={
                c: f"p1_box_{c}" for c in ["left", "top", "width", "height", "area"]
            }
        )
        df = df.merge(
            h1,
            left_on=["game_play", "step", "nfl_player_id_1"],
            right_on=["game_play", "step", "nfl_player_id"],
            how="left",
        )
        df.drop(columns=["nfl_player_id"], inplace=True)

        # P2 Helmets
        df["p2_merge_id"] = pd.to_numeric(df["nfl_player_id_2"], errors="coerce")
        h2 = helmets.rename(
            columns={
                c: f"p2_box_{c}" for c in ["left", "top", "width", "height", "area"]
            }
        )
        df = df.merge(
            h2,
            left_on=["game_play", "step", "p2_merge_id"],
            right_on=["game_play", "step", "nfl_player_id"],
            how="left",
        )
        df.drop(columns=["nfl_player_id", "p2_merge_id"], inplace=True)

        # Fill missing helmets with 0
        df = df.fillna(0)

        # --- Pairwise Feature Calculation (Physics-Informed) ---
        shifts = range(-Config.WINDOW_SIZE, Config.WINDOW_SIZE + 1)

        for lag in shifts:
            suffix = f"_lag_{lag}" if lag != 0 else ""

            # 1. Geometry: Log Distance
            x1 = df[f"x_position_1{suffix}"]
            y1 = df[f"y_position_1{suffix}"]
            x2 = df[f"x_position_2{suffix}"]
            y2 = df[f"y_position_2{suffix}"]

            dx = x1 - x2
            dy = y1 - y2
            dist = np.sqrt(dx**2 + dy**2)
            df[f"log_distance{suffix}"] = np.log1p(dist.clip(0, Config.MAX_DISTANCE))

            # 2. Geometry: Angular Differences (Shortest Arc)
            for ang_type in ["orientation", "direction"]:
                a1 = df[f"{ang_type}_1{suffix}"]
                a2 = df[f"{ang_type}_2{suffix}"]
                # Shortest arc: min(|d|, 360-|d|)
                diff = (a1 - a2 + 180) % 360 - 180
                df[f"{ang_type}_diff{suffix}"] = diff.abs()

            # 3. Motion: Relative Speed
            s1 = df[f"speed_1{suffix}"]
            s2 = df[f"speed_2{suffix}"]
            df[f"relative_speed{suffix}"] = (s1 - s2).abs()

            # 4. Motion: Clamped Closing Speed
            # Convert to radians
            d1_rad = np.radians(df[f"direction_1{suffix}"])
            d2_rad = np.radians(df[f"direction_2{suffix}"])

            vx1, vy1 = s1 * np.sin(d1_rad), s1 * np.cos(d1_rad)
            vx2, vy2 = s2 * np.sin(d2_rad), s2 * np.cos(d2_rad)

            rvx = vx1 - vx2
            rvy = vy1 - vy2

            # Unit vector P1->P2
            mask_nz = dist > 1e-6
            ux = np.zeros_like(dist)
            uy = np.zeros_like(dist)
            ux[mask_nz] = (x2 - x1)[mask_nz] / dist[mask_nz]
            uy[mask_nz] = (y2 - y1)[mask_nz] / dist[mask_nz]

            closing_speed = -(rvx * ux + rvy * uy)
            df[f"clamped_closing_speed{suffix}"] = closing_speed.clip(
                -Config.MAX_SPEED, Config.MAX_SPEED
            )

            # 5. Motion: Time to Collision
            ttc = np.zeros_like(closing_speed)
            mask_closing = closing_speed > 0.1
            ttc[mask_closing] = dist[mask_closing] / closing_speed[mask_closing]
            df[f"time_to_collision{suffix}"] = ttc.clip(0, 10.0)

            # 6. Dynamics: Relative Acceleration
            a1 = df[f"acceleration_1{suffix}"]
            a2 = df[f"acceleration_2{suffix}"]
            df[f"relative_acceleration{suffix}"] = (a1 - a2).abs()

            # 7. Explicit Clamping of Dynamics
            dyn_cols = [
                f"acceleration_1{suffix}",
                f"acceleration_2{suffix}",
                f"jerk_1{suffix}",
                f"jerk_2{suffix}",
                f"sa_1{suffix}",
                f"sa_2{suffix}",
            ]
            for col in dyn_cols:
                limit = Config.MAX_JERK if "jerk" in col else Config.MAX_ACCEL
                df[col] = df[col].clip(-limit, limit)

        return df

    def prepare_tensors(self, df, fit_scaler=False):
        """
        Flattens windowed features into groups and applies scaling.
        """

        def get_window_cols(base_list):
            cols = []
            shifts = range(-Config.WINDOW_SIZE, Config.WINDOW_SIZE + 1)
            for lag in shifts:
                suffix = f"_lag_{lag}" if lag != 0 else ""
                for base in base_list:
                    cols.append(f"{base}{suffix}")
            return cols

        # Define column groups
        cols_A = get_window_cols(Config.FEAT_GROUP_A_GEO)
        cols_B = get_window_cols(Config.FEAT_GROUP_B_MOTION)
        cols_C = get_window_cols(Config.FEAT_GROUP_C_DYNAMICS)
        cols_Vis = Config.FEAT_VISUAL  # Visuals are current-step only

        # Extract
        X_A = df[cols_A].values.astype(np.float32)
        X_B = df[cols_B].values.astype(np.float32)
        X_C = df[cols_C].values.astype(np.float32)
        X_Vis = df[cols_Vis].values.astype(np.float32)

        # Scale
        scaler_path = os.path.join(Config.WORKING_DIR, "scalers.joblib")

        if fit_scaler:
            self.scalers["A"] = StandardScaler().fit(X_A)
            self.scalers["B"] = StandardScaler().fit(X_B)
            self.scalers["C"] = StandardScaler().fit(X_C)
            self.scalers["Vis"] = StandardScaler().fit(X_Vis)
            joblib.dump(self.scalers, scaler_path)
        else:
            if not self.scalers:
                if os.path.exists(scaler_path):
                    self.scalers = joblib.load(scaler_path)
                else:
                    # Fallback if no scaler exists (should not happen in proper flow)
                    self.scalers["A"] = StandardScaler()
                    self.scalers["B"] = StandardScaler()
                    self.scalers["C"] = StandardScaler()
                    self.scalers["Vis"] = StandardScaler()

        # Transform
        if self.scalers:
            X_A = self.scalers["A"].transform(X_A)
            X_B = self.scalers["B"].transform(X_B)
            X_C = self.scalers["C"].transform(X_C)
            X_Vis = self.scalers["Vis"].transform(X_Vis)

        return X_A, X_B, X_C, X_Vis

    def get_data(self, mode="train", load_cached_data=True):
        """
        Main entry point. Checks cache, processes data, and returns tensors.
        """
        cache_file = os.path.join(Config.WORKING_DIR, f"{mode}_features.parquet")

        # 1. Try Load Cache
        if load_cached_data and os.path.exists(cache_file):
            print(f"Loading cached {mode} data from {cache_file}...")
            df = pd.read_parquet(cache_file)

            # Determine if we need to fit scaler (only if training and not done yet)
            fit_scaler = mode == "train" and not os.path.exists(
                os.path.join(Config.WORKING_DIR, "scalers.joblib")
            )

            X_A, X_B, X_C, X_Vis = self.prepare_tensors(df, fit_scaler=fit_scaler)
            y = df["contact"].values if "contact" in df.columns else None
            return (X_A, X_B, X_C, X_Vis), y, df[["contact_id"]]

        # 2. Process from Scratch
        print(f"Processing {mode} data from scratch...")
        meta, tracking, helmets = self.load_data(mode)

        # Optimization: Filter raw data to relevant game_plays
        relevant_games = meta["game_play"].unique()
        tracking = tracking[tracking["game_play"].isin(relevant_games)].copy()
        helmets = helmets[helmets["game_play"].isin(relevant_games)].copy()

        # Pipeline
        track_proc = self.process_tracking(tracking)
        helm_proc = self.process_helmets(helmets)
        df_proc = self.engineer_features(meta, track_proc, helm_proc)

        # Save Cache
        df_proc.to_parquet(cache_file)

        # Tensor Prep
        fit_scaler = mode == "train"
        X_A, X_B, X_C, X_Vis = self.prepare_tensors(df_proc, fit_scaler=fit_scaler)
        y = df_proc["contact"].values if "contact" in df_proc.columns else None

        return (X_A, X_B, X_C, X_Vis), y, df_proc[["contact_id"]]
