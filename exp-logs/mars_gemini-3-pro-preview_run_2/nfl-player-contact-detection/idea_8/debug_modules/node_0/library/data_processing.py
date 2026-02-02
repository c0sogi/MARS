import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from library import config, utils


class FeatureEngineer:
    def __init__(self):
        """
        Initializes the FeatureEngineer with caching paths and scaler placeholders.
        """
        utils.set_seed()
        self.cache_dir = config.WORKING_DIR
        self.scaler_mean = None
        self.scaler_scale = None
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_paths(self, split, debug):
        """Generates file paths for cached data."""
        suffix = "_debug" if debug else ""
        return {
            "features": os.path.join(
                self.cache_dir, f"{split}_features{suffix}.parquet"
            ),
            "labels": os.path.join(self.cache_dir, f"{split}_labels{suffix}.npy"),
            "ids": os.path.join(self.cache_dir, f"{split}_ids{suffix}.npy"),
            "scaler_mean": os.path.join(self.cache_dir, f"scaler_mean{suffix}.npy"),
            "scaler_scale": os.path.join(self.cache_dir, f"scaler_scale{suffix}.npy"),
        }

    def load_tracking_data(self, game_plays):
        """
        Loads and filters tracking data for the specified game plays.
        Computes velocity and acceleration components.
        """
        dfs = []
        # Tracking data is split into train and test files
        for name in ["train", "test"]:
            path = os.path.join(config.INPUT_DIR, f"{name}_player_tracking.csv")
            if not os.path.exists(path):
                continue

            # Read only required columns
            df = pd.read_csv(path, usecols=config.TRACKING_COLS)

            # Filter for relevant plays to save memory
            df = df[df["game_play"].isin(game_plays)]
            if not df.empty:
                dfs.append(df)

        if not dfs:
            # Return empty dataframe with expected columns if no data found
            return pd.DataFrame(columns=config.TRACKING_COLS + ["vx", "vy", "ax", "ay"])

        tracking = pd.concat(dfs, ignore_index=True)

        # Pre-process kinematics: Convert Speed/Direction to Components
        # Assuming 0 degrees is North (Y-axis) and 90 is East (X-axis) for standard conversion
        rad = np.radians(tracking["direction"])
        tracking["vx"] = tracking["speed"] * np.sin(rad)
        tracking["vy"] = tracking["speed"] * np.cos(rad)

        # Acceleration components (assuming acceleration aligns with motion direction)
        tracking["ax"] = tracking["acceleration"] * np.sin(rad)
        tracking["ay"] = tracking["acceleration"] * np.cos(rad)

        return tracking

    def merge_tracking(self, df, tracking):
        """
        Merges tracking data onto the labels/submission dataframe for both players.
        """
        # Ensure ID types match for merging
        df["nfl_player_id_1"] = df["nfl_player_id_1"].astype(str)
        df["nfl_player_id_2"] = df["nfl_player_id_2"].astype(str)
        tracking["nfl_player_id"] = tracking["nfl_player_id"].astype(str)

        # Columns to merge
        track_cols = [
            "game_play",
            "step",
            "nfl_player_id",
            "x_position",
            "y_position",
            "vx",
            "vy",
            "ax",
            "ay",
            "speed",
            "acceleration",
        ]

        # Merge Player 1
        p1_track = tracking[track_cols].add_suffix("_1")
        df = df.merge(
            p1_track,
            left_on=["game_play", "step", "nfl_player_id_1"],
            right_on=["game_play_1", "step_1", "nfl_player_id_1"],
            how="left",
        )

        # Merge Player 2
        p2_track = tracking[track_cols].add_suffix("_2")
        df = df.merge(
            p2_track,
            left_on=["game_play", "step", "nfl_player_id_2"],
            right_on=["game_play_2", "step_2", "nfl_player_id_2"],
            how="left",
        )

        return df

    def impute_ground_physics(self, df):
        """
        Imputes physics for Ground contacts (Player 2 = 'G').
        Sets Ground position to Player 1's position and kinematics to 0.
        """
        is_ground = df["nfl_player_id_2"] == "G"

        # Impute positions (Ground is where the player is)
        df.loc[is_ground, "x_position_2"] = df.loc[is_ground, "x_position_1"]
        df.loc[is_ground, "y_position_2"] = df.loc[is_ground, "y_position_1"]

        # Impute kinematics (Ground is static)
        cols_to_zero = ["vx_2", "vy_2", "ax_2", "ay_2", "speed_2", "acceleration_2"]
        for col in cols_to_zero:
            df.loc[is_ground, col] = 0.0

        df["is_ground"] = is_ground.astype(int)
        return df

    def calc_relative_kinematics(self, df):
        """
        Calculates explicit relative physics features defined in the Idea.
        """
        # 1. Distance (Log transformed)
        dx = df["x_position_1"] - df["x_position_2"]
        dy = df["y_position_1"] - df["y_position_2"]
        dist = np.sqrt(dx**2 + dy**2)
        df["distance"] = np.log1p(dist)

        # 2. Relative Speed (Magnitude of velocity difference vector)
        dvx = df["vx_1"] - df["vx_2"]
        dvy = df["vy_1"] - df["vy_2"]
        df["rel_speed"] = np.sqrt(dvx**2 + dvy**2)

        # 3. Relative Acceleration (Magnitude of acceleration difference vector)
        dax = df["ax_1"] - df["ax_2"]
        day = df["ay_1"] - df["ay_2"]
        df["rel_accel"] = np.sqrt(dax**2 + day**2)

        # 4. Closing Speed (Projection of relative velocity onto distance vector)
        # Closing Speed = - (V_rel . R) / |R|
        # Here we calculate the dot product.
        # Note: If players moving towards each other, dot product of (v1-v2) and (p1-p2) is positive?
        # v1 towards p2, v2 towards p1.
        # Let's use standard formulation: Rate of change of distance.
        # (v1x - v2x)*(x1 - x2) + ...
        dot_prod = (df["vx_1"] - df["vx_2"]) * dx + (df["vy_1"] - df["vy_2"]) * dy
        safe_dist = dist.replace(0, 1e-6)
        # If dot_prod is negative, they are closing in (distance decreasing).
        # We want "Closing Speed" as a positive magnitude for closing.
        df["closing_speed"] = -(dot_prod / safe_dist)

        # Rename columns to match config.INPUT_FEATURES
        df.rename(
            columns={"acceleration_1": "accel_1", "acceleration_2": "accel_2"},
            inplace=True,
        )

        # Fill NaNs (missing tracking data) with 0
        feat_cols = config.INPUT_FEATURES
        df[feat_cols] = df[feat_cols].fillna(0)

        return df

    def create_temporal_windows(self, df):
        """
        Creates a wide-format DataFrame with temporal windows (t-5 to t+5).
        Uses vectorized shifts respecting game/play/pair boundaries.
        """
        # Sort to ensure temporal order
        df = df.sort_values(["game_play", "nfl_player_id_1", "nfl_player_id_2", "step"])

        features = config.INPUT_FEATURES
        window_size = config.WINDOW_SIZE
        half = config.HALF_WINDOW

        shifts = range(-half, half + 1)
        shifted_data = {}

        for k in shifts:
            # Shift amount: To get value at t+k into row t, we shift by -k
            # e.g., k=1 (future), shift(-1) moves next row up.
            shift_amount = -k

            # Shift features
            feat_shifted = df[features].shift(shift_amount)

            # Mask invalid shifts (crossing game/play/pair boundaries)
            # Check if key columns match the shifted key columns
            mask = (
                (df["game_play"] == df["game_play"].shift(shift_amount))
                & (df["nfl_player_id_1"] == df["nfl_player_id_1"].shift(shift_amount))
                & (df["nfl_player_id_2"] == df["nfl_player_id_2"].shift(shift_amount))
            )

            # Fill invalid shifts with 0 (Zero Padding)
            feat_shifted = feat_shifted.where(mask, 0)

            # Rename columns with suffix
            feat_shifted.columns = [f"{col}_{k}" for col in features]
            shifted_data[k] = feat_shifted

        # Concatenate all time steps: t-5, t-4, ..., t+5
        result_dfs = [shifted_data[k] for k in sorted(shifted_data.keys())]
        wide_df = pd.concat(result_dfs, axis=1)

        return wide_df

    def process_dataset(self, split="train", load_cached_data=True, debug=False):
        """
        Main pipeline execution method.
        Checks cache, loads data, processes features, normalizes, and saves cache.
        """
        paths = self._get_cache_paths(split, debug)

        # 1. Attempt to Load Cache
        if (
            load_cached_data
            and os.path.exists(paths["features"])
            and os.path.exists(paths["labels"])
        ):
            print(f"Loading cached {split} data...")
            X_df = pd.read_parquet(paths["features"])
            y = np.load(paths["labels"])
            ids = np.load(paths["ids"])

            # Load scaler stats if training
            if split == "train" and os.path.exists(paths["scaler_mean"]):
                self.scaler_mean = np.load(paths["scaler_mean"])
                self.scaler_scale = np.load(paths["scaler_scale"])

            return X_df.values, y, ids

        # 2. Process from Scratch
        print(f"Processing {split} data from scratch...")

        # Load Metadata
        meta_path = os.path.join(config.METADATA_DIR, f"{split}.csv")
        df = pd.read_csv(meta_path)

        if debug:
            # Sample first 5 games for debugging
            gps = df["game_play"].unique()[:5]
            df = df[df["game_play"].isin(gps)].copy()

        # Load and Merge Tracking
        game_plays = df["game_play"].unique()
        tracking = self.load_tracking_data(game_plays)
        df = self.merge_tracking(df, tracking)

        # Feature Engineering
        df = self.impute_ground_physics(df)
        df = self.calc_relative_kinematics(df)

        # Create Temporal Windows (Wide Format)
        X_wide = self.create_temporal_windows(df)

        # Extract Targets and IDs
        y = df["contact"].values.astype(np.float32)
        ids = df["contact_id"].values

        # Normalization
        X_vals = X_wide.values.astype(np.float32)

        if split == "train":
            scaler = StandardScaler()
            X_vals = scaler.fit_transform(X_vals)
            self.scaler_mean = scaler.mean_
            self.scaler_scale = scaler.scale_

            # Save scaler parameters
            np.save(paths["scaler_mean"], self.scaler_mean)
            np.save(paths["scaler_scale"], self.scaler_scale)
        else:
            # Load scaler from train cache if not already in memory
            if self.scaler_mean is None:
                train_paths = self._get_cache_paths("train", debug)
                if os.path.exists(train_paths["scaler_mean"]):
                    self.scaler_mean = np.load(train_paths["scaler_mean"])
                    self.scaler_scale = np.load(train_paths["scaler_scale"])
                else:
                    raise ValueError("Scaler not fitted. Run training split first.")

            X_vals = (X_vals - self.scaler_mean) / self.scaler_scale

        # Save to Cache
        # Save Features as Parquet (preserves schema conceptually, though we use numpy for model)
        X_final_df = pd.DataFrame(X_vals, columns=X_wide.columns)
        X_final_df.to_parquet(paths["features"])
        np.save(paths["labels"], y)
        np.save(paths["ids"], ids)

        return X_vals, y, ids
