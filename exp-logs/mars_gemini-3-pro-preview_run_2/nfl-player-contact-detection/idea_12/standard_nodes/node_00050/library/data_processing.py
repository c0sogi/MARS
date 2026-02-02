import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from library.config import Config
from library.utils import seed_everything


class DataProcessor:
    """
    Handles the 'Entity-First' data engineering pipeline.
    Generates wide-format temporal features, enforces hybrid ground imputation,
    and computes explicit relative physics metrics.
    """

    def __init__(self):
        self.scaler = None
        self.scaler_path = os.path.join(Config.WORKING_DIR, "scaler.joblib")
        seed_everything(Config.SEED)

    def _load_tracking_data(self, path, keep_game_plays=None):
        """
        Loads and filters tracking data to conserve memory.
        """
        # Read only necessary columns to save memory
        cols = ["game_play", "nfl_player_id", "step"] + Config.TRACKING_COLS
        df = pd.read_csv(path, usecols=cols)

        if keep_game_plays is not None:
            df = df[df["game_play"].isin(keep_game_plays)].copy()

        return df

    def _create_wide_tracking(self, df_tracking):
        """
        Transforms long tracking data into wide format with temporal windows (t-5 to t+5).
        Optimized to avoid slow groupby.shift operations.
        """
        # Sort to ensure temporal order for shifting
        df_tracking = df_tracking.sort_values(
            ["game_play", "nfl_player_id", "step"]
        ).reset_index(drop=True)

        # Base columns to shift
        feature_cols = Config.TRACKING_COLS

        # Dictionary to collect columns
        wide_data = {}

        # Keep keys for joining
        join_keys = ["game_play", "nfl_player_id", "step"]
        for k in join_keys:
            wide_data[k] = df_tracking[k].values

        # Generate shifts
        # We shift the entire dataframe and mask invalid transitions (crossing game or player boundaries)
        gp_vals = df_tracking["game_play"]
        pid_vals = df_tracking["nfl_player_id"]

        for offset in range(-Config.WINDOW_SIZE, Config.WINDOW_SIZE + 1):
            # Calculate shift amount.
            # To get value at t+k into row t, we look at row t+k.
            # This is a shift of -k (upwards).
            # To get value at t-k into row t, we look at row t-k.
            # This is a shift of +k (downwards).
            shift_amount = offset

            # Perform shift
            shifted_gp = gp_vals.shift(-shift_amount)
            shifted_pid = pid_vals.shift(-shift_amount)

            # Valid mask: ensure we didn't shift in data from a different player/play
            valid_mask = (gp_vals == shifted_gp) & (pid_vals == shifted_pid)

            # Suffix construction
            suffix = f"_t{offset}" if offset < 0 else f"_t+{offset}"
            if offset == 0:
                suffix = "_t0"

            for col in feature_cols:
                col_name = f"{col}{suffix}"
                # Shift data
                shifted_data = df_tracking[col].shift(-shift_amount)

                # Apply mask: fill invalid (out of bounds) with 0
                # Using 0 is standard padding for this architecture
                shifted_data[~valid_mask] = 0

                # Fill NaNs resulting from shift at dataframe edges
                wide_data[col_name] = shifted_data.fillna(0).values

        return pd.DataFrame(wide_data)

    def _engineer_interaction_features(self, df):
        """
        Computes explicit physics features (distance, relative speed, closing speed)
        for every timestep in the window.
        """
        for offset in range(-Config.WINDOW_SIZE, Config.WINDOW_SIZE + 1):
            suffix = f"_t{offset}" if offset < 0 else f"_t+{offset}"
            if offset == 0:
                suffix = "_t0"

            # Extract coordinates and kinematics
            x1 = df[f"x_position_1{suffix}"]
            y1 = df[f"y_position_1{suffix}"]
            x2 = df[f"x_position_2{suffix}"]
            y2 = df[f"y_position_2{suffix}"]

            s1 = df[f"speed_1{suffix}"]
            s2 = df[f"speed_2{suffix}"]
            d1 = df[f"direction_1{suffix}"]
            d2 = df[f"direction_2{suffix}"]

            # 1. Distance
            dx = x1 - x2
            dy = y1 - y2
            dist = np.sqrt(dx**2 + dy**2)

            df[f"distance{suffix}"] = dist
            df[f"log_distance{suffix}"] = np.log1p(dist)

            # 2. Velocity Vectors
            # NFL Tracking: 0 deg is Y-axis (North), 90 deg is X-axis (East)
            # vx = speed * sin(theta), vy = speed * cos(theta)
            rad1 = np.radians(d1)
            rad2 = np.radians(d2)

            vx1 = s1 * np.sin(rad1)
            vy1 = s1 * np.cos(rad1)
            vx2 = s2 * np.sin(rad2)
            vy2 = s2 * np.cos(rad2)

            # 3. Relative Speed (Magnitude of velocity difference)
            dvx = vx1 - vx2
            dvy = vy1 - vy2
            rel_speed = np.sqrt(dvx**2 + dvy**2)
            df[f"relative_speed{suffix}"] = rel_speed

            # 4. Closing Speed
            # Rate at which distance is decreasing.
            # Closing Speed = - (v_rel . r_rel) / |r_rel|
            # Positive value means closing in.
            dot_prod = dx * dvx + dy * dvy
            dist_clamped = np.maximum(dist, 1e-6)  # Avoid division by zero
            closing_speed = -(dot_prod) / dist_clamped

            df[f"closing_speed{suffix}"] = closing_speed

        return df

    def _process_dataset(self, df_meta, df_tracking):
        """
        Core pipeline: Wide tracking creation -> Merge -> Ground Imputation -> Feature Engineering.
        """
        # 1. Create Wide Tracking Features
        # This creates a dataframe with one row per (play, player, step) containing t-5...t+5 data
        df_wide = self._create_wide_tracking(df_tracking)

        # Identify tracking columns for renaming
        tracking_cols_base = [
            c
            for c in df_wide.columns
            if c not in ["game_play", "nfl_player_id", "step"]
        ]

        # 2. Merge Player 1
        # Rename map: x_position_t0 -> x_position_1_t0
        rename_1 = {
            c: f"{c.split('_t')[0]}_1_t{c.split('_t')[1]}" for c in tracking_cols_base
        }

        df_merged = df_meta.merge(
            df_wide,
            left_on=["game_play", "nfl_player_id_1", "step"],
            right_on=["game_play", "nfl_player_id", "step"],
            how="left",
        )
        df_merged = df_merged.rename(columns=rename_1)
        df_merged = df_merged.drop(columns=["nfl_player_id"], errors="ignore")

        # 3. Merge Player 2
        # Handle 'G' (Ground) by converting to numeric (G becomes NaN)
        df_merged["nfl_player_id_2_num"] = pd.to_numeric(
            df_merged["nfl_player_id_2"], errors="coerce"
        )

        rename_2 = {
            c: f"{c.split('_t')[0]}_2_t{c.split('_t')[1]}" for c in tracking_cols_base
        }

        df_merged = df_merged.merge(
            df_wide,
            left_on=["game_play", "nfl_player_id_2_num", "step"],
            right_on=["game_play", "nfl_player_id", "step"],
            how="left",
            suffixes=("", "_drop"),
        )
        df_merged = df_merged.rename(columns=rename_2)
        df_merged = df_merged.drop(
            columns=["nfl_player_id", "nfl_player_id_2_num"], errors="ignore"
        )

        # 4. Hybrid Ground Imputation
        # If Player 2 is Ground, set P2 Pos = P1 Pos, P2 Kinematics = 0
        is_ground = df_merged["nfl_player_id_2"] == "G"
        df_merged["is_ground"] = is_ground.astype(int)

        # List of kinematic features that should be zeroed for ground
        kinematics = ["speed", "acceleration", "orientation", "direction", "sa"]

        for offset in range(-Config.WINDOW_SIZE, Config.WINDOW_SIZE + 1):
            suffix = f"_t{offset}" if offset < 0 else f"_t+{offset}"
            if offset == 0:
                suffix = "_t0"

            p1_x = f"x_position_1{suffix}"
            p1_y = f"y_position_1{suffix}"
            p2_x = f"x_position_2{suffix}"
            p2_y = f"y_position_2{suffix}"

            # Impute Position
            df_merged.loc[is_ground, p2_x] = df_merged.loc[is_ground, p1_x]
            df_merged.loc[is_ground, p2_y] = df_merged.loc[is_ground, p1_y]

            # Impute Kinematics
            for k in kinematics:
                p2_col = f"{k}_2{suffix}"
                df_merged.loc[is_ground, p2_col] = 0.0

        # Fill remaining NaNs (missing tracking data) with 0
        # Construct list of all feature columns to fill
        all_feature_cols = []
        for offset in range(-Config.WINDOW_SIZE, Config.WINDOW_SIZE + 1):
            suffix = f"_t{offset}" if offset < 0 else f"_t+{offset}"
            if offset == 0:
                suffix = "_t0"
            for feat in Config.P1_FEATURES + Config.P2_FEATURES:
                # Map config name (x_position_1) to actual col name (x_position_1_t0)
                base_feat = feat  # e.g., x_position_1
                col_name = f"{base_feat}{suffix}"
                all_feature_cols.append(col_name)

        df_merged[all_feature_cols] = df_merged[all_feature_cols].fillna(0)

        # 5. Compute Interaction Features
        df_merged = self._engineer_interaction_features(df_merged)

        # 6. Final Feature Selection
        final_cols = Config.get_feature_names()

        # Ensure all columns exist (fill 0 if somehow missing)
        for c in final_cols:
            if c not in df_merged.columns:
                df_merged[c] = 0.0

        X = df_merged[final_cols].values.astype(np.float32)

        if "contact" in df_merged.columns:
            y = df_merged["contact"].values.astype(np.float32)
        else:
            y = None

        return X, y

    def get_train_val_data(self, load_cached_data=True):
        """
        Returns processed X_train, y_train, X_val, y_val.
        Uses caching to avoid re-processing.
        """
        cache_files = ["X_train.npy", "y_train.npy", "X_val.npy", "y_val.npy"]
        cache_paths = [os.path.join(Config.WORKING_DIR, f) for f in cache_files]

        # Check cache
        if (
            load_cached_data
            and all(os.path.exists(p) for p in cache_paths)
            and os.path.exists(self.scaler_path)
        ):
            print("Loading cached training data...")
            X_train = np.load(cache_paths[0])
            y_train = np.load(cache_paths[1])
            X_val = np.load(cache_paths[2])
            y_val = np.load(cache_paths[3])
            self.scaler = joblib.load(self.scaler_path)
            return X_train, y_train, X_val, y_val

        print("Processing training data from scratch...")

        # Load Metadata
        df_train_meta = pd.read_csv(Config.TRAIN_METADATA_PATH)
        df_val_meta = pd.read_csv(Config.VAL_METADATA_PATH)

        # Load Tracking (filter for relevant games)
        train_gps = df_train_meta["game_play"].unique()
        val_gps = df_val_meta["game_play"].unique()
        all_gps = np.concatenate([train_gps, val_gps])

        df_tracking = self._load_tracking_data(
            Config.TRAIN_TRACKING_PATH, keep_game_plays=all_gps
        )

        # Process Splits
        print("Generating Train features...")
        X_train, y_train = self._process_dataset(df_train_meta, df_tracking)

        print("Generating Validation features...")
        X_val, y_val = self._process_dataset(df_val_meta, df_tracking)

        # Scaling
        print("Fitting StandardScaler...")
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)

        # Save to Cache
        print("Saving processed data to cache...")
        np.save(cache_paths[0], X_train)
        np.save(cache_paths[1], y_train)
        np.save(cache_paths[2], X_val)
        np.save(cache_paths[3], y_val)
        joblib.dump(self.scaler, self.scaler_path)

        return X_train, y_train, X_val, y_val

    def get_test_data(self, load_cached_data=True):
        """
        Returns processed X_test and the corresponding metadata dataframe.
        """
        df_test_meta = pd.read_csv(Config.TEST_METADATA_PATH)
        cache_path = os.path.join(Config.WORKING_DIR, "X_test.npy")

        if load_cached_data and os.path.exists(cache_path):
            print("Loading cached test data...")
            X_test = np.load(cache_path)
            return X_test, df_test_meta

        print("Processing test data from scratch...")

        # Load Tracking
        df_tracking = self._load_tracking_data(Config.TEST_TRACKING_PATH)

        # Process
        X_test, _ = self._process_dataset(df_test_meta, df_tracking)

        # Scale
        if self.scaler is None:
            if os.path.exists(self.scaler_path):
                self.scaler = joblib.load(self.scaler_path)
            else:
                raise ValueError(
                    "Scaler not found. Ensure training data is processed first."
                )

        X_test = self.scaler.transform(X_test)

        # Cache
        np.save(cache_path, X_test)

        return X_test, df_test_meta
