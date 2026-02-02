import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from library.config import Config


class DataProcessor:
    """
    Handles data loading, preprocessing, feature engineering, and caching for WIRK-Net.
    Implements the Wide-Input format and Hybrid Ground Imputation.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.pos_encoder = LabelEncoder()
        # Initialize scaler/encoder as None, load later if needed
        self.is_fitted = False

    def get_train_val_data(self, load_cached_data=True):
        """
        Main entry point to get processed Training and Validation data.
        Returns:
            X_train, y_train, X_val, y_val (DataFrames/Series)
        """
        # 1. Check Cache
        if (
            load_cached_data
            and os.path.exists(Config.TRAIN_FEATURES_PATH)
            and os.path.exists(Config.VAL_FEATURES_PATH)
        ):
            print("Loading cached training and validation features...")
            train_df = pd.read_parquet(Config.TRAIN_FEATURES_PATH)
            val_df = pd.read_parquet(Config.VAL_FEATURES_PATH)

            # Load fitted scaler/encoder
            if os.path.exists(Config.SCALER_PATH):
                saved_state = joblib.load(Config.SCALER_PATH)
                self.scaler = saved_state["scaler"]
                self.pos_encoder = saved_state["encoder"]
                self.is_fitted = True

            # Split X and y
            y_train = train_df["contact"]
            X_train = train_df.drop(
                columns=["contact", "game_play", "step", "contact_id", "datetime"]
            )

            y_val = val_df["contact"]
            X_val = val_df.drop(
                columns=["contact", "game_play", "step", "contact_id", "datetime"]
            )

            return X_train, y_train, X_val, y_val

        # 2. Process from Scratch
        print("Processing training and validation data from scratch...")

        # Load Metadata
        df_train_meta = pd.read_csv(Config.TRAIN_META_PATH)
        df_val_meta = pd.read_csv(Config.VAL_META_PATH)

        if Config.DEBUG:
            print(f"DEBUG MODE: Sampling {Config.DEBUG_SAMPLE_SIZE} rows.")
            df_train_meta = df_train_meta.iloc[: Config.DEBUG_SAMPLE_SIZE].copy()
            df_val_meta = df_val_meta.iloc[: int(Config.DEBUG_SAMPLE_SIZE * 0.2)].copy()

        # Load Tracking
        print("Loading tracking data...")
        df_tracking = pd.read_csv(Config.TRAIN_TRACKING_PATH)

        # Preprocess
        print("Processing Train set...")
        train_df = self._process_dataset(df_train_meta, df_tracking, is_train=True)
        print("Processing Validation set...")
        val_df = self._process_dataset(df_val_meta, df_tracking, is_train=False)

        # Save to Cache
        print("Caching features...")
        train_df.to_parquet(Config.TRAIN_FEATURES_PATH)
        val_df.to_parquet(Config.VAL_FEATURES_PATH)

        # Save Scaler/Encoder
        joblib.dump(
            {"scaler": self.scaler, "encoder": self.pos_encoder}, Config.SCALER_PATH
        )

        # Split X and y
        y_train = train_df["contact"]
        X_train = train_df.drop(
            columns=["contact", "game_play", "step", "contact_id", "datetime"]
        )

        y_val = val_df["contact"]
        X_val = val_df.drop(
            columns=["contact", "game_play", "step", "contact_id", "datetime"]
        )

        return X_train, y_train, X_val, y_val

    def get_test_data(self, load_cached_data=True):
        """
        Main entry point to get processed Test data.
        Returns:
            X_test (DataFrame), ids_test (Series)
        """
        if load_cached_data and os.path.exists(Config.TEST_FEATURES_PATH):
            print("Loading cached test features...")
            test_df = pd.read_parquet(Config.TEST_FEATURES_PATH)

            # Load scaler if not in memory
            if not self.is_fitted and os.path.exists(Config.SCALER_PATH):
                saved_state = joblib.load(Config.SCALER_PATH)
                self.scaler = saved_state["scaler"]
                self.pos_encoder = saved_state["encoder"]
                self.is_fitted = True

            ids_test = test_df["contact_id"]
            # Drop non-feature columns. Note: 'contact' might be present as placeholder 0
            cols_to_drop = ["contact", "game_play", "step", "contact_id", "datetime"]
            X_test = test_df.drop(
                columns=[c for c in cols_to_drop if c in test_df.columns]
            )
            return X_test, ids_test

        print("Processing test data from scratch...")
        df_test_meta = pd.read_csv(Config.TEST_META_PATH)
        df_tracking = pd.read_csv(Config.TEST_TRACKING_PATH)

        # Ensure we have the fitted scaler/encoder
        if not self.is_fitted:
            if os.path.exists(Config.SCALER_PATH):
                saved_state = joblib.load(Config.SCALER_PATH)
                self.scaler = saved_state["scaler"]
                self.pos_encoder = saved_state["encoder"]
                self.is_fitted = True
            else:
                raise ValueError(
                    "Scaler not found! You must process train data before test data."
                )

        test_df = self._process_dataset(df_test_meta, df_tracking, is_train=False)

        print("Caching test features...")
        test_df.to_parquet(Config.TEST_FEATURES_PATH)

        ids_test = test_df["contact_id"]
        cols_to_drop = ["contact", "game_play", "step", "contact_id", "datetime"]
        X_test = test_df.drop(columns=[c for c in cols_to_drop if c in test_df.columns])

        return X_test, ids_test

    def _process_dataset(self, df_meta, df_tracking, is_train=False):
        """
        Internal pipeline: Merge -> Impute Ground -> Feature Engineer -> Wide Window -> Scale
        """
        # Filter tracking to relevant games to speed up merge
        relevant_games = df_meta["game_play"].unique()
        df_track_subset = df_tracking[
            df_tracking["game_play"].isin(relevant_games)
        ].copy()

        # 1. Merge Tracking Data
        # Ensure nfl_player_id_1 is numeric
        df_meta["nfl_player_id_1"] = pd.to_numeric(
            df_meta["nfl_player_id_1"], errors="coerce"
        )

        # Handle nfl_player_id_2: 'G' becomes NaN when coerced to numeric
        df_meta["nfl_player_id_2_num"] = pd.to_numeric(
            df_meta["nfl_player_id_2"], errors="coerce"
        )

        # Select columns to merge
        track_cols = (
            ["game_play", "step", "nfl_player_id"]
            + Config.RAW_KINEMATIC_FEATURES
            + ["position"]
        )

        # Merge Player 1
        merged = df_meta.merge(
            df_track_subset[track_cols].add_suffix("_1"),
            left_on=["game_play", "step", "nfl_player_id_1"],
            right_on=["game_play_1", "step_1", "nfl_player_id_1"],
            how="left",
        )

        # Merge Player 2
        merged = merged.merge(
            df_track_subset[track_cols].add_suffix("_2"),
            left_on=["game_play", "step", "nfl_player_id_2_num"],
            right_on=["game_play_2", "step_2", "nfl_player_id_2"],
            how="left",
        )

        # 2. Hybrid Ground Imputation
        # Identify Ground rows
        is_ground_mask = merged["nfl_player_id_2"] == "G"
        merged["is_ground"] = is_ground_mask.astype(int)

        # Impute Position: P2 pos = P1 pos (Distance -> 0)
        for col in ["x_position", "y_position"]:
            merged.loc[is_ground_mask, f"{col}_2"] = merged.loc[
                is_ground_mask, f"{col}_1"
            ]

        # Impute Kinematics: P2 velocity/accel = 0
        zero_cols = ["speed", "acceleration", "sa", "orientation", "direction"]
        for col in zero_cols:
            merged.loc[is_ground_mask, f"{col}_2"] = 0.0

        # Impute Position Category for Ground
        # We'll use a placeholder 'GROUND' for the position column
        merged.loc[is_ground_mask, "position_2"] = "GROUND"

        # Fill remaining NaNs (missing tracking data for players) with 0 or appropriate defaults
        # This prevents errors in distance calculation
        kinematic_cols_1 = [f"{c}_1" for c in Config.RAW_KINEMATIC_FEATURES]
        kinematic_cols_2 = [f"{c}_2" for c in Config.RAW_KINEMATIC_FEATURES]
        merged[kinematic_cols_1] = merged[kinematic_cols_1].fillna(0)
        merged[kinematic_cols_2] = merged[kinematic_cols_2].fillna(0)
        merged["position_1"] = merged["position_1"].fillna("UNKNOWN")
        merged["position_2"] = merged["position_2"].fillna("UNKNOWN")

        # 3. Feature Engineering (Kinematics)
        # Distance
        dx = merged["x_position_1"] - merged["x_position_2"]
        dy = merged["y_position_1"] - merged["y_position_2"]
        dist = np.sqrt(dx**2 + dy**2)
        merged["distance"] = dist
        merged["log_distance"] = np.log1p(dist)

        # Relative Speed (Scalar difference)
        merged["relative_speed"] = np.abs(merged["speed_1"] - merged["speed_2"])

        # Relative Acceleration
        merged["relative_acceleration"] = np.abs(
            merged["acceleration_1"] - merged["acceleration_2"]
        )

        # Closing Speed: Project relative velocity vector onto distance vector
        # Convert orientation/direction to radians for vector components if needed
        # However, tracking data 'direction' is motion angle.
        # Vx = speed * sin(dir), Vy = speed * cos(dir) (NFL coordinates are usually rotated, but let's assume standard math for relative)
        # Actually, let's stick to the simpler scalar approximation if vector math is ambiguous without detailed coordinate system docs,
        # BUT the prompt mentions "Closing Speed" explicitly.
        # Standard approximation: -(diff_speed) is not enough.
        # Let's compute velocity vectors. NFL tracking: 0 degrees is Y-axis (usually).
        # We will use simple vector projection:
        # V_rel = V1 - V2.  P_rel = P2 - P1.
        # Closing Speed = dot(V_rel, P_rel / |P_rel|)

        def get_velocity_components(speed, direction_deg):
            # Convert to radians. 0 is North (Y), 90 is East (X) usually in NFL data
            # theta = deg2rad(direction)
            # vx = speed * sin(theta)
            # vy = speed * cos(theta)
            theta = np.radians(direction_deg)
            vx = speed * np.sin(theta)
            vy = speed * np.cos(theta)
            return vx, vy

        vx1, vy1 = get_velocity_components(merged["speed_1"], merged["direction_1"])
        vx2, vy2 = get_velocity_components(merged["speed_2"], merged["direction_2"])

        dvx = vx1 - vx2
        dvy = vy1 - vy2

        # Vector from P1 to P2
        p_dx = merged["x_position_2"] - merged["x_position_1"]
        p_dy = merged["y_position_2"] - merged["y_position_1"]

        # Normalize position vector (avoid div by zero)
        p_norm = np.sqrt(p_dx**2 + p_dy**2)
        p_norm = np.where(p_norm < 1e-6, 1e-6, p_norm)

        # Closing speed: How fast they are approaching. Positive = closing in.
        # Dot product of relative velocity (v1-v2) and direction to target.
        merged["closing_speed"] = dvx * (p_dx / p_norm) + dvy * (p_dy / p_norm)

        # 4. Encode Categoricals
        if is_train:
            # Fit encoder on both P1 and P2 positions
            all_pos = pd.concat([merged["position_1"], merged["position_2"]]).unique()
            # Ensure 'GROUND' and 'UNKNOWN' are in there
            self.pos_encoder.fit(all_pos.astype(str))

        # Transform
        # Handle unseen labels in validation/test by mapping to a default or 'UNKNOWN'
        # We use a safe transform helper
        def safe_transform(encoder, series):
            # Map unseen to 'UNKNOWN' if present, else first class
            known = set(encoder.classes_)
            s_str = series.astype(str)
            s_str = s_str.apply(lambda x: x if x in known else "UNKNOWN")
            # If UNKNOWN not in classes (rare), map to index 0
            if "UNKNOWN" not in known:
                # Fallback
                return np.zeros(len(s_str), dtype=int)
            return encoder.transform(s_str)

        merged["position_1_enc"] = safe_transform(
            self.pos_encoder, merged["position_1"]
        )
        merged["position_2_enc"] = safe_transform(
            self.pos_encoder, merged["position_2"]
        )

        # 5. Prepare for Wide Window
        # Select features to window
        # Raw kinematics (x2), Derived, Categorical Encoded (x2), is_ground
        feature_cols = []
        feature_cols += [f"{c}_1" for c in Config.RAW_KINEMATIC_FEATURES]
        feature_cols += [f"{c}_2" for c in Config.RAW_KINEMATIC_FEATURES]
        feature_cols += Config.DERIVED_FEATURES
        feature_cols += ["position_1_enc", "position_2_enc", "is_ground"]

        # Sort for windowing
        merged = merged.sort_values(["game_play", "step"])

        # Create Wide Features
        # Group by game_play to prevent shifting across plays
        grouped = merged.groupby("game_play")[feature_cols]

        wide_features = []
        feature_names = []

        # Window range: -W to +W
        for lag in range(-Config.WINDOW_SIZE, Config.WINDOW_SIZE + 1):
            # shift(lag): positive lag = previous data (t-1), negative lag = future data (t+1) in pandas?
            # pandas shift(1) moves t to t+1 (down). So row t gets data from t-1.
            # We want row t to contain data from t-W ... t ... t+W.
            # shift(k): row i gets value from i-k.
            # So shift(1) gives t-1. shift(-1) gives t+1.
            shifted = grouped.shift(lag)

            # Suffix
            suffix = f"_lag{lag}" if lag != 0 else ""
            shifted.columns = [f"{c}{suffix}" for c in feature_cols]

            wide_features.append(shifted)
            feature_names.extend(shifted.columns)

        # Concatenate all window steps horizontally
        df_wide = pd.concat(wide_features, axis=1)

        # Fill NaNs generated by shifting (edges of play)
        # We use 0 for padding
        df_wide = df_wide.fillna(0)

        # 6. Scaling
        # Identify continuous columns (exclude encoded categoricals and is_ground flags)
        # We scale all continuous features including lags
        # Continuous base features:
        cont_base = (
            [f"{c}_1" for c in Config.RAW_KINEMATIC_FEATURES]
            + [f"{c}_2" for c in Config.RAW_KINEMATIC_FEATURES]
            + Config.DERIVED_FEATURES
        )

        # Find all columns in df_wide that derive from cont_base
        cols_to_scale = [
            c for c in df_wide.columns if any(base in c for base in cont_base)
        ]

        if is_train:
            self.scaler.fit(df_wide[cols_to_scale])
            self.is_fitted = True

        df_wide[cols_to_scale] = self.scaler.transform(df_wide[cols_to_scale])

        # 7. Reattach Identifiers and Target
        # We need to ensure index alignment. merged and df_wide share the same index.
        meta_cols = ["contact_id", "game_play", "step", "datetime"]
        if "contact" in merged.columns:
            meta_cols.append("contact")

        final_df = pd.concat([merged[meta_cols], df_wide], axis=1)

        # Reduce memory usage
        float_cols = final_df.select_dtypes(include=["float64"]).columns
        final_df[float_cols] = final_df[float_cols].astype("float32")

        return final_df
