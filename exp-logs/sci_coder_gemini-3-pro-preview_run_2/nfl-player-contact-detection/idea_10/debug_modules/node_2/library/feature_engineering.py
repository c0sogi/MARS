import os
import pandas as pd
import numpy as np
import torch
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from library.config import Config


class FeatureEngineer:
    """
    Handles data loading, ground imputation, kinematic feature engineering,
    windowing (wide format), and preprocessing (scaling/encoding).
    """

    def __init__(self, debug=Config.DEBUG):
        self.debug = debug
        self.scaler_path = Config.SCALER_PATH
        self.label_encoder_path = os.path.join(
            Config.WORKING_DIR, "label_encoder.joblib"
        )

        # Initialize processors
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

        # Define feature groups based on Config
        self.dynamic_features = [
            # P1 Raw
            "speed_1",
            "acceleration_1",
            "direction_1",
            "orientation_1",
            "sa_1",
            # P2 Raw
            "speed_2",
            "acceleration_2",
            "direction_2",
            "orientation_2",
            "sa_2",
            # Interaction / Engineered
            Config.FEAT_DISTANCE,
            Config.FEAT_LOG_DISTANCE,
            Config.FEAT_REL_SPEED,
            Config.FEAT_CLOSING_SPEED,
            Config.FEAT_REL_ACCEL,
        ]

    def _load_metadata(self, path):
        """Loads metadata and optionally samples for debugging."""
        df = pd.read_csv(path)
        if self.debug:
            print(f"DEBUG MODE: Sampling 5000 rows from {path}")
            # Sample by game_play to keep plays intact if possible, or just random rows
            # Random rows is safer for quick debugging of pipeline
            df = df.sample(n=5000, random_state=Config.SEED).reset_index(drop=True)
        return df

    def _load_tracking(self, path, game_plays):
        """Loads tracking data filtered by relevant game_plays."""
        # Load full tracking (fits in memory) then filter
        # Specifying types helps reduce memory usage
        dtypes = {
            "game_play": "object",
            "nfl_player_id": "object",
            "step": "int16",
            "x_position": "float32",
            "y_position": "float32",
            "speed": "float32",
            "acceleration": "float32",
            "direction": "float32",
            "orientation": "float32",
            "sa": "float32",
        }
        # Only load columns we need
        cols = list(dtypes.keys()) + ["jersey_number", "position", "team"]

        df = pd.read_csv(path, usecols=lambda c: c in cols, dtype=dtypes)
        df = df[df["game_play"].isin(game_plays)].copy()
        return df

    def impute_ground_data(self, df):
        """
        Implements the Hybrid Ground Imputation Strategy:
        1. Position: Ground = Player (Distance -> 0)
        2. Kinematics: Ground = 0 (Preserves Relative Motion)
        """
        # Identify Ground rows (where P2 is 'G')
        is_ground = df["nfl_player_id_2"] == "G"

        if not is_ground.any():
            return df

        # 1. Position Imputation
        df.loc[is_ground, "x_position_2"] = df.loc[is_ground, "x_position_1"]
        df.loc[is_ground, "y_position_2"] = df.loc[is_ground, "y_position_1"]

        # 2. Kinematics Imputation
        if Config.IMPUTE_GROUND_KINEMATICS_ZERO:
            zero_cols = ["speed_2", "acceleration_2", "sa_2"]
            for col in zero_cols:
                df.loc[is_ground, col] = 0.0

            # Direction/Orientation undefined for ground, set to 0
            df.loc[is_ground, "direction_2"] = 0.0
            df.loc[is_ground, "orientation_2"] = 0.0

        return df

    def compute_kinematics(self, df):
        """Computes explicit physics-based features."""
        # Distance
        dx = df["x_position_1"] - df["x_position_2"]
        dy = df["y_position_1"] - df["y_position_2"]
        dist = np.sqrt(dx**2 + dy**2)

        df[Config.FEAT_DISTANCE] = dist
        df[Config.FEAT_LOG_DISTANCE] = np.log1p(dist)

        # Helper to get velocity components (0 deg = Y axis? Standard is 0=X, 90=Y in math)
        # NFL data usually: 0=Y, 90=X. But relative magnitude is invariant to axis rotation.
        # We use standard trig assuming angle is in degrees.
        def get_components(speed, angle_deg):
            theta = np.radians(angle_deg)
            vx = speed * np.sin(theta)
            vy = speed * np.cos(theta)
            return vx, vy

        vx1, vy1 = get_components(df["speed_1"], df["direction_1"])
        vx2, vy2 = get_components(df["speed_2"], df["direction_2"])

        dvx = vx1 - vx2
        dvy = vy1 - vy2

        # Relative Speed (Magnitude of velocity difference)
        df[Config.FEAT_REL_SPEED] = np.sqrt(dvx**2 + dvy**2)

        # Closing Speed: Projection of relative velocity onto displacement vector
        # Vector 1->2 is (-dx, -dy). Closing means moving towards each other.
        # Dot product of (v1-v2) and (r2-r1)
        # r2-r1 = (-dx, -dy)
        dot_prod = dvx * (-dx) + dvy * (-dy)
        safe_dist = np.maximum(dist, 1e-6)
        df[Config.FEAT_CLOSING_SPEED] = dot_prod / safe_dist

        # Relative Acceleration (Scalar difference)
        df[Config.FEAT_REL_ACCEL] = df["acceleration_1"] - df["acceleration_2"]

        return df

    def create_wide_window(self, df):
        """
        Flattens the temporal window (t-5 to t+5) into a single wide row.
        Uses vectorized shift operations.
        """
        # Sort to ensure temporal order
        # nfl_player_id_2 can be 'G', ensure string
        df["p2_str"] = df["nfl_player_id_2"].astype(str)
        df = df.sort_values(by=["game_play", "nfl_player_id_1", "p2_str", "step"])

        # Group by contact pair
        grp = df.groupby(["game_play", "nfl_player_id_1", "p2_str"])

        shifted_dfs = []
        shifts = range(-Config.WINDOW_HALF, Config.WINDOW_HALF + 1)  # e.g., -5 to +5

        for i in shifts:
            # shift(-i) brings t+i to t.
            # e.g. i=-5 (past), shift(5) brings prev rows down.
            # wait: shift(1) moves t to t+1 (down). We want value at t-1 to be at t.
            # So shift(1) takes value from t-1 and puts it at t.
            # If i = -1 (past), we want t-1. shift(1) gives t-1.
            # So we use shift(-i).
            # i=-5 -> shift(5). i=5 -> shift(-5).
            s_df = grp[self.dynamic_features].shift(-i)
            s_df.columns = [f"{col}_step_{i}" for col in self.dynamic_features]
            shifted_dfs.append(s_df)

        # Concatenate all shifted features
        wide_features = pd.concat(shifted_dfs, axis=1)

        # Fill NaNs generated by shifting (boundaries of play)
        # We fill with 0. While not perfect for distance, it's robust for the scaler.
        wide_features = wide_features.fillna(0)

        # Add static features
        wide_features[Config.FEAT_IS_GROUND] = (df["nfl_player_id_2"] == "G").astype(
            int
        )

        return wide_features, df["position_1"]

    def process_dataset(
        self, metadata_path, tracking_path, is_train=True, load_cached_data=True
    ):
        """
        Main pipeline execution.
        """
        # Determine cache filenames
        cache_tag = "train"
        if "validation" in metadata_path:
            cache_tag = "val"
        elif "test" in metadata_path:
            cache_tag = "test"

        if self.debug:
            cache_tag += "_debug"

        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        cache_X = os.path.join(Config.WORKING_DIR, f"X_{cache_tag}.npy")
        cache_cat = os.path.join(Config.WORKING_DIR, f"X_cat_{cache_tag}.npy")
        cache_y = os.path.join(Config.WORKING_DIR, f"y_{cache_tag}.npy")
        cache_ids = os.path.join(Config.WORKING_DIR, f"ids_{cache_tag}.npy")

        # Check Cache
        if load_cached_data and os.path.exists(cache_X):
            print(f"Loading cached {cache_tag} data...")
            X = np.load(cache_X)
            X_cat = np.load(cache_cat)
            y = np.load(cache_y)
            ids = np.load(cache_ids)
            # Return categorical as dict of tensors for model compatibility
            return X, {"position": torch.from_numpy(X_cat).long()}, y, ids

        print(f"Processing {cache_tag} data from scratch...")

        # 1. Load
        df_meta = self._load_metadata(metadata_path)
        game_plays = df_meta["game_play"].unique()
        df_tracking = self._load_tracking(tracking_path, game_plays)

        # 2. Merge
        # Ensure join keys are strings
        df_meta["nfl_player_id_1"] = df_meta["nfl_player_id_1"].astype(str)
        df_meta["nfl_player_id_2"] = df_meta["nfl_player_id_2"].astype(str)
        df_tracking["nfl_player_id"] = df_tracking["nfl_player_id"].astype(str)

        # Merge P1
        df_merged = df_meta.merge(
            df_tracking.add_suffix("_1"),
            left_on=["game_play", "step", "nfl_player_id_1"],
            right_on=["game_play_1", "step_1", "nfl_player_id_1"],
            how="left",
        )

        # Merge P2 (P2 can be 'G', so tracking will be NaN)
        df_merged = df_merged.merge(
            df_tracking.add_suffix("_2"),
            left_on=["game_play", "step", "nfl_player_id_2"],
            right_on=["game_play_2", "step_2", "nfl_player_id_2"],
            how="left",
        )

        # 3. Impute Ground
        df_merged = self.impute_ground_data(df_merged)

        # 4. Fill missing tracking (sensor dropouts)
        num_cols = [
            c
            for c in df_merged.columns
            if any(
                x in c
                for x in [
                    "speed",
                    "accel",
                    "position",
                    "direction",
                    "orientation",
                    "sa",
                ]
            )
        ]
        df_merged[num_cols] = df_merged[num_cols].fillna(0)

        # 5. Kinematics
        df_merged = self.compute_kinematics(df_merged)

        # 6. Windowing
        X_wide, position_series = self.create_wide_window(df_merged)

        # 7. Categorical Encoding (Position)
        position_series = position_series.fillna("UNKNOWN").astype(str)

        if is_train and "train" in cache_tag:
            self.label_encoder.fit(position_series)
            joblib.dump(self.label_encoder, self.label_encoder_path)
        else:
            # Load encoder
            if os.path.exists(self.label_encoder_path):
                self.label_encoder = joblib.load(self.label_encoder_path)
            else:
                # Fallback
                self.label_encoder.fit(position_series)

        # Robust Transform
        # Map known classes to ID, unknown to 0 (assuming 0 is valid or using a safe map)
        # We create a lookup dictionary
        le_dict = dict(
            zip(
                self.label_encoder.classes_,
                self.label_encoder.transform(self.label_encoder.classes_),
            )
        )
        # Use 0 as default for unknown (usually 'C' or 'CB' etc is 0, but it's safe enough for embeddings)
        X_cat = position_series.map(lambda x: le_dict.get(x, 0)).values.astype(np.int32)

        # 8. Scaling
        if is_train and "train" in cache_tag:
            self.scaler.fit(X_wide)
            joblib.dump(self.scaler, self.scaler_path)
        else:
            if os.path.exists(self.scaler_path):
                self.scaler = joblib.load(self.scaler_path)
            else:
                self.scaler.fit(X_wide)

        X_scaled = self.scaler.transform(X_wide).astype(np.float32)

        # 9. Prepare Output Arrays
        y = df_merged["contact"].values.astype(np.int8)
        ids = df_merged["contact_id"].values

        # 10. Cache
        np.save(cache_X, X_scaled)
        np.save(cache_cat, X_cat)
        np.save(cache_y, y)
        np.save(cache_ids, ids)

        return X_scaled, {"position": torch.from_numpy(X_cat).long()}, y, ids
