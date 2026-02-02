import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler
from library.config import Config
from library.utils import set_seed


class FeatureGenerator:
    """
    Handles data preprocessing, feature engineering, and caching for the NFL Contact Detection task.
    Generates temporal windows of features and saves them in a format suitable for 1D-CNNs.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.offsets = list(range(-4, 5))  # -4 to +4 (9 frames)

        # Mapping for cyclical features
        self.angle_features = ["direction", "orientation"]

    def load_tracking(self, path):
        """
        Loads and preprocesses tracking data.
        """
        df = pd.read_csv(path)

        # Precompute cyclical features for raw tracking data
        for col in self.angle_features:
            if col in df.columns:
                # Convert degrees to radians
                rad = np.deg2rad(df[col])
                df[f"sin_{col}"] = np.sin(rad)
                df[f"cos_{col}"] = np.cos(rad)

        return df

    def _compute_interactions(self, df):
        """
        Computes interaction features between Player 1 and Player 2.
        Assumes columns are suffixed with _p1 and _p2.
        """
        # Distance
        df["distance"] = np.sqrt(
            (df["x_position_p1"] - df["x_position_p2"]) ** 2
            + (df["y_position_p1"] - df["y_position_p2"]) ** 2
        )

        # Relative Speed (Magnitude of velocity difference vector)
        # v_x = speed * sin(direction)
        # v_y = speed * cos(direction)
        # Note: We already computed sin/cos of direction.
        # Tracking data: 0 deg is Y-axis (usually), but consistency matters most.

        v1_x = df["speed_p1"] * df["sin_direction_p1"]
        v1_y = df["speed_p1"] * df["cos_direction_p1"]
        v2_x = df["speed_p2"] * df["sin_direction_p2"]
        v2_y = df["speed_p2"] * df["cos_direction_p2"]

        df["rel_speed"] = np.sqrt((v1_x - v2_x) ** 2 + (v1_y - v2_y) ** 2)

        # Relative Acceleration (Magnitude of acceleration difference vector)
        # Assuming acceleration aligns with direction for vector approximation
        a1_x = df["acceleration_p1"] * df["sin_direction_p1"]
        a1_y = df["acceleration_p1"] * df["cos_direction_p1"]
        a2_x = df["acceleration_p2"] * df["sin_direction_p2"]
        a2_y = df["acceleration_p2"] * df["cos_direction_p2"]

        df["rel_acceleration"] = np.sqrt((a1_x - a2_x) ** 2 + (a1_y - a2_y) ** 2)

        return df

    def generate_features(self, df_meta, df_tracking, is_train=False):
        """
        Generates the feature set for the given metadata.
        Applies undersampling if is_train is True.
        """
        # 1. Undersampling (only for training)
        if is_train:
            positives = df_meta[df_meta["contact"] == 1]
            negatives = df_meta[df_meta["contact"] == 0]

            n_pos = len(positives)
            n_neg = int(n_pos * Config.NEGATIVE_RATIO)

            if len(negatives) > n_neg:
                negatives = negatives.sample(n=n_neg, random_state=Config.SEED)

            df_meta = (
                pd.concat([positives, negatives])
                .sample(frac=1, random_state=Config.SEED)
                .reset_index(drop=True)
            )

        # Filter tracking data to relevant games to speed up merges
        relevant_games = df_meta["game_play"].unique()
        df_tracking = df_tracking[df_tracking["game_play"].isin(relevant_games)].copy()

        # Prepare list to store feature dataframes for each timestep
        time_slices = []

        # 2. Temporal Window Generation
        for offset in self.offsets:
            # Create a temporary copy of metadata with shifted steps
            df_step = df_meta[
                ["game_play", "step", "nfl_player_id_1", "nfl_player_id_2"]
            ].copy()
            df_step["step_shifted"] = df_step["step"] + offset

            # Merge Player 1
            # Rename tracking columns to _p1
            track_p1 = df_tracking.add_suffix("_p1")
            df_merged = pd.merge(
                df_step,
                track_p1,
                left_on=["game_play", "step_shifted", "nfl_player_id_1"],
                right_on=["game_play_p1", "step_p1", "nfl_player_id_p1"],
                how="left",
            )

            # Merge Player 2
            # Rename tracking columns to _p2
            track_p2 = df_tracking.add_suffix("_p2")

            # Handle 'G' (Ground) in join keys by converting to compatible type if necessary
            # But tracking IDs are float/int. 'G' is string.
            # We merge on columns. If p2 is 'G', it won't match anything in tracking (which has numeric IDs).
            # This results in NaNs, which is what we want for now.
            df_merged = pd.merge(
                df_merged,
                track_p2,
                left_on=["game_play", "step_shifted", "nfl_player_id_2"],
                right_on=["game_play_p2", "step_p2", "nfl_player_id_p2"],
                how="left",
            )

            # 3. Handle Ground and Missing Data
            # is_ground feature
            df_merged["is_ground"] = (df_merged["nfl_player_id_2"] == "G").astype(float)

            # Fill NaNs with 0 (for Ground P2, or missing tracking data)
            # We only fill feature columns.
            feature_cols_p1 = [c for c in df_merged.columns if c.endswith("_p1")]
            feature_cols_p2 = [c for c in df_merged.columns if c.endswith("_p2")]

            df_merged[feature_cols_p1] = df_merged[feature_cols_p1].fillna(0)
            df_merged[feature_cols_p2] = df_merged[feature_cols_p2].fillna(0)

            # 4. Compute Interactions
            df_merged = self._compute_interactions(df_merged)

            # 5. Select Ordered Features
            # Ensure all features exist (some might be missing if fillna didn't catch them or calc failed)
            for feat in Config.FEATURES:
                if feat not in df_merged.columns:
                    df_merged[feat] = 0.0

            # Extract only the features in the correct order
            X_slice = df_merged[Config.FEATURES].copy()

            # Add suffix for the time step (for flattening later)
            # We use index 0 to 8 for the 9 steps
            step_idx = offset + 4
            X_slice.columns = [f"{col}_{step_idx}" for col in X_slice.columns]

            time_slices.append(X_slice)

        # 6. Concatenate all time slices horizontally
        # Result is (N, Features * 9)
        X_flattened = pd.concat(time_slices, axis=1)

        # Extract Labels and IDs
        y = (
            df_meta["contact"].values
            if "contact" in df_meta.columns
            else np.zeros(len(df_meta))
        )
        ids = df_meta["contact_id"].values

        return X_flattened, y, ids

    def _scale_data(self, X_df, is_train):
        """
        Scales the data using StandardScaler.
        Reshapes to (N*T, C) to share statistics across time steps, then reshapes back.
        """
        # X_df shape: (N, C*T)
        n_samples = len(X_df)
        n_features = Config.NUM_FEATURES
        n_steps = Config.WINDOW_SIZE

        # Reshape to (N*T, C)
        # We need to ensure the columns are ordered: f1_t0, f2_t0... f1_t1, f2_t1...
        # Current columns are f1_t0, f2_t0... (from concat axis 1)
        # Actually, concat(axis=1) appends columns: [feat_t0..., feat_t1...]
        # So we can just reshape the values.

        X_values = X_df.values.reshape(n_samples * n_steps, n_features)

        if is_train:
            self.scaler.fit(X_values)
            joblib.dump(self.scaler, Config.SCALER_PATH)
        else:
            # Load scaler if not already loaded (though self.scaler is init, we should load fitted)
            if os.path.exists(Config.SCALER_PATH):
                self.scaler = joblib.load(Config.SCALER_PATH)
            else:
                raise FileNotFoundError("Scaler not found. Run training first.")

        X_scaled_values = self.scaler.transform(X_values)

        # Reshape back to (N, C*T) and reconstruct DataFrame
        X_scaled = pd.DataFrame(
            X_scaled_values.reshape(n_samples, n_features * n_steps),
            columns=X_df.columns,
            index=X_df.index,
        )

        return X_scaled

    def process_split(self, split, load_cached_data=True):
        """
        Main processing function for a data split (train, val, test).
        Handles caching logic.
        """
        # Determine paths based on split
        if split == "train":
            meta_path = Config.TRAIN_META_PATH
            track_path = Config.TRAIN_TRACKING_PATH
            cache_X = Config.CACHE_TRAIN_X
            cache_y = Config.CACHE_TRAIN_Y
            cache_ids = Config.CACHE_TRAIN_IDS
            is_train = True
        elif split == "validation":
            meta_path = Config.VAL_META_PATH
            track_path = Config.TRAIN_TRACKING_PATH
            cache_X = Config.CACHE_VAL_X
            cache_y = Config.CACHE_VAL_Y
            cache_ids = Config.CACHE_VAL_IDS
            is_train = False
        elif split == "test":
            meta_path = Config.TEST_META_PATH
            track_path = Config.TEST_TRACKING_PATH
            cache_X = Config.CACHE_TEST_X
            cache_y = None  # No labels for test
            cache_ids = Config.CACHE_TEST_IDS
            is_train = False
        else:
            raise ValueError(f"Unknown split: {split}")

        # Check Cache
        if load_cached_data and os.path.exists(cache_X):
            print(f"Loading cached {split} data...")
            X = pd.read_parquet(cache_X)
            ids = np.load(cache_ids)

            if split != "test":
                y = np.load(cache_y)
                return X, y, ids
            else:
                return X, None, ids

        print(f"Generating {split} data from scratch...")

        # Load Data
        df_meta = pd.read_csv(meta_path)
        df_tracking = self.load_tracking(track_path)

        # Generate Features
        X_raw, y, ids = self.generate_features(df_meta, df_tracking, is_train=is_train)

        # Scale Data
        X_scaled = self._scale_data(X_raw, is_train=is_train)

        # Save to Cache
        print(f"Saving {split} data to cache...")
        X_scaled.to_parquet(cache_X)
        np.save(cache_ids, ids)

        if split != "test":
            np.save(cache_y, y)
            return X_scaled, y, ids
        else:
            return X_scaled, None, ids
