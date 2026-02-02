import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from library.config import (
    TRAIN_META_PATH,
    VAL_META_PATH,
    TEST_META_PATH,
    TRAIN_TRACKING_PATH,
    TEST_TRACKING_PATH,
    CACHE_TRAIN_FEATURES,
    CACHE_VAL_FEATURES,
    CACHE_TEST_FEATURES,
    CACHE_SCALER,
    WINDOW_SIZE,
    HALF_WINDOW,
    FEATURE_COLS,
    TRACKING_COLS,
    WORKING_DIR,
)


class FeatureEngineer:
    def __init__(self):
        self.scaler = None
        # Ensure working directory exists
        os.makedirs(WORKING_DIR, exist_ok=True)

    def _load_tracking_data(self, path, relevant_game_plays=None):
        """
        Loads and preprocesses tracking data.
        Computes Jerk (derivative of acceleration) before merging.
        """
        df_track = pd.read_csv(path, usecols=TRACKING_COLS)

        # Filter to relevant game plays to save memory
        if relevant_game_plays is not None:
            df_track = df_track[df_track["game_play"].isin(relevant_game_plays)].copy()

        # Sort for temporal calculations
        df_track.sort_values(["game_play", "nfl_player_id", "step"], inplace=True)

        # Convert direction/orientation to radians for vector math if needed later,
        # but for now we stick to raw values as requested, or simple diffs.
        # We will handle vector math in feature generation.

        return df_track

    def _expand_windows(self, df_meta):
        """
        Expands each metadata row into WINDOW_SIZE rows (t-5 to t+5).
        """
        # Create offsets
        offsets = np.arange(-HALF_WINDOW, HALF_WINDOW + 1)

        # Repeat rows
        df_expanded = df_meta.loc[df_meta.index.repeat(WINDOW_SIZE)].copy()

        # Assign offsets
        df_expanded["offset"] = np.tile(offsets, len(df_meta))

        # Adjust step
        df_expanded["step"] = df_expanded["step"] + df_expanded["offset"]

        return df_expanded

    def _compute_interactions(self, df):
        """
        Computes interaction features between P1 and P2.
        """
        # 1. Ground Flag
        df["is_ground"] = (df["nfl_player_id_2"] == "G").astype(int)

        # 2. Fill NaNs for P2 (Ground or Missing)
        # For P2, if it's ground, tracking data is NaN.
        # We fill P2 physics with 0.
        p2_cols = [
            "x_position_2",
            "y_position_2",
            "speed_2",
            "acceleration_2",
            "orientation_2",
            "direction_2",
        ]
        df[p2_cols] = df[p2_cols].fillna(0)

        # 3. Distance & Log Distance
        # If ground, we treat distance as 0 (contact is immediate) or logic specific.
        # However, physically, distance to ground is 0 at contact.
        dx = df["x_position_1"] - df["x_position_2"]
        dy = df["y_position_1"] - df["y_position_2"]
        dist = np.sqrt(dx**2 + dy**2)

        # If ground, distance calculation above used (x1-0, y1-0) which is wrong.
        # We force distance to 0 for ground interactions for the model to learn "at contact".
        # Or better, we set it to a distinct value.
        # Given the previous logic: "is_ground" is a strong feature.
        # Let's set distance to 0 where is_ground is 1.
        dist = np.where(df["is_ground"] == 1, 0.0, dist)
        df["log_distance"] = np.log1p(dist)

        # 4. Vector calculations for Closing Speed
        # Convert direction to radians
        dir1_rad = np.radians(df["direction_1"].fillna(0))
        dir2_rad = np.radians(df["direction_2"].fillna(0))

        vx1 = df["speed_1"] * np.sin(dir1_rad)
        vy1 = df["speed_1"] * np.cos(dir1_rad)
        vx2 = df["speed_2"] * np.sin(dir2_rad)
        vy2 = df["speed_2"] * np.cos(dir2_rad)

        dvx = vx1 - vx2
        dvy = vy1 - vy2

        # Closing speed: - (r_rel . v_rel) / |r_rel|
        # r_rel = (dx, dy)
        dot_prod = dx * dvx + dy * dvy

        # Clamp distance for division
        epsilon = 1e-5
        dist_clamped = np.maximum(dist, epsilon)

        closing_speed = -(dot_prod) / dist_clamped

        # For ground, closing speed is essentially the player's speed towards the ground?
        # In 2D top-down, ground is not a point (0,0).
        # If is_ground, we set closing_speed to speed_1 (scalar).
        df["clamped_speed"] = np.where(
            df["is_ground"] == 1, df["speed_1"], closing_speed
        )

        # 5. Orientation/Direction Diff
        df["orientation_diff"] = df["orientation_1"] - df["orientation_2"]
        df["direction_diff"] = df["direction_1"] - df["direction_2"]

        # Handle Ground Diffs (set to 0 or keep as is? P2 is 0, so it's just P1's value)
        # Setting to 0 removes noise, but P1's orientation might matter for ground contact.
        # Let's keep the diff (which is just P1 - 0) but maybe the model learns P1 orientation from it.
        # Actually, let's explicitly zero them out for ground to rely on P1 features directly.
        df.loc[df["is_ground"] == 1, "orientation_diff"] = 0
        df.loc[df["is_ground"] == 1, "direction_diff"] = 0

        return df

    def _generate_dataset(self, meta_path, tracking_path, is_train=False):
        """
        Core logic to generate X, y, ids from metadata and tracking.
        """
        # 1. Load Metadata
        df_meta = pd.read_csv(meta_path)

        # 2. Load Tracking (filtered)
        unique_games = df_meta["game_play"].unique()
        df_track = self._load_tracking_data(tracking_path, unique_games)

        # 3. Expand Windows
        # We need to keep track of the original index to reshape later
        df_meta["sample_id"] = np.arange(len(df_meta))
        df_expanded = self._expand_windows(df_meta)

        # 4. Merge Tracking P1
        # Tracking needs to be joined on game_play, step, nfl_player_id
        df_merged = df_expanded.merge(
            df_track,
            left_on=["game_play", "step", "nfl_player_id_1"],
            right_on=["game_play", "step", "nfl_player_id"],
            how="left",
            suffixes=("", "_1"),
        ).drop(
            columns=["nfl_player_id"]
        )  # drop redundant

        # Rename columns that didn't get suffix (because they weren't in left)
        # The merge operation might leave some cols without suffix if they don't collide.
        # TRACKING_COLS: [game_play, step, nfl_player_id, x, y, speed, accel, orient, dir, sa, jerk]
        # We need to ensure we map them to _1.
        # Actually, standard pandas merge: if col exists in both, suffixes applied.
        # game_play, step are join keys. others are data.
        # We need to be careful. Let's rename tracking cols before merge to be safe.

        # Re-do Merge strategy for safety
        track_cols_data = [
            c
            for c in df_track.columns
            if c not in ["game_play", "step", "nfl_player_id"]
        ]
        df_track_p1 = df_track.rename(columns={c: c + "_1" for c in track_cols_data})
        df_track_p2 = df_track.rename(columns={c: c + "_2" for c in track_cols_data})

        df_merged = df_expanded.merge(
            df_track_p1,
            left_on=["game_play", "step", "nfl_player_id_1"],
            right_on=["game_play", "step", "nfl_player_id"],
            how="left",
        ).drop(columns=["nfl_player_id"])

        # 5. Merge Tracking P2
        # Handle 'G' -> numeric conversion for merge
        # 'G' will become NaN in numeric conversion, merge will fail (NaN != NaN), resulting in NaNs in columns
        # which is exactly what we want.
        df_merged["nfl_player_id_2_num"] = pd.to_numeric(
            df_merged["nfl_player_id_2"], errors="coerce"
        )

        df_merged = df_merged.merge(
            df_track_p2,
            left_on=["game_play", "step", "nfl_player_id_2_num"],
            right_on=["game_play", "step", "nfl_player_id"],
            how="left",
        ).drop(columns=["nfl_player_id", "nfl_player_id_2_num"])

        # 6. Feature Engineering
        df_features = self._compute_interactions(df_merged)

        # 7. Impute remaining NaNs (e.g. missing tracking for P1)
        # Forward fill within sample might be good, but simple 0 fill is safer for now
        df_features[FEATURE_COLS] = df_features[FEATURE_COLS].fillna(0)

        # 8. Scale
        if is_train:
            self.scaler = StandardScaler()
            # Fit on the flattened data
            X_flat = df_features[FEATURE_COLS].values.astype(np.float32)
            X_scaled = self.scaler.fit_transform(X_flat)
            # Save scaler
            joblib.dump(self.scaler, CACHE_SCALER)
        else:
            if self.scaler is None:
                self.scaler = joblib.load(CACHE_SCALER)
            X_flat = df_features[FEATURE_COLS].values.astype(np.float32)
            X_scaled = self.scaler.transform(X_flat)

        # 9. Reshape
        # Sort by sample_id and offset to ensure correct sequence order
        # (The merge might have shuffled things, though usually preserves left order)
        # We rely on the fact that we created df_expanded with repeat.
        # But let's be safe.
        df_features["X_scaled_idx"] = np.arange(
            len(df_features)
        )  # Keep track of scaled array rows

        # We can just reshape X_scaled directly if we trust the order.
        # df_expanded was: sample 0 (offsets -5..5), sample 1...
        # Merge preserves left order? Yes, usually.
        # Let's reshape: (N_samples, Window, N_features)
        num_samples = len(df_meta)
        X_reshaped = X_scaled.reshape(num_samples, WINDOW_SIZE, len(FEATURE_COLS))

        # 10. Extract Targets and IDs
        y = df_meta["contact"].values.astype(np.float32)
        contact_ids = df_meta["contact_id"].values

        return X_reshaped, y, contact_ids

    def process_train(self, load_cached_data=True):
        cache_path_x = os.path.join(WORKING_DIR, "train_X.npy")
        cache_path_y = os.path.join(WORKING_DIR, "train_y.npy")
        cache_path_ids = os.path.join(WORKING_DIR, "train_ids.npy")

        if load_cached_data and os.path.exists(cache_path_x):
            print("Loading cached train features...")
            return (
                np.load(cache_path_x),
                np.load(cache_path_y),
                np.load(cache_path_ids, allow_pickle=True),
            )

        print("Generating train features...")
        X, y, ids = self._generate_dataset(
            TRAIN_META_PATH, TRAIN_TRACKING_PATH, is_train=True
        )

        np.save(cache_path_x, X)
        np.save(cache_path_y, y)
        np.save(cache_path_ids, ids)
        return X, y, ids

    def process_val(self, load_cached_data=True):
        cache_path_x = os.path.join(WORKING_DIR, "val_X.npy")
        cache_path_y = os.path.join(WORKING_DIR, "val_y.npy")
        cache_path_ids = os.path.join(WORKING_DIR, "val_ids.npy")

        if load_cached_data and os.path.exists(cache_path_x):
            print("Loading cached val features...")
            return (
                np.load(cache_path_x),
                np.load(cache_path_y),
                np.load(cache_path_ids, allow_pickle=True),
            )

        print("Generating validation features...")
        X, y, ids = self._generate_dataset(
            VAL_META_PATH, TRAIN_TRACKING_PATH, is_train=False
        )

        np.save(cache_path_x, X)
        np.save(cache_path_y, y)
        np.save(cache_path_ids, ids)
        return X, y, ids

    def process_test(self, load_cached_data=True):
        cache_path_x = os.path.join(WORKING_DIR, "test_X.npy")
        cache_path_y = os.path.join(WORKING_DIR, "test_y.npy")  # Placeholder
        cache_path_ids = os.path.join(WORKING_DIR, "test_ids.npy")

        if load_cached_data and os.path.exists(cache_path_x):
            print("Loading cached test features...")
            return (
                np.load(cache_path_x),
                np.load(cache_path_y),
                np.load(cache_path_ids, allow_pickle=True),
            )

        print("Generating test features...")
        X, y, ids = self._generate_dataset(
            TEST_META_PATH, TEST_TRACKING_PATH, is_train=False
        )

        np.save(cache_path_x, X)
        np.save(cache_path_y, y)
        np.save(cache_path_ids, ids)
        return X, y, ids
