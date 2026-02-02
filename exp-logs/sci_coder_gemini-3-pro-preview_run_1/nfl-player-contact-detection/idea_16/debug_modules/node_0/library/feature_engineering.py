import pandas as pd
import numpy as np
import os
from scipy.spatial import cKDTree
from library.config import PathConfig, FeatureConfig, ModelConfig
from library.utils import CacheManager


class FeatureEngineer:
    """
    Implements the Physics-Enhanced Feature Engineering pipeline for the DSP-EME strategy.
    Handles tracking data preprocessing, windowing, spatial density calculation,
    and geometric gating.
    """

    def __init__(self):
        self.config = FeatureConfig()
        self.paths = PathConfig()
        self.cache = CacheManager()

    def _load_tracking(self, path):
        """Loads tracking data with optimized data types."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Tracking data not found at {path}")

        df = pd.read_csv(path)
        # Optimize types to save memory
        float_cols = [c for c in df.columns if df[c].dtype == "float64"]
        df[float_cols] = df[float_cols].astype("float32")
        int_cols = [c for c in df.columns if df[c].dtype == "int64"]
        # Be careful with IDs, but step/jersey can be smaller
        df["step"] = df["step"].astype("int32")
        return df

    def _compute_derivatives(self, df):
        """Computes Jerk and Angular Jerk."""
        # Ensure sorted
        df = df.sort_values(by=["game_play", "nfl_player_id", "step"])

        # Group mask to prevent bleeding between plays/players
        # We can use shift logic. If game_play or player_id changes, result is NaN (filled with 0)

        # Jerk: d(Acceleration)/dt. dt=0.1s.
        # We can just take difference * 10.
        acc_diff = df.groupby(["game_play", "nfl_player_id"])["acceleration"].diff()
        df["jerk"] = (acc_diff / 0.1).fillna(0).astype("float32")

        # Angular Jerk: d(Orientation)/dt. Handle circular wrap.
        # Orientation is 0-360.
        # diff = (curr - prev + 180) % 360 - 180
        curr_orient = df["orientation"]
        prev_orient = df.groupby(["game_play", "nfl_player_id"])["orientation"].shift(1)

        diff = (curr_orient - prev_orient + 180) % 360 - 180
        df["angular_jerk"] = (diff / 0.1).fillna(0).astype("float32")

        return df

    def _compute_spatial_density(self, df):
        """
        Calculates the number of neighbors within 2 yards for each player at each step.
        Uses cKDTree for efficiency.
        """
        # We need to iterate over frames (game_play, step)
        # To speed up, we can group.

        # Initialize result column
        df["spatial_density"] = 0.0

        # Get unique frames
        # This loop can be slow. Optimization:
        # Process only unique frames present in the data.
        # Given the constraints, we'll try a relatively efficient loop.

        # Extract coordinates and indices
        # We map back results using index

        # Grouping by game_play and step
        groups = df.groupby(["game_play", "step"])

        densities = {}  # index -> density

        # This might still be slow for 1.2M rows (approx 60k groups).
        # But it's done once per tracking file.

        for name, group in groups:
            coords = group[["x_position", "y_position"]].values
            if len(coords) > 1:
                tree = cKDTree(coords)
                # query_ball_point returns indices of neighbors within r
                # count_neighbors returns count.
                # We want count of OTHER players. query_ball_point includes self.
                # k=100 (max neighbors)
                # Simplest: count neighbors within 2.0
                counts = tree.query_ball_point(coords, r=2.0, return_length=True)
                # Subtract 1 for self
                counts = np.array(counts) - 1

                # Assign back
                for idx, count in zip(group.index, counts):
                    densities[idx] = count
            else:
                for idx in group.index:
                    densities[idx] = 0

        # Map back
        # Creating a series from dict is faster than loc in loop
        s_density = pd.Series(densities)
        df.loc[s_density.index, "spatial_density"] = s_density.astype("float32")

        return df

    def _create_windowed_features(self, df):
        """
        Generates lag/lead features for +/- WINDOW_SIZE steps.
        """
        window_size = self.config.WINDOW_SIZE
        feature_cols = self.config.TRACKING_COLS + [
            "jerk",
            "angular_jerk",
            "spatial_density",
        ]

        # Sort is guaranteed from previous steps
        grouper = df.groupby(["game_play", "nfl_player_id"])

        # We want to create columns like x_position_lag_1, x_position_lead_1...
        # But we need to keep memory in check.
        # We will pivot or just concat shifted series.

        new_cols = {}

        # Lags (Past)
        for i in range(1, window_size + 1):
            shifted = grouper[feature_cols].shift(i)
            for col in feature_cols:
                new_cols[f"{col}_lag{i}"] = shifted[col]

        # Leads (Future)
        for i in range(1, window_size + 1):
            shifted = grouper[feature_cols].shift(-i)
            for col in feature_cols:
                new_cols[f"{col}_lead{i}"] = shifted[col]

        # Concatenate all at once to avoid fragmentation
        df_windows = pd.DataFrame(new_cols, index=df.index)

        # Handle NaNs (edges of play) with Sentinel?
        # Or fill with 0/nearest?
        # For physics, 0 might be misleading.
        # Let's fill with Sentinel for now, or ffill/bfill limited.
        # Given continuous play, ffill/bfill is reasonable for small gaps,
        # but boundaries are real.
        # We will fill with SENTINEL_VALUE.
        df_windows = df_windows.fillna(self.config.SENTINEL_VALUE)

        # Cast to float32
        df_windows = df_windows.astype("float32")

        # Join back
        df = pd.concat([df, df_windows], axis=1)

        return df

    def _preprocess_tracking_pipeline(self, tracking_df):
        """Executes the full preprocessing pipeline on tracking data."""
        print("  Computing derivatives...")
        tracking_df = self._compute_derivatives(tracking_df)

        print("  Computing spatial density...")
        tracking_df = self._compute_spatial_density(tracking_df)

        print("  Generating windowed features...")
        tracking_df = self._create_windowed_features(tracking_df)

        return tracking_df

    def _merge_and_engineer(self, metadata_df, tracking_df, is_train=True):
        """
        Merges metadata with processed tracking and computes interaction physics.
        """
        print("  Merging Player 1 data...")
        # Merge P1
        # Ensure types match
        metadata_df["game_play"] = metadata_df["game_play"].astype(str)
        metadata_df["nfl_player_id_1"] = metadata_df["nfl_player_id_1"].astype(int)

        # Tracking needs to be mergeable
        # We select all columns from tracking
        track_cols = tracking_df.columns.tolist()
        # Exclude key columns from rename list to avoid confusion, but we merge on them

        # Merge P1
        df = pd.merge(
            metadata_df,
            tracking_df,
            left_on=["game_play", "step", "nfl_player_id_1"],
            right_on=["game_play", "step", "nfl_player_id"],
            how="left",
            suffixes=("", "_p1_dup"),
        )
        # Rename tracking columns to _p1
        rename_p1 = {
            c: f"{c}_p1"
            for c in track_cols
            if c not in ["game_play", "step", "nfl_player_id"]
        }
        df = df.rename(columns=rename_p1)
        df = df.drop(columns=["nfl_player_id", "nfl_player_id_p1_dup"], errors="ignore")

        print("  Merging Player 2 data...")
        # Handle Ground: Create is_ground flag
        df[self.config.GROUND_FEATURE] = (df["nfl_player_id_2"] == "G").astype(int)

        # Create a temporary ID column for merging P2, replacing 'G' with a dummy or handling it
        # If 'G', we won't match anything in tracking (since tracking has ints).
        # We can convert nfl_player_id_2 to numeric, coercing 'G' to NaN.
        df["p2_merge_id"] = pd.to_numeric(df["nfl_player_id_2"], errors="coerce")

        # Merge P2
        df = pd.merge(
            df,
            tracking_df,
            left_on=["game_play", "step", "p2_merge_id"],
            right_on=["game_play", "step", "nfl_player_id"],
            how="left",
            suffixes=("", "_p2_dup"),
        )

        # Rename tracking columns to _p2
        rename_p2 = {
            c: f"{c}_p2"
            for c in track_cols
            if c not in ["game_play", "step", "nfl_player_id"]
        }
        df = df.rename(columns=rename_p2)
        df = df.drop(
            columns=["nfl_player_id", "nfl_player_id_p2_dup", "p2_merge_id"],
            errors="ignore",
        )

        # Fill P2 features for Ground with Sentinel/Zero
        # If is_ground == 1, P2 features are NaN after merge. Fill them.
        p2_cols = [c for c in df.columns if c.endswith("_p2")]
        # We fill with 0 for kinematics to make distance calc work (if we treated G as 0,0),
        # BUT G is not at 0,0. G is implicit.
        # For P-G, distance is conceptually 0 or undefined.
        # We will fill NaNs with SENTINEL_VALUE.
        df[p2_cols] = df[p2_cols].fillna(self.config.SENTINEL_VALUE)

        print("  Computing Interaction Physics...")
        # 1. Distance (Euclidean)
        # If Ground, Distance = 0 (Conceptually contact is immediate).
        # However, for gating, we keep all ground.
        # Let's calculate distance normally. If P2 is Sentinel (Ground), distance will be garbage.
        # We fix distance for Ground rows to 0.

        dx = df["x_position_p1"] - df["x_position_p2"]
        dy = df["y_position_p1"] - df["y_position_p2"]
        dist = np.sqrt(dx**2 + dy**2)

        # Overwrite distance for Ground
        is_g = df[self.config.GROUND_FEATURE] == 1
        dist[is_g] = 0.0
        df["distance"] = dist.astype("float32")

        # 2. Time To Collision (TTC)
        # Closing Speed = - (v_rel . r_rel) / |r_rel|
        # v_rel = v1 - v2
        vx1 = df["speed_p1"] * np.sin(
            np.radians(df["direction_p1"])
        )  # Approx component
        vy1 = df["speed_p1"] * np.cos(np.radians(df["direction_p1"]))
        vx2 = df["speed_p2"] * np.sin(np.radians(df["direction_p2"]))
        vy2 = df["speed_p2"] * np.cos(np.radians(df["direction_p2"]))

        # Fix for Ground: P2 speed is Sentinel (-1). This messes up calc.
        # If Ground, TTC is undefined (Sentinel).

        dvx = vx1 - vx2
        dvy = vy1 - vy2

        # Dot product
        dot = dvx * dx + dvy * dy

        # TTC = - (dist^2) / dot.
        # If dot >= 0 (moving away), TTC is Sentinel.
        # If dist is 0, TTC is 0.

        ttc = np.full(len(df), self.config.SENTINEL_VALUE, dtype=np.float32)

        # Mask for valid calculation (Not Ground, Dot < 0)
        valid_mask = (~is_g) & (dot < -1e-6)

        ttc[valid_mask] = -(dist[valid_mask] ** 2) / dot[valid_mask]
        df["time_to_collision"] = ttc

        # 3. Kinetic Energy Proxy
        # Relative Speed Squared
        rel_speed_sq = dvx**2 + dvy**2
        rel_speed_sq[is_g] = self.config.SENTINEL_VALUE
        df["kinetic_energy_proxy"] = rel_speed_sq.astype("float32")

        # 4. Simple diffs
        df["speed_diff"] = (df["speed_p1"] - df["speed_p2"]).abs()
        df.loc[is_g, "speed_diff"] = self.config.SENTINEL_VALUE

        # Gating (Stage 0)
        # Discard Player-Player pairs with Distance > GATING_DISTANCE
        if is_train:
            print(
                f"  Applying Geometric Gating (Threshold: {self.config.GATING_DISTANCE} yds)..."
            )
            # Keep if (Ground) OR (Distance <= Threshold)
            mask = (is_g) | (df["distance"] <= self.config.GATING_DISTANCE)
            before = len(df)
            df = df[mask]
            print(
                f"  Gating dropped {before - len(df)} rows ({(before - len(df))/before:.1%})."
            )

        return df

    def process_train_val(self, load_cached=True):
        """
        Generates features for Training and Validation sets.
        Loads tracking once, splits metadata, merges, and saves.
        """
        # Check cache first
        if load_cached:
            train_feats = self.cache.load_parquet(self.paths.TRAIN_FEATURES_PATH)
            val_feats = self.cache.load_parquet(self.paths.VAL_FEATURES_PATH)
            if train_feats is not None and val_feats is not None:
                print("Loaded Train/Val features from cache.")
                return train_feats, val_feats

        print("Generating Train/Val features from scratch...")

        # Load Raw Data
        print("Loading Tracking Data...")
        tracking_df = self._load_tracking(self.paths.TRAIN_TRACKING_PATH)

        # Preprocess Tracking (Expensive step, done once)
        print("Preprocessing Tracking Data...")
        tracking_df = self._preprocess_tracking_pipeline(tracking_df)

        # Load Metadata
        print("Loading Metadata...")
        train_meta = pd.read_csv(self.paths.TRAIN_METADATA_PATH)
        val_meta = pd.read_csv(self.paths.VAL_METADATA_PATH)

        # Process Train
        print("Processing Training Set...")
        train_feats = self._merge_and_engineer(train_meta, tracking_df, is_train=True)
        self.cache.save_parquet(train_feats, self.paths.TRAIN_FEATURES_PATH)

        # Process Val
        print("Processing Validation Set...")
        # We apply gating to Val as well to match Train distribution for metrics,
        # though strictly we should validate on full.
        # However, for this competition, non-contact is huge.
        # We usually validate on the same distribution we care about, or full.
        # Given the prompt "Discard... to focus compute", we likely want to validate
        # on the difficult cases.
        val_feats = self._merge_and_engineer(val_meta, tracking_df, is_train=True)
        self.cache.save_parquet(val_feats, self.paths.VAL_FEATURES_PATH)

        return train_feats, val_feats

    def process_test(self, load_cached=True):
        """
        Generates features for the Test set.
        """
        if load_cached:
            test_feats = self.cache.load_parquet(self.paths.TEST_FEATURES_PATH)
            if test_feats is not None:
                print("Loaded Test features from cache.")
                return test_feats

        print("Generating Test features from scratch...")

        # Load Raw Data
        tracking_df = self._load_tracking(self.paths.TEST_TRACKING_PATH)

        # Preprocess
        print("Preprocessing Test Tracking...")
        tracking_df = self._preprocess_tracking_pipeline(tracking_df)

        # Load Metadata
        test_meta = pd.read_csv(self.paths.TEST_METADATA_PATH)

        # Process
        print("Processing Test Set...")
        # Do NOT gate test set. We need predictions for all rows.
        test_feats = self._merge_and_engineer(test_meta, tracking_df, is_train=False)

        self.cache.save_parquet(test_feats, self.paths.TEST_FEATURES_PATH)
        return test_feats
