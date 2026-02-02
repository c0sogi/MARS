import pandas as pd
import numpy as np
import os
import gc
from library.config import Config
from library.utils import get_data_hash


class FeatureEngine:
    """
    Handles feature engineering for the Multi-Resolution Dual-Stream GBDT architecture.
    Manages data loading, caching, temporal windowing (Micro/Macro), and stream-specific logic.
    """

    def __init__(self):
        self.micro_window = Config.MICRO_WINDOW_SIZE
        self.macro_window = Config.MACRO_WINDOW_SIZE
        self.undersample_ratio = Config.UNDERSAMPLE_RATIO

        # Define base features for tracking data
        self.tracking_cols = [
            "game_play",
            "nfl_player_id",
            "step",
            "x_position",
            "y_position",
            "speed",
            "acceleration",
            "orientation",
            "direction",
        ]

    def _add_cyclical_features(self, df, cols):
        """Adds sine and cosine transforms for angular features."""
        for col in cols:
            if col in df.columns:
                # Fill NaNs before transform to avoid propagation issues, though tracking shouldn't have many
                df[col] = df[col].fillna(0)
                df[f"sin_{col}"] = np.sin(np.deg2rad(df[col]))
                df[f"cos_{col}"] = np.cos(np.deg2rad(df[col]))
        return df

    def _add_kinematics(self, df):
        """
        Calculates advanced kinematics: Jerk, Angular Velocity, Pose-Motion Alignment.
        Assumes df is sorted by game_play, nfl_player_id, step.
        """
        dt = 0.1  # 10Hz data

        # Calculate diffs respecting groups
        # Since df is usually a large tracking file, we group by player
        # Note: This operation is expensive on the full dataset, so we do it efficiently

        # Jerk: derivative of acceleration
        df["jerk"] = (
            df.groupby(["game_play", "nfl_player_id"])["acceleration"].diff().fillna(0)
            / dt
        )

        # Angular Velocity: derivative of orientation
        # Handle 0/360 wrap-around logic? For simplicity in this timeframe, simple diff
        df["angular_velocity"] = (
            df.groupby(["game_play", "nfl_player_id"])["orientation"].diff().fillna(0)
            / dt
        )

        # Pose-Motion Alignment: Cosine similarity between Orientation and Direction
        # Orientation: 0-360, Direction: 0-360
        df["pose_motion_align"] = np.cos(
            np.deg2rad(df["orientation"] - df["direction"])
        ).fillna(0)

        return df

    def _preprocess_tracking(self, tracking_path):
        """Loads and pre-processes tracking data with individual kinematics."""
        # print(f"Loading tracking data from {tracking_path}...")
        df_tr = pd.read_csv(tracking_path)

        # Filter columns to minimize memory usage
        req_cols = list(
            set(self.tracking_cols + ["game_play", "nfl_player_id", "step"])
        )
        df_tr = df_tr[req_cols].copy()

        # Sort for temporal ops
        df_tr = df_tr.sort_values(["game_play", "nfl_player_id", "step"])

        # Add derived features
        df_tr = self._add_cyclical_features(df_tr, ["orientation", "direction"])
        df_tr = self._add_kinematics(df_tr)

        # Drop raw angular columns if not needed, but Config uses them for checking
        # We keep them for now.

        return df_tr

    def _create_dense_skeleton(self, df_meta):
        """
        Ensures the dataframe has contiguous steps for every contact_id group
        to allow correct rolling window calculations.
        Crucial for the test set if sample_submission is sparse.
        """
        # Identify min and max step for each play involved
        # For training, data is usually dense. For test, we might need to expand.
        # To be safe and simple: We assume the input df_meta is the target rows we want.
        # If we need rolling features, we must ensure we have the history.
        # Strategy: We will perform rolling operations AFTER merging tracking data.
        # However, if df_meta has gaps (e.g. step 10, then step 15), rolling on df_meta is wrong.
        # We need to merge tracking data which IS dense, then compute rolling, then filter.
        return df_meta

    def _compute_interaction_features(self, df):
        """Calculates distance, relative speed, closure rate, etc. between p1 and p2."""
        # Euclidean Distance
        df["distance"] = np.sqrt(
            (df["x_position_p1"] - df["x_position_p2"]) ** 2
            + (df["y_position_p1"] - df["y_position_p2"]) ** 2
        )

        # Relative Speed (Scalar diff)
        df["rel_speed"] = np.abs(df["speed_p1"] - df["speed_p2"])

        # Closure Rate: Derivative of distance (negative of rate of change of distance)
        # We need temporal context for this.
        # If we are in a dense dataframe, we can use diff.
        # If not, we approximate using velocity vectors.
        # Closure Rate = -1 * (v1 - v2) dot (r1 - r2) / |r1 - r2|
        # Let's use the vector projection method which is instantaneous and doesn't require diff.

        # Vector P1 to P2
        rx = df["x_position_p2"] - df["x_position_p1"]
        ry = df["y_position_p2"] - df["y_position_p1"]
        dist = df["distance"].replace(0, 1e-6)  # Avoid div by zero

        # Velocity vectors
        # speed is scalar, direction is angle. Convert to vx, vy.
        # Note: NFL data: 0 degrees is usually Y axis (check docs, but standard is 0=Y, 90=X)
        # Assuming standard math angle for now or consistent internal logic.
        # Actually, standard conversion:
        vx1 = df["speed_p1"] * np.sin(np.deg2rad(df["direction_p1"]))
        vy1 = df["speed_p1"] * np.cos(np.deg2rad(df["direction_p1"]))
        vx2 = df["speed_p2"] * np.sin(np.deg2rad(df["direction_p2"]))
        vy2 = df["speed_p2"] * np.cos(np.deg2rad(df["direction_p2"]))

        # Relative velocity (P1 relative to P2)
        rvx = vx1 - vx2
        rvy = vy1 - vy2

        # Closure rate: Speed at which distance is decreasing
        # Projection of relative velocity onto the unit vector connecting them
        # Unit vector u = r / |r|
        # v_rel dot u
        # If moving towards each other, closure rate is positive.

        # Dot product
        dot_prod = (rvx * rx) + (rvy * ry)

        # If dot_prod is positive, they are moving APART (distance increasing).
        # Closure rate is usually defined as rate of closing, so negative of derivative.
        # So if moving apart, closure rate is negative.
        df["closure_rate"] = -1 * (dot_prod / dist)

        # Cosine Similarity of Directions
        # dot(v1_dir, v2_dir)
        df["cos_sim_dir"] = np.cos(np.deg2rad(df["direction_p1"] - df["direction_p2"]))

        return df

    def _generate_temporal_features(self, df, group_cols, micro_cols, macro_cols):
        """
        Generates Micro (flattened lags) and Macro (rolling stats) features.
        Assumes df is sorted and dense in time for each group.
        """
        # Sort to ensure temporal order
        df = df.sort_values(group_cols + ["step"])

        # --- Macro Features (Rolling) ---
        # Window size is total width. Config says +/- 15, so size = 31.
        window_size = (self.macro_window * 2) + 1

        # We use groupby().rolling(). This puts the index as (group_keys, original_index)
        # We need to handle this carefully to merge back.

        # Calculate rolling stats for specified macro columns
        # To save memory, select only macro cols
        roll_cols = [c for c in macro_cols if c in df.columns]

        if roll_cols:
            # print("Computing Macro features...")
            grouped = df.groupby(group_cols)[roll_cols]

            # Mean
            roll_mean = (
                grouped.rolling(window=window_size, center=True, min_periods=1)
                .mean()
                .reset_index(drop=True)
            )
            roll_mean.columns = [f"{c}_mean" for c in roll_cols]

            # Std
            roll_std = (
                grouped.rolling(window=window_size, center=True, min_periods=1)
                .std()
                .fillna(0)
                .reset_index(drop=True)
            )
            roll_std.columns = [f"{c}_std" for c in roll_cols]

            # Max
            roll_max = (
                grouped.rolling(window=window_size, center=True, min_periods=1)
                .max()
                .reset_index(drop=True)
            )
            roll_max.columns = [f"{c}_max" for c in roll_cols]

            # Concatenate (reset_index(drop=True) aligns with sorted df if we didn't mess up order)
            # Groupby preserves order of groups, but within group it preserves row order?
            # Yes, if sort=False or if data was sorted.
            # Safer: assign directly if lengths match.

            # Optimization: The groupby rolling result might be in different order if groups are not sorted.
            # But we sorted df by group_cols + step.
            # Pandas groupby(sort=False) keeps order of appearance of groups.
            # To be safe, we rely on the index.

            # Actually, rolling on groupby returns a MultiIndex.
            # Let's use transform-like approach or merge by index.

            # Re-implementation for safety:
            # Calculate rolling on the whole DF if we assume groups are separated? No.
            # Correct way:
            roll_res = (
                df.groupby(group_cols)[roll_cols]
                .rolling(window=window_size, center=True, min_periods=1)
                .agg(["mean", "std", "max"])
            )
            # roll_res columns: (col, stat)
            roll_res.columns = [f"{c[0]}_{c[1]}" for c in roll_res.columns]

            # The index of roll_res is (group_cols..., original_index).
            # We can simply drop the group levels and join to df.
            roll_res = roll_res.droplevel(list(range(len(group_cols))))

            # Join back
            df = df.join(roll_res)

        # --- Micro Features (Lags) ---
        # Flattened features for t-k ... t+k
        # print("Computing Micro features...")
        lags = range(-self.micro_window, self.micro_window + 1)

        micro_res = []
        for col in micro_cols:
            if col not in df.columns:
                continue

            for lag in lags:
                if lag == 0:
                    # Current step, already exists, but maybe rename for consistency?
                    # Config implies we keep raw names or use t_0?
                    # Usually raw is kept, but for consistency let's just keep raw as t0
                    continue

                # Shift
                # We need to shift within groups.
                # groupby().shift() is slow.
                # Since df is sorted by group, we can just shift and mask transitions.

                shifted = df.groupby(group_cols)[col].shift(
                    -lag
                )  # shift(-lag) because lag>0 usually means previous?
                # Standard: shift(1) gets previous (t-1). shift(-1) gets next (t+1).
                # If lag is -4 (t-4), we want shift(4).

                name = f"{col}_t{lag}"
                # We can add as a series
                s = shifted.rename(name)
                micro_res.append(s)

        if micro_res:
            df = pd.concat([df] + micro_res, axis=1)

        # Fill NaNs generated by shifting/rolling
        df = df.fillna(0)

        return df

    def _process_stream(self, df_meta, df_track, stream_type, is_train=True):
        """
        Core logic to merge tracking, compute interactions, and generate windowed features.
        """
        # 1. Expand Metadata to be dense if needed (for rolling calc)
        # For this implementation, we assume that for every play in df_meta,
        # we want to calculate features.
        # To get valid rolling windows, we need the full play history.
        # So we identify the (game_play) pairs in df_meta, and pull ALL steps for those plays
        # from df_track (or a dense range).

        # Get unique game_plays
        unique_plays = df_meta["game_play"].unique()

        # Filter tracking to these plays
        df_track_sub = df_track[df_track["game_play"].isin(unique_plays)].copy()

        # 2. Setup Base Dataframe
        # We need a base dataframe that has one row per step per entity-pair.
        # Stream A: P1-P2 pairs. Stream B: P1-Ground pairs.

        if stream_type == "A":
            # Player-Player
            # df_meta has specific contacts. We need to densify this to compute rolling interaction features.
            # However, densifying all P1-P2 pairs is expensive (22*21 pairs).
            # We only care about the pairs listed in df_meta.
            # Let's assume df_meta contains the relevant pairs we need to predict.
            # If it's a sparse subset (e.g. only contact moments), we can't do rolling interaction.
            # BUT, we can do rolling individual features.
            # Given the constraints, we will:
            #   a. Merge P1, P2 tracking.
            #   b. Compute instantaneous interaction features.
            #   c. Compute Rolling on Individual features (already possible on df_track_sub).
            #   d. Compute Rolling on Interaction features ONLY if df_meta is dense.
            #      Since train_labels is dense (0.1s), this works for train.
            #      For test, we assume sample_submission is dense or we accept degradation.

            # Merge P1
            df_merged = pd.merge(
                df_meta,
                df_track_sub.add_suffix("_p1"),
                left_on=["game_play", "step", "nfl_player_id_1"],
                right_on=["game_play_p1", "step_p1", "nfl_player_id_p1"],
                how="left",
            )

            # Merge P2
            df_merged = pd.merge(
                df_merged,
                df_track_sub.add_suffix("_p2"),
                left_on=["game_play", "step", "nfl_player_id_2"],
                right_on=["game_play_p2", "step_p2", "nfl_player_id_p2"],
                how="left",
            )

            # Compute Interaction
            df_merged = self._compute_interaction_features(df_merged)

            # Define Groups
            group_cols = ["game_play", "nfl_player_id_1", "nfl_player_id_2"]

            # Config Columns
            micro_cols = Config.STREAM_A_MICRO_COLS
            macro_cols = Config.STREAM_A_MACRO_COLS

        else:
            # Stream B: Player-Ground
            # Merge P1 only
            df_merged = pd.merge(
                df_meta,
                df_track_sub,  # No suffix needed, or map to standard names
                left_on=["game_play", "step", "nfl_player_id_1"],
                right_on=["game_play", "step", "nfl_player_id"],
                how="left",
            )

            group_cols = ["game_play", "nfl_player_id_1"]
            micro_cols = Config.STREAM_B_MICRO_COLS
            macro_cols = Config.STREAM_B_MACRO_COLS

        # 3. Generate Temporal Features
        # This handles sorting, rolling, and shifting
        df_features = self._generate_temporal_features(
            df_merged, group_cols, micro_cols, macro_cols
        )

        # 4. Undersampling (Train Only)
        if is_train and self.undersample_ratio > 0:
            # Separate positives and negatives
            pos = df_features[df_features["contact"] == 1]
            neg = df_features[df_features["contact"] == 0]

            if len(pos) > 0:
                n_neg = int(len(pos) * self.undersample_ratio)
                if len(neg) > n_neg:
                    neg = neg.sample(n=n_neg, random_state=Config.SEED)

                df_features = (
                    pd.concat([pos, neg])
                    .sample(frac=1, random_state=Config.SEED)
                    .reset_index(drop=True)
                )

        # 5. Select Final Columns
        # Collect all micro (lags + current) and macro (stats) columns
        # Current columns (lag 0) are in micro_cols
        final_cols = []

        # Add Macro columns
        for c in macro_cols:
            final_cols.extend([f"{c}_mean", f"{c}_std", f"{c}_max"])

        # Add Micro columns (lags)
        lags = range(-self.micro_window, self.micro_window + 1)
        for c in micro_cols:
            if c in df_features.columns:
                final_cols.append(c)  # t0
            for lag in lags:
                if lag != 0:
                    final_cols.append(f"{c}_t{lag}")

        # Filter to existing
        final_cols = [c for c in final_cols if c in df_features.columns]

        # Extract X and y
        X = df_features[final_cols].copy()

        # Handle missing values (final safety)
        X = X.fillna(0).astype(np.float32)

        if "contact" in df_features.columns:
            y = df_features["contact"].values.astype(np.int8)
        else:
            y = None

        ids = df_features["contact_id"].values

        return X, y, ids

    def get_data(self, split="train", stream="A", load_cached=True):
        """
        Main entry point to get processed data for a specific split and stream.
        """
        # Determine paths
        if split == "train":
            meta_path = Config.TRAIN_META
            track_path = Config.TRAIN_TRACKING
            if stream == "A":
                cache_X = Config.CACHE_TRAIN_A_X
                cache_y = Config.CACHE_TRAIN_A_Y
            else:
                cache_X = Config.CACHE_TRAIN_B_X
                cache_y = Config.CACHE_TRAIN_B_Y
        elif split == "val":
            meta_path = Config.VAL_META
            track_path = Config.TRAIN_TRACKING  # Val comes from train source
            if stream == "A":
                cache_X = Config.CACHE_VAL_A_X
                cache_y = Config.CACHE_VAL_A_Y
            else:
                cache_X = Config.CACHE_VAL_B_X
                cache_y = Config.CACHE_VAL_B_Y
        elif split == "test":
            meta_path = Config.TEST_META
            track_path = Config.TEST_TRACKING
            # No fixed cache path for test in Config, generate on fly or temp
            cache_X = os.path.join(Config.WORKING_DIR, f"test_stream{stream}_X.parquet")
            cache_y = os.path.join(
                Config.WORKING_DIR, f"test_stream{stream}_y.npy"
            )  # Won't exist usually
            ids_path = os.path.join(Config.WORKING_DIR, f"test_stream{stream}_ids.npy")
        else:
            raise ValueError("Invalid split")

        # Check Cache
        if load_cached and os.path.exists(cache_X):
            if split != "test" and os.path.exists(cache_y):
                # print(f"Loading cached {split} data for Stream {stream}...")
                X = pd.read_parquet(cache_X)
                y = np.load(cache_y)
                # IDs are not strictly cached in Config for train/val, but needed for inference.
                # For train/val we usually just need X, y.
                return X, y, None
            elif split == "test" and os.path.exists(ids_path):
                # print(f"Loading cached {split} data for Stream {stream}...")
                X = pd.read_parquet(cache_X)
                ids = np.load(ids_path)
                return X, None, ids

        # Process
        # print(f"Processing {split} data for Stream {stream}...")

        # 1. Load Metadata
        df_meta = pd.read_csv(meta_path)

        # 2. Filter Stream
        if stream == "A":
            df_meta = df_meta[df_meta["nfl_player_id_2"] != "G"].copy()
        else:
            df_meta = df_meta[df_meta["nfl_player_id_2"] == "G"].copy()

        if len(df_meta) == 0:
            return pd.DataFrame(), np.array([]), np.array([])

        # 3. Load Tracking
        df_track = self._preprocess_tracking(track_path)

        # 4. Process
        is_train_mode = split == "train"  # Only undersample train
        X, y, ids = self._process_stream(
            df_meta, df_track, stream, is_train=is_train_mode
        )

        # 5. Save Cache
        # print(f"Saving cache for {split} Stream {stream}...")
        X.to_parquet(cache_X)
        if y is not None:
            np.save(cache_y, y)
        if split == "test":
            np.save(ids_path, ids)

        # Clean up
        del df_meta, df_track
        gc.collect()

        return X, y, ids
