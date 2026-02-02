import os
import pandas as pd
import numpy as np
import gc
from library.config import (
    TRACKING_PATH_TRAIN,
    TRACKING_PATH_TEST,
    TRAIN_META_PATH,
    VAL_META_PATH,
    TEST_META_PATH,
    WORKING_DIR,
    STREAM_A_FEATURES,
    STREAM_B_FEATURES,
    WINDOW_SIZE,
    SAMPLING_RATIO,
    SEED,
)
from library.utils import get_config_hash, seed_everything


class FeatureFactory:
    def __init__(self):
        self.tracking_train = None
        self.tracking_test = None
        self.window_half = WINDOW_SIZE // 2
        # Define offsets: e.g., -4, -3, ..., 0, ..., 4
        self.offsets = list(range(-self.window_half, self.window_half + 1))

    def _load_tracking(self, split):
        """
        Loads and preprocesses tracking data for the specific split.
        Computes derivatives and trig components, then creates a wide-format
        representation with temporal lags.
        """
        path = TRACKING_PATH_TEST if split == "test" else TRACKING_PATH_TRAIN

        # Load only necessary columns to save memory
        cols = [
            "game_play",
            "nfl_player_id",
            "step",
            "x_position",
            "y_position",
            "speed",
            "acceleration",
            "sa",
            "orientation",
            "direction",
        ]
        df = pd.read_csv(path, usecols=cols)

        # Sort for correct shifting
        df = df.sort_values(["game_play", "nfl_player_id", "step"]).reset_index(
            drop=True
        )

        # --- Pre-compute Base Features ---
        # Convert angles to radians
        df["rad_orient"] = np.deg2rad(df["orientation"])
        df["rad_dir"] = np.deg2rad(df["direction"])

        # Trig components
        df["sin_orient"] = np.sin(df["rad_orient"])
        df["cos_orient"] = np.cos(df["rad_orient"])
        df["sin_dir"] = np.sin(df["rad_dir"])
        df["cos_dir"] = np.cos(df["rad_dir"])

        # Derivatives (Impact Proxies for Stream B)
        # Group by play/player to ensure we don't diff across boundaries
        # Jerk = d(Acceleration)/dt
        df["jerk"] = (
            df.groupby(["game_play", "nfl_player_id"])["acceleration"].diff().fillna(0)
        )
        # Ang Vel = d(Orientation)/dt (handling wrap-around not strictly necessary for tree models but good practice)
        # Simple diff is sufficient for trees usually, but let's be cleaner with sin/cos diffs if needed.
        # Here we stick to simple diff of the raw orientation for simplicity as per description "derivative of orientation"
        df["ang_vel"] = (
            df.groupby(["game_play", "nfl_player_id"])["orientation"].diff().fillna(0)
        )

        # Drop intermediate angle columns if not needed
        df = df.drop(columns=["rad_orient", "rad_dir"])

        # --- Create Wide Format (Temporal Windowing) ---
        # We want columns like speed_t-4, speed_t0, speed_t+4
        # We perform shifts and concatenate.

        # Base feature columns to shift
        feature_cols = [
            "x_position",
            "y_position",
            "speed",
            "acceleration",
            "sa",
            "sin_orient",
            "cos_orient",
            "sin_dir",
            "cos_dir",
            "jerk",
            "ang_vel",
        ]

        shifted_dfs = []

        # Group object for shifting
        # Note: shifting by group is slower.
        # Optimization: Since data is sorted by game_play, player, step,
        # we can just global shift and then mask boundaries.
        # However, for safety and simplicity with this data size, we use groupby shift or explicit indexing.
        # Given 1.2M rows, groupby shift is acceptable.
        grouper = df.groupby(["game_play", "nfl_player_id"])

        for offset in self.offsets:
            # Suffix: _t-4, _t0, _t+4
            suffix = (
                f"_t{offset}"
                if offset < 0
                else (f"_t+{offset}" if offset > 0 else "_t0")
            )

            # Shift
            # offset < 0 means looking back (lag), so we shift DOWN (positive integer in pandas shift)
            # offset > 0 means looking forward (lead), so we shift UP (negative integer in pandas shift)
            # Actually:
            # t-4 means data from 4 steps ago. Current row needs value from index-4.
            # df.shift(4) brings index-4 to current index.
            # So shift amount = -offset? No.
            # If we want value at t-4, we need the row that was at position i-4 to be at i.
            # df.shift(4) moves row i to i+4.
            # We want row i-4 to move to i. So df.shift(4) does exactly that (value from prev flows to curr).
            # So shift = abs(offset) if offset < 0.
            # If offset > 0 (t+4), we want row i+4 to be at i. df.shift(-4).

            shift_amount = offset  # e.g. -4
            # Wait, pandas shift(1) takes previous value.
            # We want t-4 (past). So we want shift(4).
            # We want t+4 (future). So we want shift(-4).
            # So pandas_shift = -offset.

            # Correction:
            # Logic: We want the column `speed_t-4` to contain the speed from 4 steps ago.
            # If current step is 10, we want speed from step 6.
            # `df['speed'].shift(4)` puts value from idx 6 at idx 10. Correct.
            # So we shift by `abs(offset)` if offset is negative (past).
            # If offset is positive (future), e.g., t+4, we want value from step 14 at step 10.
            # `df['speed'].shift(-4)` puts value from idx 14 at idx 10. Correct.

            # So shift_val = -offset
            shift_val = -offset  # Wait.
            # t-4 (past): offset=-4. We want shift(4). -> -(-4) = 4. Correct.
            # t+4 (future): offset=4. We want shift(-4). -> -(4) = -4. Correct.

            # Apply shift
            d_shifted = grouper[feature_cols].shift(shift_val)
            d_shifted.columns = [f"{c}{suffix}" for c in feature_cols]
            shifted_dfs.append(d_shifted)

        # Concatenate all shifted features
        df_wide = pd.concat(
            [df[["game_play", "nfl_player_id", "step"]]] + shifted_dfs, axis=1
        )

        # Clean up
        del df, shifted_dfs
        gc.collect()

        return df_wide

    def _engineer_stream_a(self, df_labels, df_wide, is_train=False):
        """
        Engineers features for Player-Player interactions.
        Merges P1 and P2 wide features and computes relative dynamics.
        """
        # Filter for non-ground contacts
        df = df_labels[df_labels["nfl_player_id_2"] != "G"].copy()

        # Ensure IDs are correct types for merging
        df["nfl_player_id_1"] = pd.to_numeric(df["nfl_player_id_1"], errors="coerce")
        df["nfl_player_id_2"] = pd.to_numeric(df["nfl_player_id_2"], errors="coerce")

        # Merge P1
        df = df.merge(
            df_wide.add_suffix("_p1"),
            left_on=["game_play", "step", "nfl_player_id_1"],
            right_on=["game_play_p1", "step_p1", "nfl_player_id_p1"],
            how="inner",
        )

        # Merge P2
        df = df.merge(
            df_wide.add_suffix("_p2"),
            left_on=["game_play", "step", "nfl_player_id_2"],
            right_on=["game_play_p2", "step_p2", "nfl_player_id_p2"],
            how="inner",
        )

        # Compute Interaction Features for each offset
        for offset in self.offsets:
            suffix = (
                f"_t{offset}"
                if offset < 0
                else (f"_t+{offset}" if offset > 0 else "_t0")
            )

            # Feature names
            x1, y1 = f"x_position_p1{suffix}", f"y_position_p1{suffix}"
            x2, y2 = f"x_position_p2{suffix}", f"y_position_p2{suffix}"
            s1, s2 = f"speed_p1{suffix}", f"speed_p2{suffix}"
            a1, a2 = f"acceleration_p1{suffix}", f"acceleration_p2{suffix}"
            sd1, cd1 = f"sin_dir_p1{suffix}", f"cos_dir_p1{suffix}"
            sd2, cd2 = f"sin_dir_p2{suffix}", f"cos_dir_p2{suffix}"

            # 1. Distance
            dist_col = f"distance{suffix}"
            df[dist_col] = np.sqrt((df[x1] - df[x2]) ** 2 + (df[y1] - df[y2]) ** 2)

            # 2. Speed Diff
            df[f"speed_diff{suffix}"] = np.abs(df[s1] - df[s2])

            # 3. Accel Diff
            df[f"accel_diff{suffix}"] = np.abs(df[a1] - df[a2])

            # 4. Cosine Similarity of Direction
            # Dot product of unit vectors
            df[f"cos_sim_dir{suffix}"] = (df[sd1] * df[sd2]) + (df[cd1] * df[cd2])

            # 5. Closure Rate
            # (v1 - v2) dot (r12) / |r12|
            # r12 = p2 - p1
            rx = df[x2] - df[x1]
            ry = df[y2] - df[y1]
            dist = df[dist_col]

            # Velocity vectors
            vx1 = df[s1] * df[cd1]
            vy1 = df[s1] * df[sd1]
            vx2 = df[s2] * df[cd2]
            vy2 = df[s2] * df[sd2]

            # Relative velocity (closing speed is positive if moving towards each other)
            # Standard definition: - d(dist)/dt.
            # Using vector projection: (v1 - v2) . (p2 - p1) / dist
            # If v1 moves to p2, v1.(p2-p1) > 0. If v2 moves to p1, -v2.(p2-p1) > 0.
            # So (v1 - v2) . (p2 - p1) / dist

            v_rel_x = vx1 - vx2
            v_rel_y = vy1 - vy2

            dot_prod = (v_rel_x * rx) + (v_rel_y * ry)

            # Handle division by zero (distance=0)
            df[f"closure_rate{suffix}"] = np.where(dist > 1e-4, dot_prod / dist, 0.0)

        # Undersampling (Train only)
        if is_train:
            pos_mask = df["contact"] == 1
            neg_mask = df["contact"] == 0

            n_pos = pos_mask.sum()
            n_neg_keep = int(n_pos * SAMPLING_RATIO)

            if n_pos > 0:
                neg_indices = (
                    df[neg_mask]
                    .sample(n=min(n_neg_keep, neg_mask.sum()), random_state=SEED)
                    .index
                )
                keep_indices = df[pos_mask].index.union(neg_indices)
                df = (
                    df.loc[keep_indices]
                    .sample(frac=1.0, random_state=SEED)
                    .reset_index(drop=True)
                )

        # Select Features
        X = df[STREAM_A_FEATURES].copy()
        y = df["contact"].values
        ids = df["contact_id"].values

        return X, y, ids

    def _engineer_stream_b(self, df_labels, df_wide, is_train=False):
        """
        Engineers features for Player-Ground interactions.
        Uses only P1 ego-motion and impact proxies.
        """
        # Filter for ground contacts
        df = df_labels[df_labels["nfl_player_id_2"] == "G"].copy()

        # Ensure IDs are correct types
        df["nfl_player_id_1"] = pd.to_numeric(df["nfl_player_id_1"], errors="coerce")

        # Merge P1
        df = df.merge(
            df_wide.add_suffix("_p1"),
            left_on=["game_play", "step", "nfl_player_id_1"],
            right_on=["game_play_p1", "step_p1", "nfl_player_id_p1"],
            how="inner",
        )

        # Rename columns to match STREAM_B_FEATURES expectations
        # STREAM_B_FEATURES expects things like 'jerk_p1_t0', 'x_position_p1_t0'
        # The merge created these correctly (e.g. jerk_t0 -> jerk_t0_p1).
        # Wait, merge with suffix '_p1' turns 'jerk_t0' into 'jerk_t0_p1'.
        # My config expects 'jerk_p1_t0'.
        # I need to align the naming convention.

        # Current columns in df: {feat}_t{offset}_p1
        # Target columns: {feat}_p1_t{offset}

        # Let's map them.
        rename_map = {}
        for col in df.columns:
            if "_p1" in col and "_t" in col:
                # Format: feat_tX_p1 -> feat_p1_tX
                parts = col.split("_")
                # Find the 't' part and 'p1' part
                # This is tricky with variable underscores.
                # Regex is safer, or structural logic.
                # Structure from _load_tracking: {base_feat}_t{offset}
                # Structure after merge: {base_feat}_t{offset}_p1

                # We want: {base_feat}_p1_t{offset}

                # Extract suffix _p1 (last 3 chars)
                base_and_time = col[:-3]  # remove _p1

                # Extract time suffix (last part starting with _t)
                # Find last underscore
                last_us = base_and_time.rfind("_t")
                if last_us != -1:
                    base_feat = base_and_time[:last_us]
                    time_suffix = base_and_time[last_us:]  # _t0 or _t-4

                    new_name = f"{base_feat}_p1{time_suffix}"
                    rename_map[col] = new_name

        df = df.rename(columns=rename_map)

        # Do the same for Stream A?
        # In Stream A: merge suffix is _p1.
        # Columns in df_wide: speed_t0.
        # Merged: speed_t0_p1.
        # Config expects: speed_p1_t0.
        # YES, I need to fix naming in Stream A as well.

        # Undersampling (Train only)
        if is_train:
            pos_mask = df["contact"] == 1
            neg_mask = df["contact"] == 0

            n_pos = pos_mask.sum()
            n_neg_keep = int(n_pos * SAMPLING_RATIO)

            if n_pos > 0:
                neg_indices = (
                    df[neg_mask]
                    .sample(n=min(n_neg_keep, neg_mask.sum()), random_state=SEED)
                    .index
                )
                keep_indices = df[pos_mask].index.union(neg_indices)
                df = (
                    df.loc[keep_indices]
                    .sample(frac=1.0, random_state=SEED)
                    .reset_index(drop=True)
                )

        # Select Features
        # Ensure all required features exist
        missing_cols = [c for c in STREAM_B_FEATURES if c not in df.columns]
        if missing_cols:
            # This might happen if naming logic failed.
            # Fallback or error.
            pass

        X = df[STREAM_B_FEATURES].copy()
        y = df["contact"].values
        ids = df["contact_id"].values

        return X, y, ids

    def _fix_column_names(self, df, suffix_key):
        """
        Helper to fix column names from {feat}_t{k}_{suffix} to {feat}_{suffix}_t{k}
        """
        rename_map = {}
        for col in df.columns:
            if suffix_key in col and "_t" in col:
                # Assumes format ..._t{k}_{suffix_key}
                # Remove suffix key
                temp = col.replace(suffix_key, "")  # leaves ..._t{k}
                # Find time part
                last_us = temp.rfind("_t")
                if last_us != -1:
                    base = temp[:last_us]
                    time = temp[last_us:]
                    # Construct new: base + suffix_key + time
                    new_name = f"{base}{suffix_key}{time}"
                    rename_map[col] = new_name
        return df.rename(columns=rename_map)

    def process_data(self, split="train", load_cached_data=True):
        """
        Main pipeline to generate features.
        Returns a dictionary with stream_a and stream_b data.
        """
        seed_everything(SEED)

        # Generate Cache Paths
        config_hash = get_config_hash()
        cache_prefix = os.path.join(WORKING_DIR, f"{split}_{config_hash}")

        paths = {
            "A_X": f"{cache_prefix}_streamA_X.parquet",
            "A_y": f"{cache_prefix}_streamA_y.npy",
            "A_ids": f"{cache_prefix}_streamA_ids.npy",
            "B_X": f"{cache_prefix}_streamB_X.parquet",
            "B_y": f"{cache_prefix}_streamB_y.npy",
            "B_ids": f"{cache_prefix}_streamB_ids.npy",
        }

        # Check Cache
        all_exist = all(os.path.exists(p) for p in paths.values())

        if load_cached_data and all_exist:
            print(f"Loading cached features for {split} from {WORKING_DIR}...")
            return {
                "stream_a": {
                    "X": pd.read_parquet(paths["A_X"]),
                    "y": np.load(paths["A_y"]),
                    "ids": np.load(paths["A_ids"], allow_pickle=True),
                },
                "stream_b": {
                    "X": pd.read_parquet(paths["B_X"]),
                    "y": np.load(paths["B_y"]),
                    "ids": np.load(paths["B_ids"], allow_pickle=True),
                },
            }

        print(f"Generating features for {split}...")

        # 1. Load Metadata
        if split == "train":
            df_meta = pd.read_csv(TRAIN_META_PATH)
        elif split == "validation":
            df_meta = pd.read_csv(VAL_META_PATH)
        else:
            df_meta = pd.read_csv(TEST_META_PATH)

        # 2. Load and Window Tracking Data
        # For validation, we use train tracking data but filter by plays in validation set?
        # Tracking files are split by train/test. Validation is a subset of train plays.
        track_split = "test" if split == "test" else "train"
        df_wide = self._load_tracking(track_split)

        # Filter tracking data to relevant plays to save memory during merge
        relevant_plays = df_meta["game_play"].unique()
        df_wide = df_wide[df_wide["game_play"].isin(relevant_plays)].copy()

        # 3. Process Stream A
        # Need to fix column names after merge inside the function
        # We override the _engineer_stream_a logic slightly to use the helper

        # --- Stream A ---
        print("Processing Stream A...")
        # Filter labels
        df_labels_a = df_meta[df_meta["nfl_player_id_2"] != "G"].copy()

        # Ensure IDs are numeric for merging
        df_labels_a["nfl_player_id_1"] = pd.to_numeric(
            df_labels_a["nfl_player_id_1"], errors="coerce"
        )
        df_labels_a["nfl_player_id_2"] = pd.to_numeric(
            df_labels_a["nfl_player_id_2"], errors="coerce"
        )

        # Merge P1
        df_a = df_labels_a.merge(
            df_wide.add_suffix("_p1"),
            left_on=["game_play", "step", "nfl_player_id_1"],
            right_on=["game_play_p1", "step_p1", "nfl_player_id_p1"],
            how="inner",
        )
        # Fix P1 names
        df_a = self._fix_column_names(df_a, "_p1")

        # Merge P2
        df_a = df_a.merge(
            df_wide.add_suffix("_p2"),
            left_on=["game_play", "step", "nfl_player_id_2"],
            right_on=["game_play_p2", "step_p2", "nfl_player_id_p2"],
            how="inner",
        )
        # Fix P2 names
        df_a = self._fix_column_names(df_a, "_p2")

        # Compute Interaction Features
        # Note: Now columns are named like x_position_p1_t0, which matches config
        for offset in self.offsets:
            suffix = (
                f"_t{offset}"
                if offset < 0
                else (f"_t+{offset}" if offset > 0 else "_t0")
            )

            x1, y1 = f"x_position_p1{suffix}", f"y_position_p1{suffix}"
            x2, y2 = f"x_position_p2{suffix}", f"y_position_p2{suffix}"
            s1, s2 = f"speed_p1{suffix}", f"speed_p2{suffix}"
            a1, a2 = f"acceleration_p1{suffix}", f"acceleration_p2{suffix}"
            sd1, cd1 = f"sin_dir_p1{suffix}", f"cos_dir_p1{suffix}"
            sd2, cd2 = f"sin_dir_p2{suffix}", f"cos_dir_p2{suffix}"

            # Distance
            dist_col = f"distance{suffix}"
            df_a[dist_col] = np.sqrt(
                (df_a[x1] - df_a[x2]) ** 2 + (df_a[y1] - df_a[y2]) ** 2
            )

            # Speed/Accel Diff
            df_a[f"speed_diff{suffix}"] = np.abs(df_a[s1] - df_a[s2])
            df_a[f"accel_diff{suffix}"] = np.abs(df_a[a1] - df_a[a2])

            # Cos Sim
            df_a[f"cos_sim_dir{suffix}"] = (df_a[sd1] * df_a[sd2]) + (
                df_a[cd1] * df_a[cd2]
            )

            # Closure Rate
            rx = df_a[x2] - df_a[x1]
            ry = df_a[y2] - df_a[y1]
            dist = df_a[dist_col]
            vx1, vy1 = df_a[s1] * df_a[cd1], df_a[s1] * df_a[sd1]
            vx2, vy2 = df_a[s2] * df_a[cd2], df_a[s2] * df_a[sd2]
            v_rel_x, v_rel_y = vx1 - vx2, vy1 - vy2
            dot_prod = (v_rel_x * rx) + (v_rel_y * ry)
            df_a[f"closure_rate{suffix}"] = np.where(dist > 1e-4, dot_prod / dist, 0.0)

        # Undersample A
        if split == "train":
            pos_mask = df_a["contact"] == 1
            neg_mask = df_a["contact"] == 0
            n_pos = pos_mask.sum()
            n_neg_keep = int(n_pos * SAMPLING_RATIO)
            if n_pos > 0:
                neg_indices = (
                    df_a[neg_mask]
                    .sample(n=min(n_neg_keep, neg_mask.sum()), random_state=SEED)
                    .index
                )
                keep_indices = df_a[pos_mask].index.union(neg_indices)
                df_a = (
                    df_a.loc[keep_indices]
                    .sample(frac=1.0, random_state=SEED)
                    .reset_index(drop=True)
                )

        X_a = df_a[STREAM_A_FEATURES].copy()
        y_a = df_a["contact"].values
        ids_a = df_a["contact_id"].values

        del df_a, df_labels_a
        gc.collect()

        # --- Stream B ---
        print("Processing Stream B...")
        df_labels_b = df_meta[df_meta["nfl_player_id_2"] == "G"].copy()

        # Ensure IDs are numeric for merging
        df_labels_b["nfl_player_id_1"] = pd.to_numeric(
            df_labels_b["nfl_player_id_1"], errors="coerce"
        )

        # Merge P1
        df_b = df_labels_b.merge(
            df_wide.add_suffix("_p1"),
            left_on=["game_play", "step", "nfl_player_id_1"],
            right_on=["game_play_p1", "step_p1", "nfl_player_id_p1"],
            how="inner",
        )
        # Fix P1 names
        df_b = self._fix_column_names(df_b, "_p1")

        # Undersample B
        if split == "train":
            pos_mask = df_b["contact"] == 1
            neg_mask = df_b["contact"] == 0
            n_pos = pos_mask.sum()
            n_neg_keep = int(n_pos * SAMPLING_RATIO)
            if n_pos > 0:
                neg_indices = (
                    df_b[neg_mask]
                    .sample(n=min(n_neg_keep, neg_mask.sum()), random_state=SEED)
                    .index
                )
                keep_indices = df_b[pos_mask].index.union(neg_indices)
                df_b = (
                    df_b.loc[keep_indices]
                    .sample(frac=1.0, random_state=SEED)
                    .reset_index(drop=True)
                )

        X_b = df_b[STREAM_B_FEATURES].copy()
        y_b = df_b["contact"].values
        ids_b = df_b["contact_id"].values

        del df_b, df_labels_b, df_wide
        gc.collect()

        # Save Cache
        print("Saving cache...")
        X_a.to_parquet(paths["A_X"])
        np.save(paths["A_y"], y_a)
        np.save(paths["A_ids"], ids_a)

        X_b.to_parquet(paths["B_X"])
        np.save(paths["B_y"], y_b)
        np.save(paths["B_ids"], ids_b)

        return {
            "stream_a": {"X": X_a, "y": y_a, "ids": ids_a},
            "stream_b": {"X": X_b, "y": y_b, "ids": ids_b},
        }
