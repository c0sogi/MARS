import pandas as pd
import numpy as np
import os
import gc
from library.config import Config
from library.utils import Timer


class TrackingFeatureGenerator:
    """
    Generates kinematic features from tracking data for Stream A of the ensemble.
    Implements Multi-Resolution Windowing (Micro/Macro) and Interaction Features.
    """

    def __init__(self):
        self.config = Config
        self.cache_dir = self.config.WORKING_DIR
        self.micro_window = self.config.WINDOW_MICRO
        self.macro_window = self.config.WINDOW_MACRO

    def generate_features(
        self, split: str, load_cached_data: bool = True
    ) -> pd.DataFrame:
        """
        Main entry point to generate features for a specific split.

        Args:
            split (str): One of 'train', 'validation', 'test'.
            load_cached_data (bool): Whether to try loading from cache.

        Returns:
            pd.DataFrame: Feature matrix including contact_id, labels, and features.
        """
        cache_path = os.path.join(self.cache_dir, f"{split}_tracking_features.parquet")

        # 1. Check Cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"[Tracking] Loading cached features from {cache_path}")
            return pd.read_parquet(cache_path)

        print(f"[Tracking] Generating features for {split}...")

        # 2. Load Metadata (Labels)
        if split == "train":
            meta_path = self.config.TRAIN_META_PATH
        elif split == "validation":
            meta_path = self.config.VAL_META_PATH
        elif split == "test":
            meta_path = self.config.TEST_META_PATH
        else:
            raise ValueError(f"Unknown split: {split}")

        df_meta = pd.read_csv(meta_path)

        # Optimization: Filter tracking data to only games present in this split
        relevant_plays = df_meta["game_play"].unique()

        # 3. Load and Process Tracking Data
        # Determine which tracking file to use
        if split == "test":
            tracking_path = self.config.TEST_TRACKING_PATH
        else:
            tracking_path = self.config.TRAIN_TRACKING_PATH

        with Timer("Process Tracking Data"):
            df_tracking_wide = self._process_tracking_data(
                tracking_path, relevant_plays
            )

        # 4. Merge and Compute Interactions
        with Timer("Merge & Interactions"):
            df_features = self._merge_and_compute_interactions(
                df_meta, df_tracking_wide
            )

        # 5. Save to Cache
        print(f"[Tracking] Saving features to {cache_path}")
        df_features.to_parquet(cache_path, index=False)

        # Cleanup
        del df_tracking_wide
        gc.collect()

        return df_features

    def _process_tracking_data(
        self, tracking_path: str, relevant_plays: np.array
    ) -> pd.DataFrame:
        """
        Loads raw tracking data, computes derived features, creates lags/rolling stats,
        and returns a wide-format dataframe.
        """
        # Load specific columns to save memory
        use_cols = [
            "game_play",
            "game_key",
            "play_id",
            "nfl_player_id",
            "step",
            "x_position",
            "y_position",
            "speed",
            "direction",
            "orientation",
            "acceleration",
            "sa",
        ]

        # Note: raw tracking has game_key and play_id, but not always game_play combined.
        # We need to handle this. Let's inspect columns by reading header first or just read all.
        # The description says 'game_play' exists in tracking.

        df_tr = pd.read_csv(tracking_path)

        # Filter relevant plays
        df_tr = df_tr[df_tr["game_play"].isin(relevant_plays)].copy()

        # Sort for windowing
        df_tr.sort_values(by=["game_play", "nfl_player_id", "step"], inplace=True)

        # --- Base Feature Engineering ---

        # Angular conversions (Degrees to Radians)
        df_tr["dir_rad"] = np.deg2rad(df_tr["direction"])
        df_tr["o_rad"] = np.deg2rad(df_tr["orientation"])

        # Cyclical Features
        df_tr["sin_direction"] = np.sin(df_tr["dir_rad"])
        df_tr["cos_direction"] = np.cos(df_tr["dir_rad"])
        df_tr["sin_orientation"] = np.sin(df_tr["o_rad"])
        df_tr["cos_orientation"] = np.cos(df_tr["o_rad"])

        # Velocity Components (for Relative Speed)
        df_tr["v_x"] = df_tr["speed"] * df_tr["sin_direction"]
        df_tr["v_y"] = df_tr["speed"] * df_tr["cos_direction"]

        # Define features to window
        features_to_lag = [
            "x_position",
            "y_position",
            "speed",
            "acceleration",
            "sa",
            "sin_direction",
            "cos_direction",
            "sin_orientation",
            "cos_orientation",
            "v_x",
            "v_y",
        ]

        features_to_roll = ["speed", "acceleration"]

        # --- Windowing (Vectorized) ---
        # We use groupby and shift/rolling.
        # To keep it efficient, we'll process groups.

        grouped = df_tr.groupby(["game_play", "nfl_player_id"])

        # 1. Micro Window (Lags)
        # Range: -WINDOW_MICRO to +WINDOW_MICRO
        lag_cols = {}
        for lag in range(-self.micro_window, self.micro_window + 1):
            suffix = f"_lag{lag}" if lag != 0 else ""
            # Shift: positive lag means looking back (t-k), negative means looking forward?
            # Usually lag k means t-k.
            # We want t-4 to t+4.
            # shift(k) takes value from t-k and puts it at t.
            # So shift(4) gives t-4. shift(-4) gives t+4.

            shifted = grouped[features_to_lag].shift(lag)
            for col in features_to_lag:
                lag_cols[f"{col}{suffix}"] = shifted[col]

        # 2. Macro Window (Rolling Stats)
        # Centered rolling window
        roll_cols = {}
        indexer = grouped.rolling(
            window=2 * self.macro_window + 1, center=True, min_periods=1
        )

        means = indexer[features_to_roll].mean().reset_index(drop=True)
        stds = indexer[features_to_roll].std().reset_index(drop=True)

        # Align indexes (reset_index on rolling drops the grouping keys, but order is preserved if not sorted)
        # However, groupby().rolling() returns MultiIndex.
        # Let's do it safer: transform.

        for col in features_to_roll:
            roll_cols[f"{col}_roll_mean"] = means[col].values
            roll_cols[f"{col}_roll_std"] = stds[col].values

        # Construct Wide DataFrame
        # Start with keys
        df_wide = df_tr[["game_play", "nfl_player_id", "step"]].copy()

        # Add Lags
        df_lags = pd.DataFrame(lag_cols, index=df_wide.index)
        df_wide = pd.concat([df_wide, df_lags], axis=1)

        # Add Rolling (Need to handle index carefully)
        # The rolling result above (means/stds) has a MultiIndex (game_play, nfl_player_id, index).
        # It matches the sorted df_tr structure.
        df_wide[list(roll_cols.keys())] = pd.DataFrame(roll_cols, index=df_wide.index)

        # Fill NaNs created by shifting/rolling at edges
        # Ideally, we fill with nearest or 0. For tracking, 0 might be misleading for position.
        # Forward/Backward fill within group is better, but simple fillna(0) is standard for GBDTs
        # as long as it's consistent.
        df_wide = df_wide.fillna(0)

        return df_wide

    def _merge_and_compute_interactions(
        self, df_meta: pd.DataFrame, df_tracking: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merges P1 and P2 tracking data and computes interaction features.
        """
        # Prepare Metadata
        # Ensure IDs are numeric for merging.
        # 'G' in nfl_player_id_2 becomes NaN
        df_meta = df_meta.copy()
        df_meta["nfl_player_id_1"] = pd.to_numeric(
            df_meta["nfl_player_id_1"], errors="coerce"
        )
        df_meta["nfl_player_id_2"] = pd.to_numeric(
            df_meta["nfl_player_id_2"], errors="coerce"
        )

        # --- Merge Player 1 ---
        # Rename columns to _p1
        p1_cols = [c for c in df_tracking.columns if c not in ["game_play", "step"]]
        rename_p1 = {c: f"{c}_p1" for c in p1_cols}
        rename_p1["nfl_player_id"] = "nfl_player_id_1"  # Key for merge

        df_p1 = df_tracking.rename(columns=rename_p1)

        df_merged = pd.merge(
            df_meta, df_p1, on=["game_play", "step", "nfl_player_id_1"], how="left"
        )

        # --- Merge Player 2 ---
        # Rename columns to _p2
        rename_p2 = {c: f"{c}_p2" for c in p1_cols}
        rename_p2["nfl_player_id"] = "nfl_player_id_2"  # Key for merge

        df_p2 = df_tracking.rename(columns=rename_p2)

        df_merged = pd.merge(
            df_merged, df_p2, on=["game_play", "step", "nfl_player_id_2"], how="left"
        )

        # --- Compute Interactions (Vectorized) ---
        # We compute this for every lag in the micro window
        interaction_features = []

        # Identify lags
        lags = range(-self.micro_window, self.micro_window + 1)

        for lag in lags:
            suffix = f"_lag{lag}" if lag != 0 else ""

            # Column names
            x1 = f"x_position{suffix}_p1"
            y1 = f"y_position{suffix}_p1"
            x2 = f"x_position{suffix}_p2"
            y2 = f"y_position{suffix}_p2"

            vx1 = f"v_x{suffix}_p1"
            vy1 = f"v_y{suffix}_p1"
            vx2 = f"v_x{suffix}_p2"
            vy2 = f"v_y{suffix}_p2"

            # Check if columns exist (they should)
            if x1 in df_merged.columns and x2 in df_merged.columns:
                # Distance
                d_col = f"distance{suffix}"
                df_merged[d_col] = np.sqrt(
                    (df_merged[x1] - df_merged[x2]) ** 2
                    + (df_merged[y1] - df_merged[y2]) ** 2
                )
                interaction_features.append(d_col)

                # Relative Speed
                s_col = f"rel_speed{suffix}"
                df_merged[s_col] = np.sqrt(
                    (df_merged[vx1] - df_merged[vx2]) ** 2
                    + (df_merged[vy1] - df_merged[vy2]) ** 2
                )
                interaction_features.append(s_col)

        # --- Cleanup ---
        # Fill NaNs for P2/Interactions (Ground contacts or missing tracking)
        # We fill with 0. GBDT handles this.
        # Note: Distance 0 might be confusing for contact, but for Ground contact
        # the model learns to ignore P2 features.
        fill_cols = [c for c in df_merged.columns if "_p2" in c] + interaction_features
        df_merged[fill_cols] = df_merged[fill_cols].fillna(0)

        # Also fill P1 NaNs if any (missing tracking)
        p1_feat_cols = [c for c in df_merged.columns if "_p1" in c]
        df_merged[p1_feat_cols] = df_merged[p1_feat_cols].fillna(0)

        # Drop raw position/velocity columns if desired to save space?
        # The prompt implies using "Micro" features which ARE these raw/lagged values.
        # So we keep them.

        # Select final columns
        # Keep ID columns, target, and all generated features
        meta_cols = [
            "contact_id",
            "game_play",
            "step",
            "nfl_player_id_1",
            "nfl_player_id_2",
        ]
        if "contact" in df_merged.columns:
            meta_cols.append("contact")

        feature_cols = (
            p1_feat_cols
            + [c for c in df_merged.columns if "_p2" in c]
            + interaction_features
        )

        final_cols = meta_cols + feature_cols

        # Ensure no duplicates
        final_cols = list(dict.fromkeys(final_cols))

        return df_merged[final_cols]
