import pandas as pd
import numpy as np
import os
from library.config import Config
from library.utils import get_cache_path, save_to_cache, load_from_cache


class FeatureBuilder:
    """
    Constructs feature sets for the Sequential Cascade Dual-Stream GBDT.
    Handles feature engineering, temporal lagging, and context integration.
    """

    def __init__(self):
        self.working_dir = Config.WORKING_DIR
        os.makedirs(self.working_dir, exist_ok=True)

    def _add_tracking_derived_features(self, df):
        """
        Adds sin/cos components for directional features.
        """
        for suffix in ["", "_p1", "_p2"]:
            # Check if base columns exist for this suffix
            # Base columns in raw data: 'direction', 'orientation'
            # We need to handle the case where we are processing the main df or a suffixed one

            # Identify columns to transform
            dir_col = f"direction{suffix}"
            ori_col = f"orientation{suffix}"

            if dir_col in df.columns:
                # Convert to radians
                rads = np.deg2rad(df[dir_col])
                df[f"sin_direction{suffix}"] = np.sin(rads)
                df[f"cos_direction{suffix}"] = np.cos(rads)

            if ori_col in df.columns:
                rads = np.deg2rad(df[ori_col])
                df[f"sin_orientation{suffix}"] = np.sin(rads)
                df[f"cos_orientation{suffix}"] = np.cos(rads)

        return df

    def _calculate_iou_dist(self, df, view):
        """
        Calculates IoU and Centroid Distance for a specific view (sideline/endzone).
        """
        # Define column names based on view
        p1_suffix = f"_{view}_p1"
        p2_suffix = f"_{view}_p2"

        l1 = df[f"left{p1_suffix}"]
        t1 = df[f"top{p1_suffix}"]
        w1 = df[f"width{p1_suffix}"]
        h1 = df[f"height{p1_suffix}"]

        l2 = df[f"left{p2_suffix}"]
        t2 = df[f"top{p2_suffix}"]
        w2 = df[f"width{p2_suffix}"]
        h2 = df[f"height{p2_suffix}"]

        # Calculate Coordinates
        r1 = l1 + w1
        b1 = t1 + h1
        r2 = l2 + w2
        b2 = t2 + h2

        # Intersection
        x_left = np.maximum(l1, l2)
        y_top = np.maximum(t1, t2)
        x_right = np.minimum(r1, r2)
        y_bottom = np.minimum(b1, b2)

        intersection_area = np.maximum(0, x_right - x_left) * np.maximum(
            0, y_bottom - y_top
        )

        # Union
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - intersection_area

        # IoU
        iou_col = f"iou_{view}"
        # Avoid division by zero
        df[iou_col] = intersection_area / (union_area + 1e-6)
        # Set to NaN if any box is missing (will be imputed later)
        mask_missing = (w1.isna()) | (w2.isna())
        df.loc[mask_missing, iou_col] = np.nan

        # Centroid Distance
        c_x1 = l1 + w1 / 2
        c_y1 = t1 + h1 / 2
        c_x2 = l2 + w2 / 2
        c_y2 = t2 + h2 / 2

        dist_col = f"dist_centroid_{view}"
        df[dist_col] = np.sqrt((c_x1 - c_x2) ** 2 + (c_y1 - c_y2) ** 2)
        df.loc[mask_missing, dist_col] = np.nan

        return df

    def _add_visual_derived_features(self, df):
        """
        Adds IoU and Centroid Distance for both views.
        """
        for view in ["sideline", "endzone"]:
            # Check if the requisite columns exist (P1 and P2 must exist)
            # They should exist for Stream A
            if f"left_{view}_p1" in df.columns and f"left_{view}_p2" in df.columns:
                df = self._calculate_iou_dist(df, view)
        return df

    def _generate_lags(self, df, feature_cols, lags, group_cols):
        """
        Generates exponential lags for specified features.
        Assumes df is sorted by group_cols + step.
        """
        # Ensure sorting
        sort_cols = group_cols + ["step"]
        df = df.sort_values(sort_cols).reset_index(drop=True)

        # Create a grouping object
        # Using a specialized approach for speed:
        # We can shift the whole dataframe and mask transitions.
        # However, since 'step' is strictly incremental (0, 1, 2...),
        # we can verify the shift validity by checking step differences.

        out_df = df.copy()

        # Pre-calculate group identifiers to detect boundaries
        # A simple way is to check if shifted group cols match current group cols
        # But checking 'step' continuity is often sufficient and faster if data is dense.
        # Let's use groupby().shift() which is robust.

        grouped = df.groupby(group_cols)

        for lag in lags:
            if lag == 0:
                continue

            # Shift features
            shifted = grouped[feature_cols].shift(
                -lag
            )  # Positive lag in Config is future?
            # Config says: "Negative values are past, positive values are future"
            # Pandas shift(k): shifts data down by k.
            # If we want t-1 (past), we want the value from the previous row to appear in current.
            # That is shift(1).
            # So lag -1 (past) corresponds to shift(1).
            # lag +1 (future) corresponds to shift(-1).

            # Therefore, pandas_shift = -lag

            suffix = f"_lag{lag}"
            shifted.columns = [f"{c}{suffix}" for c in feature_cols]

            out_df = pd.concat([out_df, shifted], axis=1)

        return out_df

    def build_stream_a_features(self, df_merged, load_cached_data=True, split="train"):
        """
        Builds features for Stream A (Player-Player Interaction).
        """
        # 1. Cache Check
        cache_config = {
            "stream": "A",
            "split": split,
            "lags_tracking": Config.LAGS_TRACKING,
            "lags_visual": Config.LAGS_VISUAL,
            "debug": Config.DEBUG,
        }

        path_X = get_cache_path(
            self.working_dir, f"features_{split}_streamA_X", cache_config, "parquet"
        )
        path_y = get_cache_path(
            self.working_dir, f"features_{split}_streamA_y", cache_config, "npy"
        )
        path_ids = get_cache_path(
            self.working_dir, f"features_{split}_streamA_ids", cache_config, "npy"
        )

        if load_cached_data and os.path.exists(path_X) and os.path.exists(path_y):
            print(f"Loading Stream A features from cache for {split}...")
            return (
                load_from_cache(path_X),
                load_from_cache(path_y),
                load_from_cache(path_ids),
            )

        print(f"Building Stream A features for {split}...")

        # 2. Filter for Player-Player pairs
        # Player 2 is NOT 'G'
        df = df_merged[df_merged["nfl_player_id_2"] != "G"].copy()

        # 3. Derived Features
        df = self._add_tracking_derived_features(df)
        df = self._add_visual_derived_features(df)

        # 4. Define Feature Groups for Lagging

        # A. Tracking Features (P1 and P2)
        # Base cols + derived sin/cos
        track_base = Config.TRACKING_BASE_COLS
        track_cols_p1 = [f"{c}_p1" for c in track_base if f"{c}_p1" in df.columns]
        track_cols_p2 = [f"{c}_p2" for c in track_base if f"{c}_p2" in df.columns]

        # B. Visual Features
        # Pairwise (IoU, Dist) + Individual (Box)
        vis_base = Config.VISUAL_BASE_COLS
        vis_cols = []

        for view in ["sideline", "endzone"]:
            # Pairwise
            if "iou" in vis_base:
                vis_cols.append(f"iou_{view}")
            if "dist_centroid" in vis_base:
                vis_cols.append(f"dist_centroid_{view}")

            # Individual
            box_attrs = ["left", "top", "width", "height"]
            for attr in box_attrs:
                if attr in vis_base:
                    vis_cols.append(f"{attr}_{view}_p1")
                    vis_cols.append(f"{attr}_{view}_p2")

        # Filter vis_cols to those that actually exist
        vis_cols = [c for c in vis_cols if c in df.columns]

        # 5. Generate Lags
        # Group by Game + Pair
        group_cols = ["game_play", "nfl_player_id_1", "nfl_player_id_2"]

        # Tracking Lags
        df = self._generate_lags(
            df, track_cols_p1 + track_cols_p2, Config.LAGS_TRACKING, group_cols
        )

        # Visual Lags
        df = self._generate_lags(df, vis_cols, Config.LAGS_VISUAL, group_cols)

        # 6. Select Final Columns
        # Collect all lagged and current columns
        # Current columns (lag 0) are included in the lists above, but _generate_lags skips 0.
        # We need to ensure current columns are in the final set.

        final_cols = []

        # Add tracking features (current + lags)
        for lag in Config.LAGS_TRACKING:
            suffix = f"_lag{lag}" if lag != 0 else ""
            final_cols.extend([f"{c}{suffix}" for c in track_cols_p1 + track_cols_p2])

        # Add visual features (current + lags)
        for lag in Config.LAGS_VISUAL:
            suffix = f"_lag{lag}" if lag != 0 else ""
            final_cols.extend([f"{c}{suffix}" for c in vis_cols])

        # 7. Imputation
        # Visual features might be NaN. Impute with -999.
        # Tracking features should generally be present, but good to be safe.
        X = df[final_cols].fillna(-999)

        # 8. Prepare Targets and IDs
        y = df["contact"].values.astype(int)
        ids = df["contact_id"].values

        # 9. Save to Cache
        print("Saving Stream A features to cache...")
        save_to_cache(X, path_X)
        save_to_cache(y, path_y)
        save_to_cache(ids, path_ids)

        return X, y, ids

    def build_stream_b_features(
        self, df_merged, interaction_predictions, load_cached_data=True, split="train"
    ):
        """
        Builds features for Stream B (Player-Ground Impact).
        Requires interaction_predictions from Stream A to build context features.

        interaction_predictions: DataFrame with ['contact_id', 'prob'] (or similar)
        """
        # 1. Cache Check
        # We include a simple identifier for the preds in the hash if possible, or just rely on split
        cache_config = {
            "stream": "B",
            "split": split,
            "lags_tracking": Config.LAGS_TRACKING,
            "context_window": Config.STREAM_B_ROLLING_WINDOW,
            "debug": Config.DEBUG,
        }

        path_X = get_cache_path(
            self.working_dir, f"features_{split}_streamB_X", cache_config, "parquet"
        )
        path_y = get_cache_path(
            self.working_dir, f"features_{split}_streamB_y", cache_config, "npy"
        )
        path_ids = get_cache_path(
            self.working_dir, f"features_{split}_streamB_ids", cache_config, "npy"
        )

        if load_cached_data and os.path.exists(path_X) and os.path.exists(path_y):
            print(f"Loading Stream B features from cache for {split}...")
            return (
                load_from_cache(path_X),
                load_from_cache(path_y),
                load_from_cache(path_ids),
            )

        print(f"Building Stream B features for {split}...")

        # 2. Filter for Player-Ground pairs
        df = df_merged[df_merged["nfl_player_id_2"] == "G"].copy()

        # 3. Derived Tracking Features (P1 only)
        df = self._add_tracking_derived_features(df)

        # 4. Context Feature Construction (Max Interaction Prob)
        print("Constructing Interaction Context features...")

        # Parse interaction predictions
        # contact_id format: game_key_play_id_step_p1_p2
        # We need to map these to (game_play, step, player_id) -> max(prob)

        preds = interaction_predictions.copy()
        if "prob" not in preds.columns and "contact" in preds.columns:
            # Handle case where col might be named 'contact' (from submission file) or 'prob'
            preds.rename(columns={"contact": "prob"}, inplace=True)

        # Vectorized parsing
        split_ids = preds["contact_id"].str.split("_", expand=True)
        # 0: game_key, 1: play_id, 2: step, 3: p1, 4: p2

        preds["game_play"] = split_ids[0] + "_" + split_ids[1]
        preds["step"] = split_ids[2].astype(int)
        preds["p1"] = split_ids[3]
        preds["p2"] = split_ids[4]

        # We need to attribute the probability to BOTH players involved in the collision
        # Create a long format: [game_play, step, player, prob]

        df_p1 = preds[["game_play", "step", "p1", "prob"]].rename(
            columns={"p1": "nfl_player_id"}
        )
        df_p2 = preds[["game_play", "step", "p2", "prob"]].rename(
            columns={"p2": "nfl_player_id"}
        )

        # Concatenate and aggregate
        context_long = pd.concat([df_p1, df_p2], axis=0)

        # Group by player-step to find max probability of ANY contact
        # This tells us: "Is this player hitting anyone right now?"
        interaction_context = (
            context_long.groupby(["game_play", "step", "nfl_player_id"])["prob"]
            .max()
            .reset_index()
        )
        interaction_context.rename(
            columns={"prob": "max_interaction_prob_raw"}, inplace=True
        )

        # Merge into Stream B dataframe
        # Stream B rows are (game_play, step, nfl_player_id_1 (Subject), nfl_player_id_2 (G))
        # We merge on nfl_player_id_1

        df = pd.merge(
            df,
            interaction_context,
            left_on=["game_play", "step", "nfl_player_id_1"],
            right_on=["game_play", "step", "nfl_player_id"],
            how="left",
        )

        # Fill missing context (no interactions predicted) with 0
        df["max_interaction_prob_raw"] = df["max_interaction_prob_raw"].fillna(0.0)

        # Apply Rolling Window
        # Sort by player trajectory
        df = df.sort_values(["game_play", "nfl_player_id_1", "step"]).reset_index(
            drop=True
        )

        # Calculate Rolling Max
        # We group by player and apply rolling max
        window = Config.STREAM_B_ROLLING_WINDOW

        # Using groupby().rolling() is standard but can be slow on many groups.
        # Since data is sorted, we can use a trick or just standard pandas rolling.
        # Given dataset size (~100k-1M rows), standard groupby rolling is acceptable.

        # However, to ensure we don't roll across different plays/players, we must group.
        # Optimization: Use transform with a lambda or explicit loop is slow.
        # Faster: df.groupby(...).rolling(...)

        print("Calculating rolling context window...")
        # Note: 'min_periods=1' ensures we get values at the start of the window
        # 'closed=right' includes current step

        # We need to handle the index resulting from groupby().rolling()
        # It creates a MultiIndex (group_keys..., original_index)

        rolled = (
            df.groupby(["game_play", "nfl_player_id_1"])["max_interaction_prob_raw"]
            .rolling(window=window, min_periods=1)
            .max()
        )

        # The index of 'rolled' will match 'df' if we drop the group keys level
        rolled = rolled.reset_index(level=[0, 1], drop=True)

        # Assign back (ensure alignment via index)
        df["max_interaction_prob"] = rolled.sort_index()

        # 5. Define Feature Groups for Lagging

        # Tracking Features (P1 only)
        track_base = Config.TRACKING_BASE_COLS
        track_cols_p1 = [f"{c}_p1" for c in track_base if f"{c}_p1" in df.columns]

        # Context Feature
        context_col = ["max_interaction_prob"]

        # 6. Generate Lags
        group_cols = ["game_play", "nfl_player_id_1"]

        # Lag Tracking
        df = self._generate_lags(df, track_cols_p1, Config.LAGS_TRACKING, group_cols)

        # Lag Context?
        # The context itself is a rolling max of the past.
        # Do we need lags of the rolling max?
        # The description says: "Feature Space: Kinematics... and Interaction Context".
        # It doesn't explicitly demand lags of the context feature, but "Feature Space" usually implies current state.
        # However, since it's a rolling max of the *past* 10 steps, it already encodes history.
        # We will keep just the current value of the rolling feature.

        # 7. Select Final Columns
        final_cols = []

        # Tracking Lags
        for lag in Config.LAGS_TRACKING:
            suffix = f"_lag{lag}" if lag != 0 else ""
            final_cols.extend([f"{c}{suffix}" for c in track_cols_p1])

        # Add Context
        final_cols.append("max_interaction_prob")

        # 8. Imputation
        X = df[final_cols].fillna(-999)

        # 9. Prepare Targets and IDs
        y = df["contact"].values.astype(int)
        ids = df["contact_id"].values

        # 10. Save to Cache
        print("Saving Stream B features to cache...")
        save_to_cache(X, path_X)
        save_to_cache(y, path_y)
        save_to_cache(ids, path_ids)

        return X, y, ids
