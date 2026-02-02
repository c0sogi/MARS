import os
import gc
import numpy as np
import pandas as pd
import torch
from library.config import Config
from library.utils import seed_everything


class FeatureEngineer:
    """
    Implements the Entity-First data pipeline for KCVR-Net (Idea 23).
    Handles tracking and visual data processing, wide-window generation,
    and physics-based feature engineering with numerical stability corrections.
    """

    def __init__(self, use_tqdm=False):
        self.use_tqdm = use_tqdm
        self.cache_dir = Config.WORKING_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_path(self, split, suffix):
        return os.path.join(self.cache_dir, f"{split}_{suffix}.parquet")

    def _apply_shifts(self, df, group_cols, feature_cols, window_size):
        """
        Efficiently creates lag/lead features using vectorized shifts and masking.
        Avoids slow groupby().apply() operations.
        """
        # Ensure data is sorted
        sort_cols = group_cols + ["step"]
        df = df.sort_values(sort_cols).reset_index(drop=True)

        # Base columns for validity check
        id_col = group_cols[0]  # game_play
        id_col_2 = group_cols[1] if len(group_cols) > 1 else None

        # Pre-calculate shift identifiers to avoid repeated access
        gp_series = df[id_col]
        if id_col_2:
            id2_series = df[id_col_2]

        shifts = range(-window_size, window_size + 1)

        # Container for new columns
        new_cols = {}

        for s in shifts:
            # Pandas shift: positive k shifts data down (t gets t-k value).
            # We want t to have t-s. If s=-5 (past), we want value from 5 rows ago -> shift(5).
            # If s=5 (future), we want value from 5 rows ahead -> shift(-5).
            shift_amount = -s

            # Create mask for valid shifts (must stay within same group)
            # Shift the group columns by the same amount to check equality
            gp_shifted = gp_series.shift(shift_amount)
            mask = gp_series == gp_shifted

            if id_col_2:
                id2_shifted = id2_series.shift(shift_amount)
                mask = mask & (id2_series == id2_shifted)

            # Apply shift to features
            for col in feature_cols:
                col_name = f"{col}_{s}"
                shifted_data = df[col].shift(shift_amount)
                # Apply mask: invalid shifts become NaN (or 0/fill later)
                # We use numpy where for speed, filling invalid with NaN
                new_cols[col_name] = np.where(mask, shifted_data, np.nan)

        # Concatenate all new columns at once
        df_shifts = pd.DataFrame(new_cols, index=df.index)

        # We only return the shifted columns + keys, not the original df to save memory if needed
        # But usually we merge back. Here we return the wide dataframe.
        # We keep the key columns for merging.
        result = pd.concat([df[group_cols + ["step"]], df_shifts], axis=1)
        return result

    def process_tracking(self, tracking_path, game_plays):
        """
        Loads and processes tracking data into wide format.
        """
        # Load data
        df = pd.read_csv(tracking_path, usecols=Config.TRACKING_COLS)

        # Filter relevant plays
        df = df[df["game_play"].isin(game_plays)].copy()

        # Standardize orientation/direction to 0-360
        df["orientation"] = df["orientation"] % 360
        df["direction"] = df["direction"] % 360

        # Generate Wide Features (Entity-First)
        # Features to shift: x, y, speed, direction, orientation, acceleration, sa
        feats_to_shift = [
            c
            for c in Config.TRACKING_COLS
            if c not in ["game_play", "step", "nfl_player_id"]
        ]

        df_wide = self._apply_shifts(
            df,
            group_cols=["game_play", "nfl_player_id"],
            feature_cols=feats_to_shift,
            window_size=Config.WINDOW_SIZE,
        )

        return df_wide

    def process_visuals(self, helmets_path, game_plays):
        """
        Loads and processes helmet data:
        1. Maps frame to step.
        2. Applies Max-Pooling (select largest box per player/step).
        3. Generates wide format.
        """
        df = pd.read_csv(helmets_path, usecols=Config.HELMET_COLS + ["frame"])

        # Filter
        df = df[df["game_play"].isin(game_plays)].copy()

        # Map Frame to Step
        # Snap is frame 300 (step 0). 59.94 Hz -> ~6 frames per 0.1s step.
        # step = round((frame - 300) / 6)
        df["step"] = ((df["frame"] - 300) / 5.994).round().astype(int)

        # Calculate Area for Max-Pooling
        df["area"] = df["width"] * df["height"]

        # Max-Pooling Strategy: Sort by area desc, drop duplicates on keys
        # This keeps the largest box for each player at each step
        df = df.sort_values("area", ascending=False)
        df = df.drop_duplicates(subset=["game_play", "step", "nfl_player_id"])

        # Generate Wide Features
        feats_to_shift = ["left", "width", "top", "height", "area"]

        df_wide = self._apply_shifts(
            df,
            group_cols=["game_play", "nfl_player_id"],
            feature_cols=feats_to_shift,
            window_size=Config.WINDOW_SIZE,
        )

        return df_wide

    def apply_stability_corrections(self, df):
        """
        Applies numerical stability corrections:
        1. Angular continuity (Shortest Arc).
        2. Clamping unbounded features.
        """
        shifts = range(-Config.WINDOW_SIZE, Config.WINDOW_SIZE + 1)

        for s in shifts:
            # 1. Angular Continuity for Relative Angle
            # relative_angle = dir1 - dir2
            col = f"relative_angle_{s}"
            if col in df.columns:
                # (a - b) % 360
                # But we already computed simple diff. Let's fix it.
                # We need to assume the raw calc was simple subtraction.
                # shortest arc: min(|d|, 360-|d|)
                # Implementation: ((d + 180) % 360) - 180 is signed shortest distance
                # Or just absolute shortest distance for magnitude

                # We'll use signed shortest distance normalized to [-180, 180]
                df[col] = (df[col] + 180) % 360 - 180

            # 2. Clamping
            # Features like closing_speed can explode
            clamp_cols = [f"closing_speed_{s}", f"relative_speed_{s}"]
            for c in clamp_cols:
                if c in df.columns:
                    df[c] = df[c].clip(Config.CLAMP_MIN, Config.CLAMP_MAX)

            # Fill NaNs (from shifts or missing data) with 0
            # This is crucial for the DCN to not propagate NaNs
            # We select all columns for this lag
            lag_cols = [c for c in df.columns if c.endswith(f"_{s}")]
            df[lag_cols] = df[lag_cols].fillna(0)

        return df

    def merge_and_format(self, meta_df, track_wide, vis_wide):
        """
        Merges metadata with wide feature streams and computes pairwise interactions.
        """
        # 1. Merge Player 1 Tracking
        # Rename columns to indicate P1
        p1_track = track_wide.add_suffix("_1")
        # Fix join keys (they got suffixed too)
        p1_track = p1_track.rename(
            columns={
                "game_play_1": "game_play",
                "nfl_player_id_1": "nfl_player_id_1",
                "step_1": "step",
            }
        )

        merged = meta_df.merge(
            p1_track, on=["game_play", "step", "nfl_player_id_1"], how="left"
        )

        # 2. Merge Player 2 Tracking
        # Handle Ground: If nfl_player_id_2 is 'G', we don't merge, we impute later.
        # We first merge for non-G rows.

        # Prepare P2 track
        p2_track = track_wide.add_suffix("_2")
        p2_track = p2_track.rename(
            columns={
                "game_play_2": "game_play",
                "nfl_player_id_2": "nfl_player_id_2",
                "step_2": "step",
            }
        )

        # Convert G to NaN or handle type mismatch if necessary
        # nfl_player_id_2 in meta is object (string), in track is int.
        # We need to align types.
        # Extract numeric IDs from meta where possible
        is_ground = merged["nfl_player_id_2"] == "G"

        # Create a temp column for merging P2
        merged["p2_merge_id"] = pd.to_numeric(
            merged["nfl_player_id_2"], errors="coerce"
        )

        merged = merged.merge(
            p2_track,
            left_on=["game_play", "step", "p2_merge_id"],
            right_on=["game_play", "step", "nfl_player_id_2"],
            how="left",
            suffixes=("", "_drop"),
        )
        merged = merged.drop(
            columns=["p2_merge_id", "nfl_player_id_2_drop"], errors="ignore"
        )

        # 3. Merge Visuals P1
        p1_vis = vis_wide.add_suffix("_1")
        p1_vis = p1_vis.rename(
            columns={
                "game_play_1": "game_play",
                "nfl_player_id_1": "nfl_player_id_1",
                "step_1": "step",
            }
        )
        merged = merged.merge(
            p1_vis, on=["game_play", "step", "nfl_player_id_1"], how="left"
        )

        # 4. Merge Visuals P2
        p2_vis = vis_wide.add_suffix("_2")
        p2_vis = p2_vis.rename(
            columns={
                "game_play_2": "game_play",
                "nfl_player_id_2": "nfl_player_id_2",
                "step_2": "step",
            }
        )
        merged["p2_merge_id"] = pd.to_numeric(
            merged["nfl_player_id_2"], errors="coerce"
        )
        merged = merged.merge(
            p2_vis,
            left_on=["game_play", "step", "p2_merge_id"],
            right_on=["game_play", "step", "nfl_player_id_2"],
            how="left",
            suffixes=("", "_drop"),
        )
        merged = merged.drop(
            columns=["p2_merge_id", "nfl_player_id_2_drop"], errors="ignore"
        )

        # 5. Impute Ground & Compute Pairwise Features
        shifts = range(-Config.WINDOW_SIZE, Config.WINDOW_SIZE + 1)

        for s in shifts:
            # Suffix for this lag
            sfx = f"_{s}"

            # Feature names for this lag
            x1 = f"x_position{sfx}_1"
            y1 = f"y_position{sfx}_1"
            s1 = f"speed{sfx}_1"
            d1 = f"direction{sfx}_1"

            x2 = f"x_position{sfx}_2"
            y2 = f"y_position{sfx}_2"
            s2 = f"speed{sfx}_2"
            d2 = f"direction{sfx}_2"

            # --- Ground Imputation ---
            # If Ground, P2 pos = P1 pos, P2 speed = 0
            # We use numpy where for vectorization

            # Fill NaNs in P1 first (missing tracking)
            for col in [x1, y1, s1, d1]:
                if col in merged.columns:
                    merged[col] = merged[col].fillna(0)

            # Impute P2 for Ground
            # Note: is_ground is a boolean series aligned with merged
            if x2 in merged.columns:
                merged.loc[is_ground, x2] = merged.loc[is_ground, x1]
                merged.loc[is_ground, y2] = merged.loc[is_ground, y1]
                merged.loc[is_ground, s2] = 0
                merged.loc[is_ground, d2] = 0
                # Zero out other P2 kinematic features for ground
                for k in ["acceleration", "sa", "orientation"]:
                    k_col = f"{k}{sfx}_2"
                    if k_col in merged.columns:
                        merged.loc[is_ground, k_col] = 0

                # Visuals for Ground: Zero out
                for v in ["left", "width", "top", "height", "area"]:
                    v_col = f"{v}{sfx}_2"
                    if v_col in merged.columns:
                        merged.loc[is_ground, v_col] = 0

            # Fill remaining NaNs in P2 (missing tracking for non-ground)
            for col in [x2, y2, s2, d2]:
                if col in merged.columns:
                    merged[col] = merged[col].fillna(0)

            # --- Pairwise Feature Calculation ---
            # Distance
            dx = merged[x1] - merged[x2]
            dy = merged[y1] - merged[y2]
            dist_col = f"distance{sfx}"
            merged[dist_col] = np.sqrt(dx**2 + dy**2)

            # Relative Speed (Scalar diff as per simple physics, or vector?)
            # Config lists 'relative_speed'. Vector magnitude is more physical.
            # vx = s * sin(d_rad), vy = s * cos(d_rad)
            # NFL direction: 0 is Y axis (North), 90 is X axis (East)?
            # Standard: 0=X, 90=Y. NFL tracking data: 0=X, 90=Y usually, or 0=Y.
            # Regardless, consistent math works.
            d1_rad = np.radians(merged[d1])
            d2_rad = np.radians(merged[d2])

            vx1 = merged[s1] * np.sin(d1_rad)
            vy1 = merged[s1] * np.cos(d1_rad)
            vx2 = merged[s2] * np.sin(d2_rad)
            vy2 = merged[s2] * np.cos(d2_rad)

            dvx = vx1 - vx2
            dvy = vy1 - vy2

            merged[f"relative_speed{sfx}"] = np.sqrt(dvx**2 + dvy**2)

            # Closing Speed: Projection of rel velocity onto distance vector
            # Unit vector from 1 to 2: (dx, dy) / dist
            # Actually closing speed is usually defined as -d(dist)/dt.
            # Positive closing speed means getting closer.
            # Vector pointing 2->1 is (-dx, -dy). Rel vel is v1 - v2.
            # Closing = - ( (v1-v2) dot (p1-p2) ) / |p1-p2|
            # Let's use: (v2 - v1) dot (p2 - p1) / dist?
            # Standard definition: rate of decrease of distance.
            # v_rel = v1 - v2. r_rel = p1 - p2.
            # closing = - (v_rel . r_rel) / |r_rel|
            # if dist is 0, closing is 0.

            # dot product
            dot = dvx * dx + dvy * dy
            # If dist > 0: -dot / dist. Else 0.
            merged[f"closing_speed{sfx}"] = np.where(
                merged[dist_col] > 1e-6, -(dot / merged[dist_col]), 0
            )

            # Relative Angle
            merged[f"relative_angle{sfx}"] = merged[d1] - merged[d2]

            # Visual IoU (Approximate)
            # We don't have true IoU without box overlap logic, but we can compute
            # a proxy or just 0 if not needed. Config lists 'visual_iou'.
            # Let's compute simple 1D overlap product / union area.
            # Actually, standard IoU is fine.
            # Box 1: l1, t1, w1, h1 -> r1=l1+w1, b1=t1+h1
            l1 = merged[f"left{sfx}_1"].fillna(0)
            t1 = merged[f"top{sfx}_1"].fillna(0)
            w1 = merged[f"width{sfx}_1"].fillna(0)
            h1 = merged[f"height{sfx}_1"].fillna(0)
            r1 = l1 + w1
            b1 = t1 + h1

            l2 = merged[f"left{sfx}_2"].fillna(0)
            t2 = merged[f"top{sfx}_2"].fillna(0)
            w2 = merged[f"width{sfx}_2"].fillna(0)
            h2 = merged[f"height{sfx}_2"].fillna(0)
            r2 = l2 + w2
            b2 = t2 + h2

            x_left = np.maximum(l1, l2)
            y_top = np.maximum(t1, t2)
            x_right = np.minimum(r1, r2)
            y_bottom = np.minimum(b1, b2)

            inter_w = np.maximum(0, x_right - x_left)
            inter_h = np.maximum(0, y_bottom - y_top)
            inter_area = inter_w * inter_h

            area1 = w1 * h1
            area2 = w2 * h2
            union_area = area1 + area2 - inter_area

            merged[f"visual_iou{sfx}"] = np.where(
                union_area > 1e-6, inter_area / union_area, 0
            )

        # Apply corrections
        merged = self.apply_stability_corrections(merged)

        # Extract Final Feature Matrices
        # Flattened order: [feat1_t-5, feat2_t-5, ..., feat1_t+5, ...]
        # We iterate lags, then features

        kin_cols = []
        vis_cols = []

        for s in shifts:
            sfx = f"_{s}"
            # Kinematic Features from Config
            # Map Config names to dataframe names
            # Config: x_position_1 -> x_position_s_1
            for k in Config.KINEMATIC_FEATURES:
                # Construct column name in merged df
                # e.g. "x_position_1" -> split -> "x_position" + sfx + "_1"
                # "distance" -> "distance" + sfx
                if "_1" in k:
                    base = k.replace("_1", "")
                    col = f"{base}{sfx}_1"
                elif "_2" in k:
                    base = k.replace("_2", "")
                    col = f"{base}{sfx}_2"
                else:
                    col = f"{k}{sfx}"

                kin_cols.append(col)

            # Visual Features
            for v in Config.VISUAL_FEATURES:
                if "_1" in v:
                    base = v.replace("_1", "")
                    col = f"{base}{sfx}_1"
                elif "_2" in v:
                    base = v.replace("_2", "")
                    col = f"{base}{sfx}_2"
                else:
                    col = f"{v}{sfx}"
                vis_cols.append(col)

        # Ensure all columns exist (fill missing with 0)
        for c in kin_cols + vis_cols:
            if c not in merged.columns:
                merged[c] = 0.0

        X_kin = merged[kin_cols].values.astype(np.float32)
        X_vis = merged[vis_cols].values.astype(np.float32)
        y = merged["contact"].values.astype(np.float32)

        # Return contact_ids for test submission creation
        ids = merged["contact_id"].values if "contact_id" in merged.columns else None

        return X_kin, X_vis, y, ids

    def process_data(self, split="train", load_cached=True):
        """
        Main orchestrator. Checks cache, processes data, saves cache.
        """
        # Paths
        cache_kin = self._get_cache_path(split, "X_kin")
        cache_vis = self._get_cache_path(split, "X_vis")
        cache_y = self._get_cache_path(split, "y")
        cache_ids = self._get_cache_path(split, "ids")

        # Check Cache
        if load_cached and os.path.exists(cache_kin):
            print(f"Loading cached {split} data...")
            X_kin = np.load(cache_kin.replace(".parquet", ".npy"))
            X_vis = np.load(cache_vis.replace(".parquet", ".npy"))
            y = np.load(cache_y.replace(".parquet", ".npy"))
            ids = np.load(cache_ids.replace(".parquet", ".npy"), allow_pickle=True)
            return X_kin, X_vis, y, ids

        print(f"Processing {split} data from scratch...")

        # 1. Load Metadata
        if split == "train":
            meta_path = Config.TRAIN_METADATA_PATH
            track_path = Config.TRAIN_TRACKING_PATH
            helmet_path = Config.TRAIN_HELMETS_PATH
        elif split == "validation":
            meta_path = Config.VAL_METADATA_PATH
            # Validation uses train tracking/helmets file but different rows
            track_path = Config.TRAIN_TRACKING_PATH
            helmet_path = Config.TRAIN_HELMETS_PATH
        else:  # test
            meta_path = Config.TEST_METADATA_PATH
            track_path = Config.TEST_TRACKING_PATH
            helmet_path = Config.TEST_HELMETS_PATH

        meta_df = pd.read_csv(meta_path)

        # Optimization: Only load tracking/helmets for relevant game_plays
        unique_gps = meta_df["game_play"].unique()

        # 2. Process Streams
        print("  Processing Tracking...")
        track_wide = self.process_tracking(track_path, unique_gps)

        print("  Processing Visuals...")
        vis_wide = self.process_visuals(helmet_path, unique_gps)

        # 3. Merge
        print("  Merging and Engineering Features...")
        X_kin, X_vis, y, ids = self.merge_and_format(meta_df, track_wide, vis_wide)

        # 4. Save Cache
        print("  Saving Cache...")
        np.save(cache_kin.replace(".parquet", ".npy"), X_kin)
        np.save(cache_vis.replace(".parquet", ".npy"), X_vis)
        np.save(cache_y.replace(".parquet", ".npy"), y)
        np.save(cache_ids.replace(".parquet", ".npy"), ids)

        # Clean up
        del track_wide, vis_wide, meta_df
        gc.collect()

        return X_kin, X_vis, y, ids
