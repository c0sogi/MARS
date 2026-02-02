import pandas as pd
import numpy as np
import os
import gc
from library.config import Config
from library.utils import generate_hash


class FeatureEngineer:
    def __init__(self):
        self.cache_dir = Config.WORKING_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_path(self, prefix, data_hash):
        return os.path.join(self.cache_dir, f"{prefix}_{data_hash}.parquet")

    def _save_ids_y(self, df, prefix, data_hash):
        # Save IDs and Targets separately for easy loading during training
        # IDs: contact_id
        # y: contact (if exists)
        ids_path = os.path.join(self.cache_dir, f"{prefix}_ids_{data_hash}.npy")
        y_path = os.path.join(self.cache_dir, f"{prefix}_y_{data_hash}.npy")

        np.save(ids_path, df["contact_id"].values)
        if "contact" in df.columns:
            np.save(y_path, df["contact"].values)
        else:
            # For test set where contact might be placeholder, still save it
            np.save(y_path, np.zeros(len(df)))

        return ids_path, y_path

    def _load_ids_y(self, prefix, data_hash):
        ids_path = os.path.join(self.cache_dir, f"{prefix}_ids_{data_hash}.npy")
        y_path = os.path.join(self.cache_dir, f"{prefix}_y_{data_hash}.npy")

        if os.path.exists(ids_path) and os.path.exists(y_path):
            return np.load(ids_path, allow_pickle=True), np.load(y_path)
        return None, None

    def _calculate_iou(self, box1, box2):
        """
        Vectorized IoU calculation.
        Box format: [left, width, top, height]
        """
        # box: x, w, y, h
        # Convert to x1, y1, x2, y2
        b1_x1, b1_y1 = box1[:, 0], box1[:, 2]
        b1_x2, b1_y2 = b1_x1 + box1[:, 1], b1_y1 + box1[:, 3]

        b2_x1, b2_y1 = box2[:, 0], box2[:, 2]
        b2_x2, b2_y2 = b2_x1 + box2[:, 1], b2_y1 + box2[:, 3]

        # Intersection
        x_left = np.maximum(b1_x1, b2_x1)
        y_top = np.maximum(b1_y1, b2_y1)
        x_right = np.minimum(b1_x2, b2_x2)
        y_bottom = np.minimum(b1_y2, b2_y2)

        intersection_area = np.maximum(0, x_right - x_left) * np.maximum(
            0, y_bottom - y_top
        )

        b1_area = box1[:, 1] * box1[:, 3]
        b2_area = box2[:, 1] * box2[:, 3]

        union_area = b1_area + b2_area - intersection_area

        # Avoid division by zero
        iou = np.zeros_like(intersection_area, dtype=np.float32)
        mask = union_area > 0
        iou[mask] = intersection_area[mask] / union_area[mask]

        return iou

    def _add_lags(self, df, group_cols, feature_cols, lags):
        """
        Adds lag features to dataframe.
        Assumes df is sorted by group_cols + ['step'].
        """
        # Create a copy to avoid fragmentation warnings if adding many columns
        df_out = df.copy()

        # We can use groupby().shift() but it's slow for many groups.
        # Faster approach: Global shift with mask check.
        # 1. Ensure sorted
        # df_out should already be sorted by caller

        # 2. Create a group identifier
        # Combine group cols into a single ID for boundary checking
        if len(group_cols) == 1:
            group_ids = df_out[group_cols[0]]
        else:
            # Fast tuple hashing or string concat
            # String concat is safer for mixed types
            group_ids = df_out[group_cols[0]].astype(str)
            for col in group_cols[1:]:
                group_ids = group_ids + "_" + df_out[col].astype(str)

        group_ids = group_ids.values

        for lag in lags:
            if lag == 0:
                continue

            for col in feature_cols:
                new_col_name = f"{col}_lag{lag}"
                shifted_vals = df_out[col].shift(lag)
                shifted_groups = pd.Series(group_ids).shift(lag)

                # Mask where group changed (i.e. shifted data comes from previous play/pair)
                mask = group_ids == shifted_groups

                # Fill invalid shifts with 0 or NaN (we use 0 for XGBoost usually, or keep NaN)
                # Using 0 for missing lags is standard in this context unless feature is strictly non-zero
                df_out[new_col_name] = np.where(mask, shifted_vals, 0.0)

        return df_out

    def process_stream_a(
        self, df_labels, df_tracking, df_helmets, load_cached_data=True
    ):
        """
        Generates features for Stream A (Interaction: Player vs Player).
        Features: System Energy, Relative Geometry, Visual Consensus.
        """
        # 1. Generate Hash
        # Hash depends on the content of inputs. Using shape/head/tail summary for speed.
        input_summary = {
            "labels_len": len(df_labels),
            "tracking_len": len(df_tracking),
            "helmets_len": len(df_helmets),
            "first_contact": (
                str(df_labels.iloc[0]["contact_id"]) if not df_labels.empty else ""
            ),
            "config": Config.STREAM_A_FEATURES,
            "lags": Config.LAGS_ENERGY + Config.LAGS_VISUAL,
        }
        data_hash = generate_hash(input_summary)
        cache_path = self._get_cache_path("features_streamA_X", data_hash)

        # 2. Load Cache
        if load_cached_data and os.path.exists(cache_path):
            ids, y = self._load_ids_y("features_streamA", data_hash)
            if ids is not None:
                X = pd.read_parquet(cache_path)
                return X, y, ids

        print(f"Generating Stream A features (Hash: {data_hash})...")

        # 3. Preprocessing
        # Filter tracking to relevant plays
        relevant_plays = df_labels["game_play"].unique()
        df_track = df_tracking[df_tracking["game_play"].isin(relevant_plays)].copy()

        # Ensure IDs are strings for merging
        df_labels["nfl_player_id_1"] = df_labels["nfl_player_id_1"].astype(str)
        df_labels["nfl_player_id_2"] = df_labels["nfl_player_id_2"].astype(str)
        df_track["nfl_player_id"] = df_track["nfl_player_id"].astype(str)

        # 4. Merge Tracking for P1 and P2
        # P1
        df_merged = pd.merge(
            df_labels,
            df_track.add_suffix("_p1"),
            left_on=["game_play", "step", "nfl_player_id_1"],
            right_on=["game_play_p1", "step_p1", "nfl_player_id_p1"],
            how="left",
        )
        # P2
        df_merged = pd.merge(
            df_merged,
            df_track.add_suffix("_p2"),
            left_on=["game_play", "step", "nfl_player_id_2"],
            right_on=["game_play_p2", "step_p2", "nfl_player_id_p2"],
            how="left",
        )

        # Fill missing tracking (e.g. if player not tracked at that step)
        track_cols = [
            "x_position",
            "y_position",
            "speed",
            "acceleration",
            "direction",
            "orientation",
        ]
        for col in track_cols:
            df_merged[f"{col}_p1"] = df_merged[f"{col}_p1"].fillna(0)
            df_merged[f"{col}_p2"] = df_merged[f"{col}_p2"].fillna(0)

        # 5. Compute Geometry Features
        df_merged["distance"] = np.sqrt(
            (df_merged["x_position_p1"] - df_merged["x_position_p2"]) ** 2
            + (df_merged["y_position_p1"] - df_merged["y_position_p2"]) ** 2
        )

        df_merged["relative_speed"] = np.abs(
            df_merged["speed_p1"] - df_merged["speed_p2"]
        )

        # Closure Rate: -(v1 - v2) dot (r1 - r2) / |r1 - r2|
        # Simplify: Derivative of distance over time (finite difference later) or projection
        # We'll use a simplified projection based on speed and direction
        # Convert direction to radians (0 is north/up? usually NFL data 90 is x, 0 is y. Assuming standard trig)
        # Actually standard NFL tracking: 0 is Y axis (short), 90 is X axis (long).
        # But we just need consistency.
        def get_xy_vel(speed, direction):
            rad = np.radians(direction)
            # Assuming 0 is Y (North), 90 is X (East)
            vy = speed * np.cos(rad)
            vx = speed * np.sin(rad)
            return vx, vy

        vx1, vy1 = get_xy_vel(df_merged["speed_p1"], df_merged["direction_p1"])
        vx2, vy2 = get_xy_vel(df_merged["speed_p2"], df_merged["direction_p2"])

        dx = df_merged["x_position_p1"] - df_merged["x_position_p2"]
        dy = df_merged["y_position_p1"] - df_merged["y_position_p2"]

        # Relative velocity vector
        rvx = vx1 - vx2
        rvy = vy1 - vy2

        # Project onto displacement vector
        # closure rate > 0 means closing in? Usually defined as -dR/dt.
        # If moving towards each other, distance decreases.
        dot_prod = (rvx * dx) + (rvy * dy)
        dist_safe = df_merged["distance"].replace(0, 1e-6)
        # If dot_prod is negative, they are moving towards each other (since dx is p1-p2)
        # Wait: v1 moving to p2, v2 moving to p1.
        # Let's just use the dot product normalized.
        df_merged["closure_rate"] = -(dot_prod / dist_safe)

        # 6. Visual Consensus Features
        # Map step to frame: frame = 300 + step * 6 (approx 59.94Hz * 0.1s)
        # We round to nearest integer
        df_merged["frame_approx"] = (
            (300 + df_merged["step"] * 5.994).round().astype(int)
        )

        # Prepare helmets
        df_h = df_helmets[df_helmets["game_play"].isin(relevant_plays)].copy()
        df_h["nfl_player_id"] = df_h["nfl_player_id"].astype(str)

        # Pivot helmets to have one row per (game_play, frame, player) with Side/End boxes
        # Helmets cols: game_play, view, frame, nfl_player_id, left, width, top, height
        # We need to join efficiently.
        # Let's separate Sideline and Endzone
        df_h_side = df_h[df_h["view"] == "Sideline"].set_index(
            ["game_play", "frame", "nfl_player_id"]
        )
        df_h_end = df_h[df_h["view"] == "Endzone"].set_index(
            ["game_play", "frame", "nfl_player_id"]
        )

        # Helper to get boxes for a list of keys
        def get_boxes(keys, df_lookup):
            # keys is a DataFrame or list of tuples
            # We use merge/join
            # df_keys: game_play, frame_approx, player_id
            # df_lookup index: game_play, frame, player_id
            # Rename frame_approx to frame for join
            base = keys.rename(columns={"frame_approx": "frame"})
            joined = base.merge(
                df_lookup,
                left_on=["game_play", "frame", "pid"],
                right_index=True,
                how="left",
            )
            # Fill missing with 0
            cols = ["left", "width", "top", "height"]
            return joined[cols].fillna(0).values

        # Keys for P1 and P2
        keys_p1 = df_merged[["game_play", "frame_approx", "nfl_player_id_1"]].rename(
            columns={"nfl_player_id_1": "pid"}
        )
        keys_p2 = df_merged[["game_play", "frame_approx", "nfl_player_id_2"]].rename(
            columns={"nfl_player_id_2": "pid"}
        )

        # Get boxes
        box_p1_side = get_boxes(keys_p1, df_h_side)
        box_p2_side = get_boxes(keys_p2, df_h_side)
        box_p1_end = get_boxes(keys_p1, df_h_end)
        box_p2_end = get_boxes(keys_p2, df_h_end)

        # Calculate IoUs
        iou_side = self._calculate_iou(box_p1_side, box_p2_side)
        iou_end = self._calculate_iou(box_p1_end, box_p2_end)

        df_merged["iou_side"] = iou_side
        df_merged["iou_end"] = iou_end

        # Consensus metrics
        df_merged["max_iou"] = np.maximum(iou_side, iou_end)
        df_merged["min_iou"] = np.minimum(iou_side, iou_end)
        df_merged["iou_diff"] = np.abs(iou_side - iou_end)

        # 7. Temporal Lags
        # Sort for shifting
        df_merged.sort_values(
            by=["game_play", "nfl_player_id_1", "nfl_player_id_2", "step"], inplace=True
        )

        # Define features to lag
        energy_feats = ["speed_p1", "acceleration_p1", "speed_p2", "acceleration_p2"]
        geo_feats = ["distance", "relative_speed", "closure_rate"]
        vis_feats = ["max_iou", "min_iou", "iou_diff"]

        # Apply Energy/Geo Lags (0 to 15)
        # We use Config.LAGS_ENERGY for these
        # Note: Config says LAGS_ENERGY = [0, -1, 1...]
        # We need to handle positive (future) and negative (past) lags.
        # shift(1) is previous step (t-1). shift(-1) is next step (t+1).
        # Config lags: positive int usually means t-k (past), negative t+k (future)?
        # Or standard notation: lag k is t-k.
        # Let's assume Config list integers are passed directly to shift().
        # shift(k): positive k gets value from previous rows (past).
        # shift(-k): negative k gets value from future rows.

        group_cols = ["game_play", "nfl_player_id_1", "nfl_player_id_2"]

        # Apply Energy Lags
        df_merged = self._add_lags(
            df_merged, group_cols, energy_feats + geo_feats, Config.LAGS_ENERGY
        )

        # Apply Visual Lags (Sparse)
        df_merged = self._add_lags(df_merged, group_cols, vis_feats, Config.LAGS_VISUAL)

        # 8. Select Final Columns
        # Base columns (lag 0) + Lagged columns
        # Config.STREAM_A_FEATURES defines the logical groups, we need to flatten
        final_cols = []

        # Add base features (lag 0) if 0 in lags
        base_features = energy_feats + geo_feats + vis_feats

        # Collect all generated columns
        for col in df_merged.columns:
            # Check if it's a feature or a lagged feature
            # Simple check: starts with one of the base names
            is_feat = False
            for base in base_features:
                if col == base or col.startswith(f"{base}_lag"):
                    is_feat = True
                    break
            if is_feat:
                final_cols.append(col)

        X = df_merged[final_cols].copy()
        y = (
            df_merged["contact"].values
            if "contact" in df_merged.columns
            else np.zeros(len(df_merged))
        )
        ids = df_merged["contact_id"].values

        # Save Cache
        X.to_parquet(cache_path, index=False)
        self._save_ids_y(df_merged, "features_streamA", data_hash)

        return X, y, ids

    def process_stream_b(self, df_labels, df_tracking, load_cached_data=True):
        """
        Generates features for Stream B (Impact: Player vs Ground).
        Features: Finite-Difference Ego-Dynamics, Strict Invariance.
        """
        # 1. Generate Hash
        input_summary = {
            "labels_len": len(df_labels),
            "tracking_len": len(df_tracking),
            "first_contact": (
                str(df_labels.iloc[0]["contact_id"]) if not df_labels.empty else ""
            ),
            "config": Config.STREAM_B_FEATURES,
            "lags": Config.LAGS_ENERGY,
        }
        data_hash = generate_hash(input_summary)
        cache_path = self._get_cache_path("features_streamB_X", data_hash)

        # 2. Load Cache
        if load_cached_data and os.path.exists(cache_path):
            ids, y = self._load_ids_y("features_streamB", data_hash)
            if ids is not None:
                X = pd.read_parquet(cache_path)
                return X, y, ids

        print(f"Generating Stream B features (Hash: {data_hash})...")

        # 3. Preprocessing
        relevant_plays = df_labels["game_play"].unique()
        df_track = df_tracking[df_tracking["game_play"].isin(relevant_plays)].copy()

        df_labels["nfl_player_id_1"] = df_labels["nfl_player_id_1"].astype(str)
        df_track["nfl_player_id"] = df_track["nfl_player_id"].astype(str)

        # 4. Merge Tracking for P1
        df_merged = pd.merge(
            df_labels,
            df_track,
            left_on=["game_play", "step", "nfl_player_id_1"],
            right_on=["game_play", "step", "nfl_player_id"],
            how="left",
        )

        # Fill missing
        track_cols = ["speed", "acceleration", "direction", "orientation"]
        for col in track_cols:
            df_merged[col] = df_merged[col].fillna(0)

        # 5. Compute Ego-Dynamics
        # Convert to radians
        # Orientation: 0-360.
        # Direction: 0-360.
        # Theta = Direction - Orientation
        # If Theta = 0, moving forward (Surge).
        # If Theta = 90, moving right (Sway).

        theta_rad = np.radians(df_merged["direction"] - df_merged["orientation"])

        df_merged["v_surge"] = df_merged["speed"] * np.cos(theta_rad)
        df_merged["v_sway"] = df_merged["speed"] * np.sin(theta_rad)

        # Finite Difference for Ego-Acceleration and Jerk
        # We need to sort first
        df_merged.sort_values(by=["game_play", "nfl_player_id_1", "step"], inplace=True)
        group_cols = ["game_play", "nfl_player_id_1"]

        # Group ID for masking
        group_ids = (df_merged["game_play"] + "_" + df_merged["nfl_player_id_1"]).values

        def diff_col(col_name):
            # Calculate difference between t and t-1
            # 0.1s timestep
            vals = df_merged[col_name].values
            shifted = df_merged[col_name].shift(1).fillna(0).values
            shifted_groups = pd.Series(group_ids).shift(1).values

            mask = group_ids == shifted_groups
            diff = np.where(mask, vals - shifted, 0.0)
            return diff / 0.1  # per second

        df_merged["ego_acc_surge"] = diff_col("v_surge")
        df_merged["ego_acc_sway"] = diff_col("v_sway")

        df_merged["ego_jerk_surge"] = diff_col("ego_acc_surge")
        df_merged["ego_jerk_sway"] = diff_col("ego_acc_sway")

        # 6. Temporal Lags
        # Features to lag: Scalars + Ego Dynamics
        feats = [
            "speed",
            "acceleration",
            "v_surge",
            "v_sway",
            "ego_acc_surge",
            "ego_acc_sway",
            "ego_jerk_surge",
            "ego_jerk_sway",
        ]

        df_merged = self._add_lags(df_merged, group_cols, feats, Config.LAGS_ENERGY)

        # 7. Select Final Columns
        final_cols = []
        for col in df_merged.columns:
            is_feat = False
            for base in feats:
                if col == base or col.startswith(f"{base}_lag"):
                    is_feat = True
                    break
            if is_feat:
                final_cols.append(col)

        X = df_merged[final_cols].copy()
        y = (
            df_merged["contact"].values
            if "contact" in df_merged.columns
            else np.zeros(len(df_merged))
        )
        ids = df_merged["contact_id"].values

        # Save Cache
        X.to_parquet(cache_path, index=False)
        self._save_ids_y(df_merged, "features_streamB", data_hash)

        return X, y, ids
