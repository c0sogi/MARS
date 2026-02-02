import os
import pandas as pd
import numpy as np
import gc
from library.config import Config
from library.utils import get_config_hash, validate_schema
from library.data_loader import DataLoader


class FeatureEngineer:
    """
    Implements the Scale-Aligned Dual-Stream feature engineering pipeline.
    Generates two distinct datasets:
    - Stream A: Interaction Model (Player-Player) with Scale-Aligned Residuals.
    - Stream B: Impact Model (Player-Ground) with Rotational Energy Profiles.
    """

    def __init__(self):
        self.loader = DataLoader()
        self.cache_dir = Config.CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

    def process_data(
        self, mode: str, load_cached_data: bool = True, debug: bool = False
    ):
        """
        Main entry point for feature engineering.
        Loads raw data, computes features for both streams, and handles caching.

        Args:
            mode (str): 'train', 'validation', or 'test'.
            load_cached_data (bool): Whether to try loading from cache.
            debug (bool): Debug mode for faster execution on subsets.

        Returns:
            dict: Dictionary containing X, y, ids for both streams.
                  keys: 'stream_a', 'stream_b'
                  values: dict with 'X', 'y', 'ids'
        """
        # 1. Generate Cache Paths
        config_hash = get_config_hash()
        debug_suffix = "_debug" if debug else ""

        cache_paths = {
            "stream_a": {
                "X": os.path.join(
                    self.cache_dir,
                    f"features_{mode}_{config_hash}{debug_suffix}_streamA_X.parquet",
                ),
                "y": os.path.join(
                    self.cache_dir,
                    f"features_{mode}_{config_hash}{debug_suffix}_streamA_y.npy",
                ),
                "ids": os.path.join(
                    self.cache_dir,
                    f"features_{mode}_{config_hash}{debug_suffix}_streamA_ids.npy",
                ),
            },
            "stream_b": {
                "X": os.path.join(
                    self.cache_dir,
                    f"features_{mode}_{config_hash}{debug_suffix}_streamB_X.parquet",
                ),
                "y": os.path.join(
                    self.cache_dir,
                    f"features_{mode}_{config_hash}{debug_suffix}_streamB_y.npy",
                ),
                "ids": os.path.join(
                    self.cache_dir,
                    f"features_{mode}_{config_hash}{debug_suffix}_streamB_ids.npy",
                ),
            },
        }

        # 2. Try Loading from Cache
        if load_cached_data:
            if self._check_cache_exists(cache_paths):
                print(
                    f"[{mode.upper()}] Loading features from cache ({config_hash})..."
                )
                return self._load_cache(cache_paths)
            else:
                print(
                    f"[{mode.upper()}] Cache miss or partial. Recomputing features..."
                )

        # 3. Load Raw Data
        df_labels, df_tracking, df_helmets, _ = self.loader.load_data(
            mode, load_cached_data=load_cached_data, debug=debug
        )

        # 4. Preprocessing & Splitting
        # Ensure sorting for temporal operations
        df_labels = df_labels.sort_values(by=["game_play", "step"]).reset_index(
            drop=True
        )

        # Split into Stream A (Player-Player) and Stream B (Player-Ground)
        mask_ground = df_labels["nfl_player_id_2"] == "G"
        df_a = df_labels[~mask_ground].copy()
        df_b = df_labels[mask_ground].copy()

        print(f"[{mode.upper()}] Stream A (Interaction) Samples: {len(df_a)}")
        print(f"[{mode.upper()}] Stream B (Impact) Samples: {len(df_b)}")

        # 5. Build Features
        data_a = self._build_stream_a_features(df_a, df_tracking, df_helmets)
        data_b = self._build_stream_b_features(
            df_b, df_tracking
        )  # Helmets not used for B

        # 6. Save to Cache
        if not debug:  # Don't cache debug runs to avoid pollution
            self._save_cache(data_a, cache_paths["stream_a"])
            self._save_cache(data_b, cache_paths["stream_b"])

        return {"stream_a": data_a, "stream_b": data_b}

    def _build_stream_a_features(self, df_labels, df_tracking, df_helmets):
        """
        Constructs features for Stream A (Interaction Model).
        Key Innovation: Scale-Aligned Cross-Modal Residuals.
        """
        if df_labels.empty:
            return {"X": pd.DataFrame(), "y": np.array([]), "ids": np.array([])}

        print("Building Stream A Features (Interaction)...")

        # Standardize IDs to strings to prevent merge errors (Cite debug_lesson_13)
        df_labels["nfl_player_id_1"] = df_labels["nfl_player_id_1"].astype(str)
        df_labels["nfl_player_id_2"] = df_labels["nfl_player_id_2"].astype(str)

        df_tracking_str = df_tracking.copy()
        df_tracking_str["nfl_player_id"] = df_tracking_str["nfl_player_id"].astype(str)

        # --- A. Merge Tracking Data ---
        # Player 1
        df_merged = pd.merge(
            df_labels,
            df_tracking_str.add_suffix("_p1"),
            left_on=["game_play", "step", "nfl_player_id_1"],
            right_on=["game_play_p1", "step_p1", "nfl_player_id_p1"],
            how="left",
        )
        # Player 2
        df_merged = pd.merge(
            df_merged,
            df_tracking_str.add_suffix("_p2"),
            left_on=["game_play", "step", "nfl_player_id_2"],
            right_on=["game_play_p2", "step_p2", "nfl_player_id_p2"],
            how="left",
        )

        # --- B. Merge Visual Data (Helmets) ---
        # Map step to frame: Snap (step 0) is frame 300. 59.94fps approx.
        # Formula: frame = 300 + step * (59.94 / 10)
        df_merged["frame_approx"] = (
            (300 + df_merged["step"] * 5.994).round().astype(int)
        )

        # Helper to merge specific view
        def merge_view(df, view_name):
            view_helmets = df_helmets[df_helmets["view"] == view_name].copy()
            view_helmets["nfl_player_id"] = view_helmets["nfl_player_id"].astype(str)

            # Merge P1
            df = pd.merge(
                df,
                view_helmets.add_suffix(f"_{view_name}_p1"),
                left_on=["game_play", "frame_approx", "nfl_player_id_1"],
                right_on=["game_play", "frame", "nfl_player_id"],
                how="left",
            )
            # Merge P2
            df = pd.merge(
                df,
                view_helmets.add_suffix(f"_{view_name}_p2"),
                left_on=["game_play", "frame_approx", "nfl_player_id_2"],
                right_on=["game_play", "frame", "nfl_player_id"],
                how="left",
            )
            return df

        df_merged = merge_view(df_merged, "Sideline")
        df_merged = merge_view(df_merged, "Endzone")

        # --- C. Feature Calculation ---

        # 1. Physical Distance & Closure
        dx = df_merged["x_position_p1"] - df_merged["x_position_p2"]
        dy = df_merged["y_position_p1"] - df_merged["y_position_p2"]
        df_merged["dist"] = np.sqrt(dx**2 + dy**2).fillna(
            100
        )  # Fill NaNs with large distance

        # 2. Visual IoU (Consensus)
        def calc_iou(row, view):
            # Extract box coords
            try:
                l1, w1, t1, h1 = (
                    row[f"left_{view}_p1"],
                    row[f"width_{view}_p1"],
                    row[f"top_{view}_p1"],
                    row[f"height_{view}_p1"],
                )
                l2, w2, t2, h2 = (
                    row[f"left_{view}_p2"],
                    row[f"width_{view}_p2"],
                    row[f"top_{view}_p2"],
                    row[f"height_{view}_p2"],
                )

                if pd.isna(l1) or pd.isna(l2):
                    return 0.0

                x_left = max(l1, l2)
                y_top = max(t1, t2)
                x_right = min(l1 + w1, l2 + w2)
                y_bottom = min(t1 + h1, t2 + h2)

                if x_right < x_left or y_bottom < y_top:
                    return 0.0

                intersection = (x_right - x_left) * (y_bottom - y_top)
                area1 = w1 * h1
                area2 = w2 * h2
                return intersection / (area1 + area2 - intersection)
            except:
                return 0.0

        # Vectorized IoU calculation is complex due to NaNs, using apply for clarity/safety given data size isn't massive
        # Optimization: Use numpy arrays for speed
        for view in ["Sideline", "Endzone"]:
            l1 = df_merged[f"left_{view}_p1"].values
            w1 = df_merged[f"width_{view}_p1"].values
            t1 = df_merged[f"top_{view}_p1"].values
            h1 = df_merged[f"height_{view}_p1"].values

            l2 = df_merged[f"left_{view}_p2"].values
            w2 = df_merged[f"width_{view}_p2"].values
            t2 = df_merged[f"top_{view}_p2"].values
            h2 = df_merged[f"height_{view}_p2"].values

            # Intersection
            x_left = np.maximum(l1, l2)
            y_top = np.maximum(t1, t2)
            x_right = np.minimum(l1 + w1, l2 + w2)
            y_bottom = np.minimum(t1 + h1, t2 + h2)

            inter_w = np.maximum(0, x_right - x_left)
            inter_h = np.maximum(0, y_bottom - y_top)
            intersection = inter_w * inter_h

            area1 = w1 * h1
            area2 = w2 * h2
            union = area1 + area2 - intersection

            iou = np.divide(
                intersection, union, out=np.zeros_like(intersection), where=union != 0
            )
            df_merged[f"iou_{view}"] = np.nan_to_num(iou)

        df_merged["max_iou"] = df_merged[["iou_Sideline", "iou_Endzone"]].max(axis=1)

        # --- D. Scale-Aligned Residuals (Temporal) ---
        # Sort for rolling ops
        df_merged = df_merged.sort_values(
            by=["game_play", "nfl_player_id_1", "nfl_player_id_2", "step"]
        )

        # Group key for temporal ops
        group_cols = ["game_play", "nfl_player_id_1", "nfl_player_id_2"]

        # Calculate Rates (Finite Difference)
        # Closure Rate: Positive means getting closer (distance decreasing)
        # Looming Rate: Positive means visual overlap increasing
        df_merged["dist_diff"] = df_merged.groupby(group_cols)["dist"].diff().fillna(0)
        df_merged["closure_rate"] = (
            -1 * df_merged["dist_diff"]
        )  # Negate so closing is positive

        df_merged["iou_diff"] = (
            df_merged.groupby(group_cols)["max_iou"].diff().fillna(0)
        )
        df_merged["looming_rate"] = df_merged["iou_diff"]

        # Dynamic Scaling (Rolling Z-Score)
        window = Config.WINDOW_SIZE

        def rolling_zscore(x):
            r = x.rolling(window=window, min_periods=1)
            m = r.mean()
            s = r.std().replace(0, 1)  # Avoid div by zero
            return (x - m) / s

        # Apply transformation
        # Note: transform with rolling is efficient
        df_merged["z_closure"] = (
            df_merged.groupby(group_cols)["closure_rate"]
            .transform(rolling_zscore)
            .fillna(0)
        )
        df_merged["z_looming"] = (
            df_merged.groupby(group_cols)["looming_rate"]
            .transform(rolling_zscore)
            .fillna(0)
        )

        # The Innovation: Scale-Aligned Residual
        df_merged["scale_aligned_residual"] = (
            df_merged["z_closure"] - df_merged["z_looming"]
        )

        # --- E. Flattening & Feature Selection ---
        feature_cols = [
            "dist",
            "closure_rate",
            "max_iou",
            "looming_rate",
            "scale_aligned_residual",
            "speed_p1",
            "acceleration_p1",
            "speed_p2",
            "acceleration_p2",
        ]

        # Create Lag Features
        lags = Config.VISUAL_CONSENSUS_LAGS  # e.g., [0, 4, 8, 15]

        final_feats = []
        for col in feature_cols:
            for lag in lags:
                feat_name = f"{col}_lag{lag}"
                # Shift positive looks back in time (lag)
                df_merged[feat_name] = (
                    df_merged.groupby(group_cols)[col].shift(lag).fillna(0)
                )
                final_feats.append(feat_name)

        # Identifiers and Target
        ids = df_merged["contact_id"].values
        y = df_merged["contact"].values
        X = df_merged[final_feats].copy()

        # Validate Schema
        validate_schema(X, final_feats)

        return {"X": X, "y": y, "ids": ids}

    def _build_stream_b_features(self, df_labels, df_tracking):
        """
        Constructs features for Stream B (Impact Model / Player-Ground).
        Key Innovation: Rotational Energy Profiles & Hybrid Context.
        Explicitly BLOCKS visual features.
        """
        if df_labels.empty:
            return {"X": pd.DataFrame(), "y": np.array([]), "ids": np.array([])}

        print("Building Stream B Features (Impact)...")

        # --- A. Merge Tracking (Player 1 only) ---
        df_merged = pd.merge(
            df_labels,
            df_tracking,
            left_on=["game_play", "step", "nfl_player_id_1"],
            right_on=["game_play", "step", "nfl_player_id"],
            how="left",
        )

        # Fill NaNs in tracking (rare but possible)
        num_cols = [
            "x_position",
            "y_position",
            "speed",
            "acceleration",
            "direction",
            "orientation",
            "sa",
        ]
        df_merged[num_cols] = df_merged[num_cols].fillna(0)

        # --- B. Rotational Energy & Ego-Dynamics ---
        # Convert angles to radians
        # NFL Data: 0 is North (Y), 90 is East (X).
        # Orientation (O) and Direction (D).
        # Relative Angle = D - O.

        rad_dir = np.radians(df_merged["direction"])
        rad_orient = np.radians(df_merged["orientation"])

        # Project Speed onto Orientation (Surge) and Perpendicular (Sway)
        # Surge = Speed * cos(Dir - Orient)
        # Sway = Speed * sin(Dir - Orient)

        relative_angle = rad_dir - rad_orient
        df_merged["v_surge"] = df_merged["speed"] * np.cos(relative_angle)
        df_merged["v_sway"] = df_merged["speed"] * np.sin(relative_angle)

        # Rotational Energy (Sway Energy)
        df_merged["energy_sway"] = 0.5 * (df_merged["v_sway"] ** 2)

        # Ego Jerk (Derivative of Acceleration)
        # Sort first
        df_merged = df_merged.sort_values(by=["game_play", "nfl_player_id", "step"])
        group_cols = ["game_play", "nfl_player_id"]

        df_merged["jerk"] = (
            df_merged.groupby(group_cols)["acceleration"].diff().fillna(0)
        )

        # --- C. Flattening (Hybrid Context) ---
        feature_cols = [
            "speed",
            "acceleration",
            "jerk",
            "v_surge",
            "v_sway",
            "energy_sway",
            "orientation",
            "direction",
            "sa",
        ]

        # Use a dense window for Stream B as ground impacts are transient/dynamic
        # Flatten window t-5 to t+5 (1 second context)
        # Config.WINDOW_SIZE is 15, let's use lags 0, 2, 5, 10 for profile shape
        lags = [0, 2, 5, 10]

        final_feats = []
        for col in feature_cols:
            for lag in lags:
                feat_name = f"{col}_lag{lag}"
                df_merged[feat_name] = (
                    df_merged.groupby(group_cols)[col].shift(lag).fillna(0)
                )
                final_feats.append(feat_name)

        # Identifiers and Target
        ids = df_merged["contact_id"].values
        y = df_merged["contact"].values
        X = df_merged[final_feats].copy()

        # Validate Schema
        validate_schema(X, final_feats)

        return {"X": X, "y": y, "ids": ids}

    def _check_cache_exists(self, paths):
        for stream in paths:
            for key in paths[stream]:
                if not os.path.exists(paths[stream][key]):
                    return False
        return True

    def _load_cache(self, paths):
        result = {}
        for stream in ["stream_a", "stream_b"]:
            result[stream] = {
                "X": pd.read_parquet(paths[stream]["X"]),
                "y": np.load(paths[stream]["y"]),
                "ids": np.load(paths[stream]["ids"]),
            }
        return result

    def _save_cache(self, data, paths):
        data["X"].to_parquet(paths["X"], index=False)
        np.save(paths["y"], data["y"])
        np.save(paths["ids"], data["ids"])
