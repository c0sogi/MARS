import pandas as pd
import numpy as np
import os
import gc
from library.config import Config
from library.utils import Timer


class HelmetFeatureGenerator:
    """
    Generates visual-geometric features from helmet bounding box data for Stream B.
    Handles frame-to-step synchronization, view aggregation (Sideline/Endzone),
    and geometric interaction computation (IoU, Centroid Distance).
    """

    def __init__(self):
        self.config = Config
        self.cache_dir = self.config.WORKING_DIR
        self.micro_window = self.config.WINDOW_MICRO
        # Frame-to-Step conversion constants
        self.FPS = 59.94
        self.STEP_FREQ = 10.0
        self.FRAMES_TO_SNAP = 300  # 5 seconds * 59.94 fps

    def generate_features(
        self, split: str, load_cached_data: bool = True
    ) -> pd.DataFrame:
        """
        Main entry point to generate helmet features for a specific split.

        Args:
            split (str): One of 'train', 'validation', 'test'.
            load_cached_data (bool): Whether to try loading from cache.

        Returns:
            pd.DataFrame: Feature matrix including contact_id and geometric features.
        """
        cache_path = os.path.join(self.cache_dir, f"{split}_helmet_features.parquet")

        # 1. Check Cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"[Helmets] Loading cached features from {cache_path}")
            return pd.read_parquet(cache_path)

        print(f"[Helmets] Generating features for {split}...")

        # 2. Load Metadata
        if split == "train":
            meta_path = self.config.TRAIN_META_PATH
        elif split == "validation":
            meta_path = self.config.VAL_META_PATH
        elif split == "test":
            meta_path = self.config.TEST_META_PATH
        else:
            raise ValueError(f"Unknown split: {split}")

        df_meta = pd.read_csv(meta_path)

        # Filter to relevant plays to optimize helmet loading
        relevant_plays = df_meta["game_play"].unique()

        # 3. Load and Preprocess Helmet Data
        # Determine file path
        if split == "test":
            helmets_path = self.config.TEST_HELMETS_PATH
        else:
            helmets_path = self.config.TRAIN_HELMETS_PATH

        with Timer("Process Helmet Data"):
            df_helmets = self._process_helmet_data(helmets_path, relevant_plays)

        # 4. Compute Interactions and Aggregate Views
        with Timer("Compute Interactions & Aggregate"):
            df_features = self._compute_interactions_and_aggregate(df_meta, df_helmets)

        # 5. Temporal Windowing
        with Timer("Temporal Windowing"):
            df_features = self._apply_windowing(df_features)

        # 6. Save to Cache
        print(f"[Helmets] Saving features to {cache_path}")
        df_features.to_parquet(cache_path, index=False)

        # Cleanup
        del df_helmets
        gc.collect()

        return df_features

    def _process_helmet_data(
        self, helmets_path: str, relevant_plays: np.array
    ) -> pd.DataFrame:
        """
        Loads helmet data, filters by play, maps frames to steps, and computes basic box features.
        """
        # Load specific columns
        use_cols = [
            "game_play",
            "view",
            "frame",
            "nfl_player_id",
            "left",
            "top",
            "width",
            "height",
        ]

        df = pd.read_csv(helmets_path, usecols=lambda c: c in use_cols)

        # Filter plays
        df = df[df["game_play"].isin(relevant_plays)].copy()

        # Map Frames to Steps (10Hz)
        # Formula: Step = round((Frame - 300) / 5.994)
        df["step"] = np.round(
            (df["frame"] - self.FRAMES_TO_SNAP) / (self.FPS / self.STEP_FREQ)
        ).astype(int)

        # Filter invalid steps (negative steps are pre-snap, usually not relevant for contact)
        # However, we keep a small buffer if needed, but generally step >= 0
        # We'll allow negative steps if they exist in metadata, but usually tracking starts at 0.
        # Let's just keep them all for now and filter by inner join later.

        # Deduplicate: Multiple frames map to the same step.
        # We keep the frame closest to the exact step time.
        # Calculate exact step float
        df["step_exact"] = (df["frame"] - self.FRAMES_TO_SNAP) / (
            self.FPS / self.STEP_FREQ
        )
        df["dist_to_step"] = np.abs(df["step"] - df["step_exact"])

        # Sort by distance and keep first (closest frame to the integer step)
        df.sort_values("dist_to_step", inplace=True)
        df.drop_duplicates(
            subset=["game_play", "view", "nfl_player_id", "step"],
            keep="first",
            inplace=True,
        )

        # Compute Basic Geometric Features
        # Centroids
        df["centroid_x"] = df["left"] + df["width"] / 2
        df["centroid_y"] = df["top"] + df["height"] / 2
        # Area
        df["area"] = df["width"] * df["height"]

        # Select columns for merge
        keep_cols = [
            "game_play",
            "view",
            "step",
            "nfl_player_id",
            "left",
            "top",
            "width",
            "height",
            "centroid_x",
            "centroid_y",
            "area",
        ]
        return df[keep_cols]

    def _compute_interactions_and_aggregate(
        self, df_meta: pd.DataFrame, df_helmets: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merges helmet data for P1 and P2, computes IoU/Distance per view,
        and aggregates across views (Max/Min).
        """
        # Prepare Metadata
        # Ensure IDs are numeric
        df_meta = df_meta.copy()
        df_meta["nfl_player_id_1"] = pd.to_numeric(
            df_meta["nfl_player_id_1"], errors="coerce"
        )
        df_meta["nfl_player_id_2"] = pd.to_numeric(
            df_meta["nfl_player_id_2"], errors="coerce"
        )

        # 1. Merge P1 Helmets
        # We join on (game_play, step, nfl_player_id_1).
        # Note: df_helmets has 'view'. This merge will explode df_meta by the number of views available.

        # Rename helmet cols for P1
        p1_cols = [
            c for c in df_helmets.columns if c not in ["game_play", "step", "view"]
        ]
        rename_p1 = {c: f"{c}_p1" for c in p1_cols}
        rename_p1["nfl_player_id"] = "nfl_player_id_1"

        df_p1 = df_helmets.rename(columns=rename_p1)

        # Inner join: We only care if we have visual data for P1 (at least)
        # If P1 is not visible, we can't compute visual features.
        df_merged = pd.merge(
            df_meta, df_p1, on=["game_play", "step", "nfl_player_id_1"], how="inner"
        )

        # 2. Merge P2 Helmets
        # Join on (game_play, step, view, nfl_player_id_2).
        # We must match the VIEW.

        rename_p2 = {c: f"{c}_p2" for c in p1_cols}
        rename_p2["nfl_player_id"] = "nfl_player_id_2"

        df_p2 = df_helmets.rename(columns=rename_p2)

        # Left join: P2 might be 'G' (Ground) or not visible/missing in that view.
        df_merged = pd.merge(
            df_merged,
            df_p2,
            on=["game_play", "step", "view", "nfl_player_id_2"],
            how="left",
        )

        # 3. Compute Interactions (Vectorized)

        # --- IoU (Intersection over Union) ---
        # Coordinates
        x1_min = df_merged["left_p1"]
        y1_min = df_merged["top_p1"]
        x1_max = x1_min + df_merged["width_p1"]
        y1_max = y1_min + df_merged["height_p1"]

        x2_min = df_merged["left_p2"]
        y2_min = df_merged["top_p2"]
        x2_max = x2_min + df_merged["width_p2"]
        y2_max = y2_min + df_merged["height_p2"]

        # Intersection
        inter_x_min = np.maximum(x1_min, x2_min)
        inter_y_min = np.maximum(y1_min, y2_min)
        inter_x_max = np.minimum(x1_max, x2_max)
        inter_y_max = np.minimum(y1_max, y2_max)

        inter_w = np.maximum(0, inter_x_max - inter_x_min)
        inter_h = np.maximum(0, inter_y_max - inter_y_min)
        inter_area = inter_w * inter_h

        # Union
        area1 = df_merged["area_p1"]
        area2 = df_merged["area_p2"]
        union_area = area1 + area2 - inter_area

        # IoU (Handle divide by zero or NaN P2)
        df_merged["iou"] = np.where(
            (union_area > 0) & (df_merged["area_p2"].notna()),
            inter_area / union_area,
            0.0,
        )

        # --- Centroid Distance ---
        # Euclidean distance in pixel space
        d_x = df_merged["centroid_x_p1"] - df_merged["centroid_x_p2"]
        d_y = df_merged["centroid_y_p1"] - df_merged["centroid_y_p2"]

        # If P2 is missing, distance is large/undefined. We fill with a large number or NaN.
        # We'll use NaN and handle in aggregation.
        df_merged["dist_centroids"] = np.sqrt(d_x**2 + d_y**2)

        # --- Area Ratio ---
        # Ratio of P2 area to P1 area (Visual size comparison)
        df_merged["area_ratio"] = np.where(
            (df_merged["area_p1"] > 0) & (df_merged["area_p2"].notna()),
            df_merged["area_p2"] / df_merged["area_p1"],
            0.0,
        )

        # 4. Aggregate Across Views
        # Group by the unique contact identifier (and step)
        # We want to take the "strongest" signal from either view.
        # IoU: Max
        # Dist: Min (Closest visual proximity)
        # Area: Max (Best visibility)

        group_cols = [
            "contact_id",
            "game_play",
            "step",
            "nfl_player_id_1",
            "nfl_player_id_2",
        ]
        if "contact" in df_merged.columns:
            group_cols.append("contact")

        # Define aggregations
        aggs = {
            "iou": "max",
            "dist_centroids": "min",
            "area_ratio": "max",
            "area_p1": "max",
            "area_p2": "max",  # If P2 is G/missing, this will be NaN -> 0
        }

        df_agg = df_merged.groupby(group_cols, as_index=False).agg(aggs)

        # Fill NaNs resulting from missing P2 or no views
        df_agg.fillna(0, inplace=True)

        # For distance, if 0 (filled) but IoU is 0, it implies missing P2.
        # However, min(NaN) is NaN -> filled 0.
        # If P2 is missing, distance 0 is misleading.
        # But for tree models, consistency matters.
        # If P2 is 'G', dist is 0. This effectively tells the model "P2 is not a visual object".

        return df_agg

    def _apply_windowing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies temporal windowing (lags) to the aggregated features.
        """
        # Sort for correct shifting
        # Primary sort: Game, then Pair, then Step
        df.sort_values(
            ["game_play", "nfl_player_id_1", "nfl_player_id_2", "step"], inplace=True
        )

        features_to_lag = ["iou", "dist_centroids", "area_ratio", "area_p1", "area_p2"]

        grouped = df.groupby(["game_play", "nfl_player_id_1", "nfl_player_id_2"])

        lag_cols = {}
        for lag in range(-self.micro_window, self.micro_window + 1):
            if lag == 0:
                continue  # Keep original columns as is

            suffix = f"_lag{lag}"
            shifted = grouped[features_to_lag].shift(lag)

            for col in features_to_lag:
                lag_cols[f"{col}{suffix}"] = shifted[col]

        # Concatenate lags
        df_lags = pd.DataFrame(lag_cols, index=df.index)
        df_final = pd.concat([df, df_lags], axis=1)

        # Fill NaNs at edges
        df_final.fillna(0, inplace=True)

        return df_final
