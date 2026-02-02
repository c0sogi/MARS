import pandas as pd
import numpy as np
import os
from library.config import Config
from library.data_loader import load_metadata, load_tracking, load_helmets
from library.utils import use_config_cache


class FeatureEngine:
    def __init__(self, split_name):
        self.split_name = split_name
        self.df_meta = load_metadata(split_name)
        self.df_track = load_tracking(split_name)
        self.df_helmets = load_helmets(split_name)

        # Pre-calculate common mappings
        self._prepare_tracking()
        self._prepare_helmets()

    def _prepare_tracking(self):
        """
        Pre-calculates dynamics and individual lags on the continuous tracking data.
        """
        df = self.df_track.copy()

        # --- Ego Dynamics ---
        # Convert degrees to radians (Tracking: 0 is usually Y-axis or X-axis depending on convention,
        # but relative difference is what matters for surge/sway)
        # Assuming standard mathematical angle or consistent NFL tracking convention.
        # Direction: movement angle. Orientation: facing angle.

        rad_dir = np.radians(df["direction"])
        rad_orient = np.radians(df["orientation"])

        # Angle difference
        theta = rad_dir - rad_orient

        # Surge (velocity projected onto orientation)
        df["v_surge"] = df["speed"] * np.cos(theta)

        # Sway (velocity perpendicular to orientation)
        df["v_sway"] = df["speed"] * np.sin(theta)

        # Rotational Energy
        df["rotational_energy"] = df["v_sway"] ** 2

        # Ego Jerk (Derivative of acceleration)
        # Sort by play and player and step to ensure correct diff
        df = df.sort_values(["game_play", "nfl_player_id", "step"])

        # Calculate delta acceleration / delta time (0.1s)
        # We use a simple finite difference
        df["ego_jerk"] = (
            df.groupby(["game_play", "nfl_player_id"])["acceleration"].diff().fillna(0)
            * 10.0
        )

        # --- Generate Individual Lags ---
        # These are used for Stream B and Stream A (System Energy)
        # We process lags here on the full tracking set to ensure continuity
        features_to_lag = [
            "x_position",
            "y_position",
            "speed",
            "acceleration",
            "direction",
            "orientation",
            "v_surge",
            "v_sway",
            "rotational_energy",
            "ego_jerk",
        ]

        # Generate lags
        for lag in Config.LAGS:
            if lag == 0:
                continue
            suffix = f"_lag{lag}"
            shifted = df.groupby(["game_play", "nfl_player_id"])[features_to_lag].shift(
                lag
            )
            shifted.columns = [f"{c}{suffix}" for c in shifted.columns]
            df = pd.concat([df, shifted], axis=1)

        self.df_track_processed = df

    def _prepare_helmets(self):
        """
        Maps helmet boxes to steps and pivots views.
        """
        df = self.df_helmets.copy()

        # Map frame to step
        # Snap is at 5s (frame 300). 59.94 Hz.
        # step = (frame - 300) / (59.94 / 10) approx (frame - 300) / 6
        df["step"] = ((df["frame"] - 300) / 5.994).round().astype(int)

        # Filter relevant columns
        cols = [
            "game_play",
            "step",
            "nfl_player_id",
            "view",
            "left",
            "width",
            "top",
            "height",
        ]
        df = df[cols]

        # We need to compute IoU between pairs later.
        # To make lookup fast, we can't easily pivot everything because we need pairs.
        # We will index this by (game_play, step, nfl_player_id) for fast merging.
        # Since there are two views, we pivot views.

        df_pivot = df.pivot_table(
            index=["game_play", "step", "nfl_player_id"],
            columns="view",
            values=["left", "width", "top", "height"],
            aggfunc="first",  # Should be unique per view
        )

        # Flatten columns: left_Sideline, width_Sideline, etc.
        df_pivot.columns = [f"{c[0]}_{c[1]}" for c in df_pivot.columns]
        self.df_helmets_processed = df_pivot.reset_index()

    def _calculate_iou(self, box1, box2):
        """
        Vectorized IoU calculation.
        box: [left, width, top, height]
        """
        # Unpack
        l1, w1, t1, h1 = box1
        l2, w2, t2, h2 = box2

        r1 = l1 + w1
        b1 = t1 + h1
        r2 = l2 + w2
        b2 = t2 + h2

        # Intersection
        x_left = np.maximum(l1, l2)
        y_top = np.maximum(t1, t2)
        x_right = np.minimum(r1, r2)
        y_bottom = np.minimum(b1, b2)

        w_int = np.maximum(0, x_right - x_left)
        h_int = np.maximum(0, y_bottom - y_top)
        area_int = w_int * h_int

        area1 = w1 * h1
        area2 = w2 * h2

        iou = area_int / (area1 + area2 - area_int + 1e-6)
        return iou

    def _generate_stream_a(self):
        """
        Generates features for Interaction Model (Player vs Player).
        """
        # Filter for player contacts
        df = self.df_meta[self.df_meta["nfl_player_id_2"] != "G"].copy()

        # Merge Tracking P1
        track_cols = self.df_track_processed.columns
        df = df.merge(
            self.df_track_processed.add_suffix("_p1"),
            left_on=["game_play", "step", "nfl_player_id_1"],
            right_on=["game_play_p1", "step_p1", "nfl_player_id_p1"],
            how="left",
        )

        # Merge Tracking P2
        df = df.merge(
            self.df_track_processed.add_suffix("_p2"),
            left_on=["game_play", "step", "nfl_player_id_2"],
            right_on=["game_play_p2", "step_p2", "nfl_player_id_p2"],
            how="left",
        )

        # --- Relational Scalars (Current Time) ---
        dx = df["x_position_p1"] - df["x_position_p2"]
        dy = df["y_position_p1"] - df["y_position_p2"]
        df["distance"] = np.sqrt(dx**2 + dy**2)

        # --- Visual Consensus ---
        # Merge Helmets P1
        df = df.merge(
            self.df_helmets_processed.add_suffix("_p1"),
            left_on=["game_play", "step", "nfl_player_id_1"],
            right_on=["game_play_p1", "step_p1", "nfl_player_id_p1"],
            how="left",
        )
        # Merge Helmets P2
        df = df.merge(
            self.df_helmets_processed.add_suffix("_p2"),
            left_on=["game_play", "step", "nfl_player_id_2"],
            right_on=["game_play_p2", "step_p2", "nfl_player_id_p2"],
            how="left",
        )

        # Calculate IoU for Sideline
        box1_side = [
            df["left_Sideline_p1"],
            df["width_Sideline_p1"],
            df["top_Sideline_p1"],
            df["height_Sideline_p1"],
        ]
        box2_side = [
            df["left_Sideline_p2"],
            df["width_Sideline_p2"],
            df["top_Sideline_p2"],
            df["height_Sideline_p2"],
        ]
        iou_side = self._calculate_iou(box1_side, box2_side)

        # Calculate IoU for Endzone
        box1_end = [
            df["left_Endzone_p1"],
            df["width_Endzone_p1"],
            df["top_Endzone_p1"],
            df["height_Endzone_p1"],
        ]
        box2_end = [
            df["left_Endzone_p2"],
            df["width_Endzone_p2"],
            df["top_Endzone_p2"],
            df["height_Endzone_p2"],
        ]
        iou_end = self._calculate_iou(box1_end, box2_end)

        # Consensus
        df["max_iou"] = np.maximum(iou_side.fillna(0), iou_end.fillna(0))
        df["min_iou"] = np.minimum(iou_side.fillna(0), iou_end.fillna(0))
        df["iou_diff"] = np.abs(iou_side.fillna(0) - iou_end.fillna(0))

        # --- Temporal Calculations (Lags for Pair Features) ---
        # We need lags of distance and max_iou to calculate rates
        # Since df is a subset (labels), we must group and shift.
        # Ensure sorted
        df = df.sort_values(["game_play", "nfl_player_id_1", "nfl_player_id_2", "step"])

        # Lags for distance and iou
        pair_feats = ["distance", "max_iou", "min_iou", "iou_diff"]

        # We create lags for pair features directly on the labeled data
        # Assuming labels are dense enough (every 0.1s).
        # If there are gaps, shift will grab the previous row which might not be t-1.
        # However, for this competition, labels are generally provided for continuous sequences.

        grp = df.groupby(["game_play", "nfl_player_id_1", "nfl_player_id_2"])

        # Calculate Rates (Finite Difference)
        # Closure Rate: -d(distance)/dt
        # We use shift(1) for immediate rate
        prev_dist = grp["distance"].shift(1)
        # step is 0.1s.
        df["raw_closure_rate"] = -1.0 * (df["distance"] - prev_dist) * 10.0

        # Visual Looming Rate: d(max_iou)/dt
        prev_iou = grp["max_iou"].shift(1)
        df["visual_looming_rate"] = (df["max_iou"] - prev_iou) * 10.0

        # Scale-Aligned Consistency
        df["consistency_composite"] = (
            df["raw_closure_rate"] * Config.GLOBAL_SCALE_FACTOR
        ) - df["visual_looming_rate"]

        # Fill NaNs from shifting
        df[["raw_closure_rate", "visual_looming_rate", "consistency_composite"]] = df[
            ["raw_closure_rate", "visual_looming_rate", "consistency_composite"]
        ].fillna(0)

        # Now generate sparse lags for these pair features
        pair_feats_to_lag = [
            "distance",
            "max_iou",
            "min_iou",
            "iou_diff",
            "raw_closure_rate",
            "visual_looming_rate",
            "consistency_composite",
        ]

        for lag in Config.LAGS:
            if lag == 0:
                continue
            suffix = f"_lag{lag}"
            shifted = grp[pair_feats_to_lag].shift(lag)
            shifted.columns = [f"{c}{suffix}" for c in shifted.columns]
            df = pd.concat([df, shifted], axis=1)

        # --- Feature Selection ---
        # Collect all features
        # 1. Relational Lags (Distance, Closure)
        # 2. Consistency Lags
        # 3. Visual Consensus Lags
        # 4. System Energy (P1/P2 Speed/Accel Lags - already in df from merge)

        feature_cols = []

        # Add base pair features + lags
        for base in pair_feats_to_lag:
            feature_cols.append(base)
            for lag in Config.LAGS:
                if lag == 0:
                    continue
                feature_cols.append(f"{base}_lag{lag}")

        # Add P1/P2 System Energy (Speed, Accel) + Lags
        energy_feats = ["speed", "acceleration"]
        for p in ["p1", "p2"]:
            for base in energy_feats:
                col = f"{base}_{p}"
                feature_cols.append(col)
                for lag in Config.LAGS:
                    if lag == 0:
                        continue
                    feature_cols.append(f"{base}_lag{lag}_{p}")

        # Filter columns that exist
        feature_cols = [c for c in feature_cols if c in df.columns]

        # Extract X, IDs, y
        X = df[feature_cols].fillna(0).astype(np.float32)
        ids = df["contact_id"].values
        y = df["contact"].values if "contact" in df.columns else None

        return X, ids, y

    def _generate_stream_b(self):
        """
        Generates features for Impact Model (Player vs Ground).
        """
        # Filter for ground contacts
        df = self.df_meta[self.df_meta["nfl_player_id_2"] == "G"].copy()

        # Merge Tracking P1 (with precomputed lags)
        df = df.merge(
            self.df_track_processed,
            left_on=["game_play", "step", "nfl_player_id_1"],
            right_on=["game_play", "step", "nfl_player_id"],
            how="left",
        )

        # Features to include (Base + Lags)
        base_feats = [
            "x_position",
            "y_position",
            "speed",
            "acceleration",
            "direction",
            "orientation",
            "v_surge",
            "v_sway",
            "rotational_energy",
            "ego_jerk",
        ]

        feature_cols = []
        for base in base_feats:
            feature_cols.append(base)
            for lag in Config.LAGS:
                if lag == 0:
                    continue
                feature_cols.append(f"{base}_lag{lag}")

        # Filter columns
        feature_cols = [c for c in feature_cols if c in df.columns]

        X = df[feature_cols].fillna(0).astype(np.float32)
        ids = df["contact_id"].values
        y = df["contact"].values if "contact" in df.columns else None

        return X, ids, y

    def run(self):
        X_a, ids_a, y_a = self._generate_stream_a()
        X_b, ids_b, y_b = self._generate_stream_b()
        return X_a, ids_a, y_a, X_b, ids_b, y_b


# --- Wrapper Functions for Caching ---


@use_config_cache(
    cache_keys=[
        "train_stream_a",
        "train_stream_a_ids",
        "train_stream_a_y",
        "train_stream_b",
        "train_stream_b_ids",
        "train_stream_b_y",
    ]
)
def generate_train_features(load_cached_data=True):
    engine = FeatureEngine("train")
    return engine.run()


@use_config_cache(
    cache_keys=[
        "val_stream_a",
        "val_stream_a_ids",
        "val_stream_a_y",
        "val_stream_b",
        "val_stream_b_ids",
        "val_stream_b_y",
    ]
)
def generate_val_features(load_cached_data=True):
    engine = FeatureEngine("validation")
    return engine.run()


@use_config_cache(
    cache_keys=[
        "test_stream_a",
        "test_stream_a_ids",
        "test_stream_b",
        "test_stream_b_ids",
    ]
)
def generate_test_features(load_cached_data=True):
    # Test set doesn't have labels (y), so we modify the return signature
    engine = FeatureEngine("test")
    X_a, ids_a, _ = engine._generate_stream_a()
    X_b, ids_b, _ = engine._generate_stream_b()
    return X_a, ids_a, X_b, ids_b


def process_data(split_name, load_cached_data=True):
    """
    Main entry point for feature generation.
    """
    if split_name == "train":
        return generate_train_features(load_cached_data=load_cached_data)
    elif split_name == "validation":
        return generate_val_features(load_cached_data=load_cached_data)
    elif split_name == "test":
        return generate_test_features(load_cached_data=load_cached_data)
    else:
        raise ValueError(f"Unknown split: {split_name}")
