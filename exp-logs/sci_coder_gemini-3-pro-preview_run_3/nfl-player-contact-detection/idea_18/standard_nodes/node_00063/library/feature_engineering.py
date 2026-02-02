import pandas as pd
import numpy as np
import os
import gc
from library.config import Config
from library.utils import get_hash
from library.data_loader import DataLoader


class FeatureEngineer:
    """
    Implements the Biomechanical Dual-Stream feature engineering pipeline.
    Separates processing for Player-Player interactions (Stream A) and
    Player-Ground impacts (Stream B).
    """

    def __init__(self):
        self.config = Config
        self.loader = DataLoader()

    def _calculate_vector_components(self, speed, angle_deg):
        """
        Converts speed and angle (degrees, 0=North, CW) to Cartesian components.
        NFL tracking: 0 is Y-axis (North), 90 is X-axis (East).
        """
        theta = np.radians(angle_deg)
        # 0 deg -> +Y, 90 deg -> +X
        v_x = speed * np.sin(theta)
        v_y = speed * np.cos(theta)
        return v_x, v_y

    def _project_onto_orientation(self, v_x, v_y, orientation_deg):
        """
        Projects a vector (v_x, v_y) onto the orientation unit vector (Surge)
        and its orthogonal complement (Sway).
        """
        phi = np.radians(orientation_deg)
        # Orientation unit vector (Surge direction)
        o_x = np.sin(phi)
        o_y = np.cos(phi)

        # Orthogonal unit vector (Sway direction, 90 deg clockwise)
        # (cos, -sin)
        perp_x = np.cos(phi)
        perp_y = -np.sin(phi)

        surge = v_x * o_x + v_y * o_y
        sway = v_x * perp_x + v_y * perp_y

        return surge, sway

    def _create_lags(self, df, feature_cols, lags):
        """
        Generates lagged features for the specified columns using groupby.
        """
        # Sort by game_play and step to ensure correct shifting
        df = df.sort_values(["game_play", "step"])

        lagged_dfs = []
        # Keep the original columns (lag 0)
        # We will rename them later or keep as is, but usually we want explicit naming

        for lag in lags:
            if lag == 0:
                # Just copy the features
                tmp = df[feature_cols].copy()
                tmp.columns = [f"{col}_lag0" for col in feature_cols]
                lagged_dfs.append(tmp)
            else:
                # Shift
                shifted = df.groupby("game_play")[feature_cols].shift(
                    -lag
                )  # shift(-lag) because lag>0 usually means past?
                # Wait, standard notation: t-1 is past. shift(1) gets previous row.
                # Config lags are [-15, ... 15].
                # If lag is -15 (past), we want shift(15).
                # If lag is +15 (future), we want shift(-15).
                # So shift(-lag) is correct if lag is relative time index (t+lag).

                shifted.columns = [f"{col}_lag{lag}" for col in feature_cols]
                lagged_dfs.append(shifted)

        return pd.concat(lagged_dfs, axis=1)

    def _calculate_bbox_metrics(self, df, view):
        """
        Calculates IoU and Center Distance for P1 and P2 bounding boxes in a specific view.
        Returns Series for IoU and Distance.
        """
        # Columns: left_View_p1, width_View_p1, top_View_p1, height_View_p1
        l1 = df[f"left_{view}_p1"]
        w1 = df[f"width_{view}_p1"]
        t1 = df[f"top_{view}_p1"]
        h1 = df[f"height_{view}_p1"]

        l2 = df[f"left_{view}_p2"]
        w2 = df[f"width_{view}_p2"]
        t2 = df[f"top_{view}_p2"]
        h2 = df[f"height_{view}_p2"]

        # Fill missing with sentinel for calculation (will fix later or let sentinel propagate)
        # Actually, if missing, IoU is 0, Dist is large or sentinel.
        # Let's use 0 for missing coords to avoid errors, then mask.

        # Intersection
        x_left = np.maximum(l1, l2)
        y_top = np.maximum(t1, t2)
        x_right = np.minimum(l1 + w1, l2 + w2)
        y_bottom = np.minimum(t1 + h1, t2 + h2)

        intersection_area = np.maximum(0, x_right - x_left) * np.maximum(
            0, y_bottom - y_top
        )

        area1 = w1 * h1
        area2 = w2 * h2

        union_area = area1 + area2 - intersection_area

        # IoU
        iou = intersection_area / (union_area + 1e-6)  # avoid div by zero

        # Center Distance
        c1_x = l1 + w1 / 2
        c1_y = t1 + h1 / 2
        c2_x = l2 + w2 / 2
        c2_y = t2 + h2 / 2

        dist = np.sqrt((c1_x - c2_x) ** 2 + (c1_y - c2_y) ** 2)

        # Handle missing data (sentinel -999 in config)
        # If any bbox param is NaN, result should be sentinel
        mask = (
            l1.isna()
            | l2.isna()
            | (l1 == self.config.MISSING_VAL)
            | (l2 == self.config.MISSING_VAL)
        )

        iou = iou.mask(mask, self.config.MISSING_VAL)
        dist = dist.mask(mask, self.config.MISSING_VAL)

        return iou, dist

    def process_stream_a(self, df):
        """
        Stream A: Interaction Model (Player-Player).
        Features: Field-Relational, Ego-Relational, Visual.
        """
        print("Processing Stream A (Player-Player)...")

        # 1. Base Kinematics
        # Calculate P1 Velocity
        vx_p1, vy_p1 = self._calculate_vector_components(
            df["speed_p1"], df["direction_p1"]
        )
        # Calculate P2 Velocity
        vx_p2, vy_p2 = self._calculate_vector_components(
            df["speed_p2"], df["direction_p2"]
        )

        # 2. Field-Centric Relational
        # Euclidean Distance
        df["dist_p1_p2"] = np.sqrt(
            (df["x_position_p1"] - df["x_position_p2"]) ** 2
            + (df["y_position_p1"] - df["y_position_p2"]) ** 2
        )
        # Relative Speed Scalar
        df["rel_speed_scalar"] = df["speed_p1"] - df["speed_p2"]  # Simple diff

        # Explicit Interaction Primitives (Cite solution_lesson_node_00062)
        # Closure Rate: -d(Distance)/dt. Positive = Closing in.
        # Sort to ensure time-based diff is correct
        df = df.sort_values(["game_play", "step"])
        df["dist_diff"] = df.groupby("game_play")["dist_p1_p2"].diff().fillna(0)
        df["closure_rate"] = -df["dist_diff"] / 0.1

        # 3. Ego-Centric Relational (The Innovation)
        # Relative Velocity Vector
        rel_vx = vx_p2 - vx_p1
        rel_vy = vy_p2 - vy_p1

        # Project onto P1's orientation
        surge, sway = self._project_onto_orientation(
            rel_vx, rel_vy, df["orientation_p1"]
        )
        df["rel_surge"] = surge
        df["rel_sway"] = sway

        # 4. Visual Features
        # Sideline
        iou_side, dist_side = self._calculate_bbox_metrics(df, "Sideline")
        df["iou_sideline"] = iou_side
        df["dist_pixel_sideline"] = dist_side

        # Endzone
        iou_end, dist_end = self._calculate_bbox_metrics(df, "Endzone")
        df["iou_endzone"] = iou_end
        df["dist_pixel_endzone"] = dist_end

        # 5. Temporal Pyramids
        features_to_lag = [
            "dist_p1_p2",
            "closure_rate",
            "rel_speed_scalar",
            "rel_surge",
            "rel_sway",
            "iou_sideline",
            "dist_pixel_sideline",
            "iou_endzone",
            "dist_pixel_endzone",
            "acceleration_p1",
            "acceleration_p2",  # Added base accel as context
        ]

        # Fill NaNs in features before lagging (especially visual)
        df[features_to_lag] = df[features_to_lag].fillna(self.config.MISSING_VAL)

        X_lagged = self._create_lags(df, features_to_lag, self.config.LAGS)

        return X_lagged, df["contact"].values, df["contact_id"].values

    def process_stream_b(self, df):
        """
        Stream B: Impact Model (Player-Ground).
        Features: Field-Centric Kinematics, Ego-Centric Self-Motion.
        """
        print("Processing Stream B (Player-Ground)...")

        # 1. Base Kinematics
        vx_p1, vy_p1 = self._calculate_vector_components(
            df["speed_p1"], df["direction_p1"]
        )
        ax_p1, ay_p1 = self._calculate_vector_components(
            df["acceleration_p1"], df["direction_p1"]
        )  # Approx direction for accel

        # 2. Field-Centric Kinematics (Anchor)
        # We use raw positions, speed, accel, orientation

        # 3. Derived Kinematics: Jerk
        # Jerk is derivative of acceleration.
        # Since steps are 0.1s, Jerk ~ (Accel_t - Accel_{t-1}) / 0.1
        # We can approximate this using the lagged generation later or compute explicitly here.
        # Let's compute scalar jerk magnitude explicitly.
        # Sort first
        df = df.sort_values(["game_play", "step"])
        df["jerk_p1"] = df.groupby("game_play")["acceleration_p1"].diff() / 0.1
        df["jerk_p1"] = df["jerk_p1"].fillna(0)

        # 4. Ego-Centric Self-Motion (Augmentation)
        # Project P1's OWN velocity/accel onto P1's orientation

        # Velocity Projection
        v_surge, v_sway = self._project_onto_orientation(
            vx_p1, vy_p1, df["orientation_p1"]
        )
        df["self_v_surge"] = v_surge
        df["self_v_sway"] = v_sway

        # Acceleration Projection
        a_surge, a_sway = self._project_onto_orientation(
            ax_p1, ay_p1, df["orientation_p1"]
        )
        df["self_a_surge"] = a_surge
        df["self_a_sway"] = a_sway

        # Pose-Motion Alignment
        # Difference between orientation and direction
        df["pose_motion_diff"] = (df["orientation_p1"] - df["direction_p1"]) % 360
        df["pose_motion_diff"] = np.where(
            df["pose_motion_diff"] > 180,
            360 - df["pose_motion_diff"],
            df["pose_motion_diff"],
        )

        # 5. Temporal Pyramids
        features_to_lag = [
            "speed_p1",
            "acceleration_p1",
            "jerk_p1",
            "self_v_surge",
            "self_v_sway",
            "self_a_surge",
            "self_a_sway",
            "pose_motion_diff",
            "sa_p1",  # Signed acceleration
        ]

        df[features_to_lag] = df[features_to_lag].fillna(0)

        X_lagged = self._create_lags(df, features_to_lag, self.config.LAGS)

        return X_lagged, df["contact"].values, df["contact_id"].values

    def generate_features(self, mode, load_cached_data=True):
        """
        Main entry point. Loads data, splits into streams, processes, and returns features.
        Implements caching.
        """
        # Create cache keys
        cache_key_a = get_hash({"mode": mode, "stream": "A", "lags": self.config.LAGS})
        cache_key_b = get_hash({"mode": mode, "stream": "B", "lags": self.config.LAGS})

        path_X_a = os.path.join(
            self.config.WORKING_DIR, f"features_{mode}_streamA_{cache_key_a}_X.parquet"
        )
        path_y_a = os.path.join(
            self.config.WORKING_DIR, f"features_{mode}_streamA_{cache_key_a}_y.npy"
        )
        path_ids_a = os.path.join(
            self.config.WORKING_DIR, f"features_{mode}_streamA_{cache_key_a}_ids.npy"
        )

        path_X_b = os.path.join(
            self.config.WORKING_DIR, f"features_{mode}_streamB_{cache_key_b}_X.parquet"
        )
        path_y_b = os.path.join(
            self.config.WORKING_DIR, f"features_{mode}_streamB_{cache_key_b}_y.npy"
        )
        path_ids_b = os.path.join(
            self.config.WORKING_DIR, f"features_{mode}_streamB_{cache_key_b}_ids.npy"
        )

        # Check cache
        cache_exists = (
            os.path.exists(path_X_a)
            and os.path.exists(path_y_a)
            and os.path.exists(path_ids_a)
            and os.path.exists(path_X_b)
            and os.path.exists(path_y_b)
            and os.path.exists(path_ids_b)
        )

        if load_cached_data and cache_exists:
            print(f"Loading cached features for {mode}...")
            X_a = pd.read_parquet(path_X_a)
            y_a = np.load(path_y_a)
            ids_a = np.load(path_ids_a)

            X_b = pd.read_parquet(path_X_b)
            y_b = np.load(path_y_b)
            ids_b = np.load(path_ids_b)

            return (X_a, y_a, ids_a), (X_b, y_b, ids_b)

        # Process from scratch
        print(f"Generating features for {mode} from scratch...")

        # Load merged data
        df_merged = self.loader.get_dataset(mode, load_cached_data=load_cached_data)

        # Split into Stream A and Stream B
        # Stream A: Player 2 is NOT 'G'
        mask_stream_a = df_merged["nfl_player_id_2"] != "G"
        df_a = df_merged[mask_stream_a].copy()

        # Stream B: Player 2 IS 'G'
        mask_stream_b = df_merged["nfl_player_id_2"] == "G"
        df_b = df_merged[mask_stream_b].copy()

        print(f"Stream A samples: {len(df_a)}, Stream B samples: {len(df_b)}")

        # Process
        X_a, y_a, ids_a = self.process_stream_a(df_a)
        X_b, y_b, ids_b = self.process_stream_b(df_b)

        # Save to cache
        print("Saving features to cache...")
        os.makedirs(self.config.WORKING_DIR, exist_ok=True)

        X_a.to_parquet(path_X_a, index=False)
        np.save(path_y_a, y_a)
        np.save(path_ids_a, ids_a)

        X_b.to_parquet(path_X_b, index=False)
        np.save(path_y_b, y_b)
        np.save(path_ids_b, ids_b)

        # Cleanup
        del df_merged, df_a, df_b
        gc.collect()

        return (X_a, y_a, ids_a), (X_b, y_b, ids_b)
