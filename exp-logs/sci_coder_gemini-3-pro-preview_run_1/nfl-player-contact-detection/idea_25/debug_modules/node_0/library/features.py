import os
import numpy as np
import pandas as pd
from library.config import Config
from library.data_loader import load_metadata, load_tracking, merge_tracking_data


class FeatureEngineer:
    """
    Implements the Collision-Aligned Vector-Spectral Feature Engineering pipeline.
    """

    def __init__(self):
        self.working_dir = Config.WORKING_DIR
        os.makedirs(self.working_dir, exist_ok=True)

    def process_data(self, dataset_type="train", load_cached_data=True):
        """
        Main entry point to generate features for a specific dataset split.

        Args:
            dataset_type (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            pd.DataFrame: The dataframe containing all features and metadata.
        """
        # Determine cache path
        if dataset_type == "train":
            cache_path = Config.CACHE_TRAIN_FEATURES
        elif dataset_type == "val":
            cache_path = Config.CACHE_VAL_FEATURES
        elif dataset_type == "test":
            cache_path = Config.CACHE_TEST_FEATURES
        else:
            raise ValueError(f"Unknown dataset_type: {dataset_type}")

        # 1. Try Load Cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"[{dataset_type.upper()}] Loading features from cache: {cache_path}")
            try:
                return pd.read_parquet(cache_path)
            except Exception as e:
                print(f"Error loading cache ({e}). Recomputing...")

        print(f"[{dataset_type.upper()}] Generating features...")

        # 2. Load and Merge Data
        df_meta = load_metadata(dataset_type)

        # For val, we use train tracking. For test, test tracking.
        tracking_type = "test" if dataset_type == "test" else "train"
        df_tracking = load_tracking(tracking_type)

        # Merge
        df = merge_tracking_data(df_meta, df_tracking, dataset_type, load_cached_data)

        # 3. Feature Engineering
        df = self._compute_physics_features(df)

        # 4. Save Cache
        print(f"[{dataset_type.upper()}] Saving features to {cache_path}...")
        df.to_parquet(cache_path, index=False)

        return df

    def _compute_physics_features(self, df):
        """
        Computes vector-spectral physics features and applies sentinel values.
        """
        # Identify Ground interactions
        is_ground = df["nfl_player_id_2"] == "G"

        # --- 1. Basic Relative Geometry (Player-Player) ---
        # Initialize columns with default values (will be overwritten for P-P)
        # We use float32 to save memory
        df["distance"] = Config.GROUND_DISTANCE_SENTINEL

        # Calculate deltas for Player-Player
        # Note: Ground rows have NaNs in _p2 columns, so operations will result in NaN
        # We will fill them later.
        dx = df["x_position_p1"] - df["x_position_p2"]
        dy = df["y_position_p1"] - df["y_position_p2"]

        dist = np.sqrt(dx**2 + dy**2)

        # Assign distance for non-ground
        df.loc[~is_ground, "distance"] = dist[~is_ground]

        # --- 2. Vector Decomposition (Collision Alignment) ---
        # Unit vector r_hat (P2 -> P1)
        # Avoid division by zero
        safe_dist = dist.replace(0, 1e-6)
        rx = dx / safe_dist
        ry = dy / safe_dist

        # Relative Velocity
        dvx = df["speed_p1"] * np.sin(np.radians(df["direction_p1"])) - df[
            "speed_p2"
        ] * np.sin(np.radians(df["direction_p2"]))
        dvy = df["speed_p1"] * np.cos(np.radians(df["direction_p1"])) - df[
            "speed_p2"
        ] * np.cos(np.radians(df["direction_p2"]))

        # Store raw relative velocity
        df["rel_v_x"] = dvx
        df["rel_v_y"] = dvy

        # Radial Velocity (Dot product with r_hat)
        # v_rad = v_rel . r_hat
        v_rad = dvx * rx + dvy * ry
        df["radial_velocity"] = v_rad

        # Tangential Velocity (Magnitude of rejection)
        # v_tan = || v_rel - v_rad * r_hat ||
        v_tan_x = dvx - v_rad * rx
        v_tan_y = dvy - v_rad * ry
        df["tangential_velocity"] = np.sqrt(v_tan_x**2 + v_tan_y**2)

        # Relative Acceleration
        # Acceleration is magnitude. We need vector components.
        # We assume acceleration direction aligns with motion direction or orientation?
        # Tracking data gives 'acceleration' (magnitude) and 'sa' (signed accel).
        # We don't have explicit acceleration angle.
        # Approximation: Assume acceleration is along the direction of motion (speed vector).
        # This is imperfect but standard when accel angle is missing.
        ax_p1 = df["acceleration_p1"] * np.sin(np.radians(df["direction_p1"]))
        ay_p1 = df["acceleration_p1"] * np.cos(np.radians(df["direction_p1"]))
        ax_p2 = df["acceleration_p2"] * np.sin(np.radians(df["direction_p2"]))
        ay_p2 = df["acceleration_p2"] * np.cos(np.radians(df["direction_p2"]))

        dax = ax_p1 - ax_p2
        day = ay_p1 - ay_p2

        # Radial Acceleration
        a_rad = dax * rx + day * ry
        df["radial_acceleration"] = a_rad

        # Tangential Acceleration
        a_tan_x = dax - a_rad * rx
        a_tan_y = day - a_rad * ry
        df["tangential_acceleration"] = np.sqrt(a_tan_x**2 + a_tan_y**2)

        # --- 3. Spectral Features (Rolling RMS of Radial Accel) ---
        # We need to sort to ensure temporal order for rolling operations
        # Sort keys: game_play, p1, p2, step
        df.sort_values(
            by=["game_play", "nfl_player_id_1", "nfl_player_id_2", "step"], inplace=True
        )

        # Define a grouper
        # Note: Groupby rolling is slow. We use a vectorized approach with shift.
        # We calculate rolling std dev over the window.
        # Ideally we want: df.groupby(pair)['radial_acceleration'].rolling(5).std()
        # Optimization: Use shift for fixed window size 5

        col = "radial_acceleration"
        # Create shifted columns
        shifts = []
        for i in range(Config.SPECTRAL_WINDOW_SIZE):
            shifts.append(df[col].shift(i))

        # Check group consistency (ensure we don't roll over to next play/pair)
        # We check if (game_play, p1, p2) is same as lag i
        group_cols = ["game_play", "nfl_player_id_1", "nfl_player_id_2"]
        # Create a unique group ID for faster comparison
        # We can just check if group_cols match for shift(window-1)
        # If the last lag is in the same group, all intermediate lags are too (due to sorting)

        # Concatenate shifts to compute std
        shift_df = pd.concat(shifts, axis=1)

        # Compute rolling std (ddof=0 or 1, 1 is standard sample std)
        roll_std = shift_df.std(axis=1, ddof=1)

        # Mask boundaries
        # Identify where the group changes within the window
        # Compare current row group with the (window-1)-th previous row group
        # If they differ, the window crosses a boundary -> set to 0 or NaN
        # Actually, for the first few frames of a play, spectral energy is undefined/low.
        # We'll fill with 0.

        # Create a hash or string for group ID to check boundary
        # Using game_play is usually sufficient if sorted, but p1/p2 matters.
        # Let's use a simple check:
        # If step < window_size - 1, it's partial.
        # But steps might not start at 0.
        # We'll assume the shift logic is valid only if the group ID matches.

        # Robust boundary check:
        # Check if shift(window-1) group ID == current group ID
        # Since we have 3 group cols, let's just check the index if we reset it? No.
        # Let's check the step difference. If step - shift(4).step == 4, it's contiguous.
        step_diff = df["step"] - df["step"].shift(Config.SPECTRAL_WINDOW_SIZE - 1)
        valid_window = step_diff == (Config.SPECTRAL_WINDOW_SIZE - 1)

        df["radial_accel_spectral_energy"] = roll_std.where(valid_window, 0.0)

        # --- 4. Quadratic Reachability (Gating Feature) ---
        # d(t) = d0 + v_rad * t + 0.5 * a_rad * t^2
        # We want min(d(t)) for t in [0, 1.0]
        # t_vertex = -v_rad / a_rad

        # Avoid div by zero in vertex calc
        safe_a_rad = df["radial_acceleration"].replace(0, 1e-6)
        t_vertex = -df["radial_velocity"] / safe_a_rad

        # Evaluate d(t) at t=0, t=1, and t=vertex (if in range)
        # d(0) = distance
        d_0 = df["distance"]

        # d(1)
        d_1 = (
            d_0
            + df["radial_velocity"] * 1.0
            + 0.5 * df["radial_acceleration"] * (1.0**2)
        )

        # d(vertex)
        d_v = (
            d_0
            + df["radial_velocity"] * t_vertex
            + 0.5 * df["radial_acceleration"] * (t_vertex**2)
        )

        # Initialize min_dist with min(d_0, d_1)
        min_dist = np.minimum(d_0, d_1)

        # Update with d_v where 0 < t_vertex < 1
        valid_vertex = (t_vertex > 0) & (t_vertex < 1.0)
        min_dist = np.where(valid_vertex, np.minimum(min_dist, d_v), min_dist)

        df["quadratic_min_dist"] = min_dist

        # --- 5. Context Features ---
        df["speed_diff"] = np.abs(df["speed_p1"] - df["speed_p2"])
        df["acc_diff"] = np.abs(df["acceleration_p1"] - df["acceleration_p2"])

        # --- 6. Sentinel Application (Ground) ---
        # For Ground rows, we set specific values
        # Distance is already -1.0
        # Set vector features to 0
        vector_cols = [
            "radial_velocity",
            "tangential_velocity",
            "radial_acceleration",
            "tangential_acceleration",
            "radial_accel_spectral_energy",
            "rel_v_x",
            "rel_v_y",
            "speed_diff",
            "acc_diff",
            "quadratic_min_dist",
        ]

        for col in vector_cols:
            df.loc[is_ground, col] = 0.0

        # Also ensure P2 columns are 0 for Ground (they are NaNs currently)
        p2_cols = [c for c in df.columns if c.endswith("_p2")]
        for col in p2_cols:
            df.loc[is_ground, col] = 0.0

        # Fill any remaining NaNs (e.g. from missing tracking for players) with 0
        # This ensures the model doesn't crash
        df.fillna(0.0, inplace=True)

        return df
