import os
import gc
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from library.config import (
    WORKING_DIR,
    TRAIN_FEATURES_PATH,
    VAL_FEATURES_PATH,
    TEST_FEATURES_PATH,
    FEATURE_COLS,
    WINDOW_PRE,
    WINDOW_POST,
    GATING_THRESHOLD,
    GROUND_DISTANCE_SENTINEL,
    SEED,
    RAW_VECTOR_BASE_COLS,
    SPECTRAL_FEATURES,
    PHYSICS_FEATURES,
)
from library.utils import get_logger, Timer, suppress_warnings

suppress_warnings()
logger = get_logger("feature_engineering")


class FeatureEngineer:
    def __init__(self, cache_dir=WORKING_DIR):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def _preprocess_tracking(self, tracking_df):
        """
        Calculates Cartesian velocity and acceleration vectors from tracking data.
        Returns a dictionary mapping (game_play, nfl_player_id) -> numpy array of shape (max_step+1, features).
        Features indices: 0:x, 1:y, 2:vx, 3:vy, 4:theta(orientation)
        """
        df = tracking_df.copy()

        # Fill NaNs
        df["direction"] = df["direction"].fillna(0)
        df["speed"] = df["speed"].fillna(0)
        df["orientation"] = df["orientation"].fillna(0)
        df["x_position"] = df["x_position"].fillna(0)
        df["y_position"] = df["y_position"].fillna(0)

        # Convert direction to radians (assuming 0 is Y-axis, 90 is X-axis based on NFL standard)
        rads = np.radians(df["direction"].values)
        df["v_x"] = df["speed"].values * np.sin(rads)
        df["v_y"] = df["speed"].values * np.cos(rads)

        # We will extract x, y, vx, vy, orientation
        # We assume steps are integers and we can index an array with them.
        keep_cols = [
            "game_play",
            "nfl_player_id",
            "step",
            "x_position",
            "y_position",
            "v_x",
            "v_y",
            "orientation",
        ]
        df_sub = df[keep_cols]
        df_sub["game_play"] = df_sub["game_play"].astype(str)

        # Optimization: Build lookup dictionary
        lookup = {}
        grouped = df_sub.groupby(["game_play", "nfl_player_id"])

        for name, group in grouped:
            steps = group["step"].values.astype(int)
            if len(steps) == 0:
                continue
            max_step = steps.max()

            # Features: x, y, vx, vy, orientation (5 feats)
            data = np.zeros((max_step + 1, 5), dtype=np.float32)

            vals = group[
                ["x_position", "y_position", "v_x", "v_y", "orientation"]
            ].values
            data[steps] = vals

            lookup[name] = data

        return lookup

    def _compute_features_batch(self, metadata_batch, lookup):
        """
        Computes features for a batch of metadata rows using vectorized operations.
        """
        w_pre = WINDOW_PRE
        w_post = WINDOW_POST
        w_total = w_pre + w_post + 1
        n_samples = len(metadata_batch)

        # Initialize output arrays
        # Physics Primitives
        dist_arr = np.zeros(n_samples, dtype=np.float32)
        speed_p1_arr = np.zeros(n_samples, dtype=np.float32)
        speed_p2_arr = np.zeros(n_samples, dtype=np.float32)
        acc_p1_arr = np.zeros(n_samples, dtype=np.float32)
        acc_p2_arr = np.zeros(n_samples, dtype=np.float32)
        closing_speed_arr = np.zeros(n_samples, dtype=np.float32)
        ttc_arr = np.zeros(n_samples, dtype=np.float32)
        orient_p1_arr = np.zeros(n_samples, dtype=np.float32)
        orient_p2_arr = np.zeros(n_samples, dtype=np.float32)
        dir_p1_arr = np.zeros(n_samples, dtype=np.float32)
        dir_p2_arr = np.zeros(n_samples, dtype=np.float32)

        # Spectral
        spec_rad_arr = np.zeros(n_samples, dtype=np.float32)
        spec_tan_arr = np.zeros(n_samples, dtype=np.float32)

        # Window tensors
        p1_windows = np.zeros((n_samples, w_total, 5), dtype=np.float32)
        p2_windows = np.zeros((n_samples, w_total, 5), dtype=np.float32)

        # Extract Windows
        # Note: Iterating rows to extract windows is necessary due to dictionary lookup,
        # but computation afterwards is vectorized.
        for i, row in enumerate(metadata_batch.itertuples(index=False)):
            gp = str(row.game_play)
            step = int(row.step)
            p1 = int(row.nfl_player_id_1)
            p2_raw = row.nfl_player_id_2

            # P1 Data
            if (gp, p1) in lookup:
                track = lookup[(gp, p1)]
                track_len = len(track)
                start = step - w_pre
                end = step + w_post + 1

                t_start = max(0, start)
                t_end = min(track_len, end)
                w_start = t_start - start
                w_end = w_start + (t_end - t_start)

                if t_end > t_start:
                    p1_windows[i, w_start:w_end, :] = track[t_start:t_end, :]

            # P2 Data
            if p2_raw != "G":
                p2 = int(p2_raw)
                if (gp, p2) in lookup:
                    track = lookup[(gp, p2)]
                    track_len = len(track)
                    start = step - w_pre
                    end = step + w_post + 1

                    t_start = max(0, start)
                    t_end = min(track_len, end)
                    w_start = t_start - start
                    w_end = w_start + (t_end - t_start)

                    if t_end > t_start:
                        p2_windows[i, w_start:w_end, :] = track[t_start:t_end, :]

        # --- Vectorized Computation ---

        # 1. Calculate Acceleration (Gradient of Velocity)
        # shape: (N, W, 2)
        # 0:x, 1:y, 2:vx, 3:vy, 4:theta
        p1_vx = p1_windows[:, :, 2]
        p1_vy = p1_windows[:, :, 3]
        p1_ax = np.gradient(p1_vx, axis=1) / 0.1
        p1_ay = np.gradient(p1_vy, axis=1) / 0.1

        p2_vx = p2_windows[:, :, 2]
        p2_vy = p2_windows[:, :, 3]
        p2_ax = np.gradient(p2_vx, axis=1) / 0.1
        p2_ay = np.gradient(p2_vy, axis=1) / 0.1

        # 2. Relative Vectors
        rel_px = p1_windows[:, :, 0] - p2_windows[:, :, 0]
        rel_py = p1_windows[:, :, 1] - p2_windows[:, :, 1]
        dist_window = np.sqrt(rel_px**2 + rel_py**2)

        rel_vx = p1_vx - p2_vx
        rel_vy = p1_vy - p2_vy

        rel_ax = p1_ax - p2_ax
        rel_ay = p1_ay - p2_ay

        # 3. Physics Primitives at t=0 (Center of window)
        c = w_pre
        dist_arr = dist_window[:, c]

        speed_p1_arr = np.sqrt(p1_vx[:, c] ** 2 + p1_vy[:, c] ** 2)
        speed_p2_arr = np.sqrt(p2_vx[:, c] ** 2 + p2_vy[:, c] ** 2)

        acc_p1_arr = np.sqrt(p1_ax[:, c] ** 2 + p1_ay[:, c] ** 2)
        acc_p2_arr = np.sqrt(p2_ax[:, c] ** 2 + p2_ay[:, c] ** 2)

        # Closing Speed
        safe_dist = np.where(dist_arr < 1e-6, 1e-6, dist_arr)
        ux = rel_px[:, c] / safe_dist
        uy = rel_py[:, c] / safe_dist
        closing_speed_arr = -(rel_vx[:, c] * ux + rel_vy[:, c] * uy)

        # TTC
        ttc_arr = np.where(closing_speed_arr > 1e-3, dist_arr / closing_speed_arr, 10.0)

        orient_p1_arr = p1_windows[:, c, 4]
        orient_p2_arr = p2_windows[:, c, 4]

        dir_p1_arr = np.degrees(np.arctan2(p1_vy[:, c], p1_vx[:, c]))
        dir_p2_arr = np.degrees(np.arctan2(p2_vy[:, c], p2_vx[:, c]))

        # 4. Spectral Features (Vector Decomposed)
        dw_safe = np.where(dist_window < 1e-6, 1e-6, dist_window)
        ux_w = rel_px / dw_safe
        uy_w = rel_py / dw_safe

        # Radial Acc: dot(rel_acc, u)
        acc_rad = rel_ax * ux_w + rel_ay * uy_w

        # Tangential Acc: Magnitude of (rel_acc - rad*u)
        tan_x = rel_ax - acc_rad * ux_w
        tan_y = rel_ay - acc_rad * uy_w
        acc_tan = np.sqrt(tan_x**2 + tan_y**2)

        # High Pass (Mean subtraction) + RMS
        acc_rad_centered = acc_rad - np.mean(acc_rad, axis=1, keepdims=True)
        acc_tan_centered = acc_tan - np.mean(acc_tan, axis=1, keepdims=True)

        spec_rad_arr = np.sqrt(np.mean(acc_rad_centered**2, axis=1))
        spec_tan_arr = np.sqrt(np.mean(acc_tan_centered**2, axis=1))

        # 5. Flattened Raw Vectors
        # Stack: [rel_vx, rel_vy, rel_ax, rel_ay]
        raw_feats = np.stack([rel_vx, rel_vy, rel_ax, rel_ay], axis=2)  # (N, W, 4)

        # Reorder time axis to: [0, -1, ..., -pre, +1, ..., +post]
        idx_order = [c]
        for k in range(1, w_pre + 1):
            idx_order.append(c - k)
        for k in range(1, w_post + 1):
            idx_order.append(c + k)

        raw_feats_ordered = raw_feats[:, idx_order, :]  # (N, W, 4)

        # Transpose to (N, 4, W) so flattening groups by feature type
        raw_feats_ordered = np.transpose(raw_feats_ordered, (0, 2, 1))
        flat_vecs_arr = raw_feats_ordered.reshape(n_samples, -1)

        # 6. Sentinel & Gating
        is_ground = (metadata_batch["nfl_player_id_2"] == "G").values
        dist_arr[is_ground] = GROUND_DISTANCE_SENTINEL

        min_d = np.min(dist_window, axis=1)
        keep_mask = (min_d < GATING_THRESHOLD) | is_ground

        # Assemble
        phys_data = np.stack(
            [
                dist_arr,
                speed_p1_arr,
                speed_p2_arr,
                acc_p1_arr,
                acc_p2_arr,
                closing_speed_arr,
                ttc_arr,
                orient_p1_arr,
                orient_p2_arr,
                dir_p1_arr,
                dir_p2_arr,
            ],
            axis=1,
        )

        spec_data = np.stack([spec_rad_arr, spec_tan_arr], axis=1)

        all_feats = np.hstack([phys_data, spec_data, flat_vecs_arr])

        df_feats = pd.DataFrame(all_feats, columns=FEATURE_COLS)
        df_feats["contact_id"] = metadata_batch["contact_id"].values
        df_feats["game_play"] = metadata_batch["game_play"].values
        df_feats["step"] = metadata_batch["step"].values
        if "contact" in metadata_batch.columns:
            df_feats["contact"] = metadata_batch["contact"].values

        return df_feats[keep_mask].copy()

    def generate_features(
        self, metadata_df, tracking_path, mode="train", load_cached_data=True
    ):
        """
        Main pipeline execution.
        """
        if mode == "train":
            cache_path = TRAIN_FEATURES_PATH
        elif mode == "val":
            cache_path = VAL_FEATURES_PATH
        else:
            cache_path = TEST_FEATURES_PATH

        if load_cached_data and os.path.exists(cache_path):
            logger.info(f"Loading cached features from {cache_path}")
            return pd.read_parquet(cache_path)

        logger.info(f"Generating features for {mode} set...")

        with Timer("Loading and Preprocessing Tracking", logger):
            tr_df = pd.read_csv(tracking_path)
            lookup = self._preprocess_tracking(tr_df)
            del tr_df
            gc.collect()

        batch_size = 10000
        results = []

        with Timer(f"Processing {len(metadata_df)} interactions", logger):
            for i in range(0, len(metadata_df), batch_size):
                batch = metadata_df.iloc[i : i + batch_size]
                df_batch = self._compute_features_batch(batch, lookup)
                results.append(df_batch)

        full_df = pd.concat(results, axis=0, ignore_index=True)

        logger.info(f"Saving features to {cache_path}")
        full_df.to_parquet(cache_path, index=False)

        return full_df
