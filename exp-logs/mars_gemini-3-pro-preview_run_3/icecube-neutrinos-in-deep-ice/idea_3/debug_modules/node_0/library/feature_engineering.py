import pandas as pd
import numpy as np
import os
import gc
from pathlib import Path
from library.config import (
    SENSOR_GEOMETRY_PATH,
    CACHE_DIR,
    FEATURE_NAMES,
    TARGET_COLS,
    INPUT_DIR,
    SEED,
    LGBM_PARAMS,
)
from library.utils import spherical_to_cartesian


class FeatureExtractor:
    def __init__(self):
        """
        Initializes the FeatureExtractor by loading the sensor geometry
        and creating a fast lookup map for sensor positions.
        """
        self.geometry = pd.read_csv(SENSOR_GEOMETRY_PATH)

        # Create a mapping array for fast lookup: sensor_id -> x, y, z
        # We assume sensor_ids are integers. We map them to indices in a numpy array.
        max_sensor_id = self.geometry["sensor_id"].max()
        self.geo_map = np.zeros((max_sensor_id + 1, 3), dtype=np.float32)

        # Fill the map
        # sensor_geometry.csv columns: sensor_id, x, y, z
        ids = self.geometry["sensor_id"].values
        coords = self.geometry[["x", "y", "z"]].values
        self.geo_map[ids] = coords

    def extract_features(self, meta_df, mode="train", load_cached_data=True):
        """
        Main method to extract features for a given metadata DataFrame.

        Args:
            meta_df (pd.DataFrame): Metadata containing batch_id and event_id.
            mode (str): 'train' or 'test'. Used to locate raw files.
            load_cached_data (bool): If True, attempts to load from cache.

        Returns:
            tuple: (X, y, ids)
                X (np.ndarray): Feature matrix.
                y (np.ndarray): Target matrix (if available in meta_df), else None.
                ids (np.ndarray): Event IDs corresponding to rows.
        """
        # Ensure cache directory exists
        os.makedirs(CACHE_DIR, exist_ok=True)

        # We will collect features for the requested events here
        # Since meta_df might span multiple batches, we iterate through them.
        unique_batches = meta_df["batch_id"].unique()

        X_list = []
        y_list = []
        ids_list = []

        # Pre-calculate targets if they exist in metadata
        has_targets = "azimuth" in meta_df.columns and "zenith" in meta_df.columns

        # Create a dictionary for fast lookup of targets by event_id if needed
        if has_targets:
            # Convert spherical to cartesian targets
            tx, ty, tz = spherical_to_cartesian(
                meta_df["azimuth"].values, meta_df["zenith"].values
            )
            target_map = pd.DataFrame(
                {
                    "event_id": meta_df["event_id"].values,
                    "target_x": tx,
                    "target_y": ty,
                    "target_z": tz,
                }
            ).set_index("event_id")

        # Iterate over batches
        for batch_id in unique_batches:
            # Get features for the *entire* batch (cached or computed)
            batch_X, batch_ids = self._get_batch_features(
                batch_id, mode, load_cached_data
            )

            # Filter to keep only the events present in the current meta_df
            # We use a mask for efficiency
            requested_ids = meta_df[meta_df["batch_id"] == batch_id]["event_id"].values

            # Intersect batch_ids with requested_ids
            # Note: batch_ids from processing are unique. requested_ids are unique.
            # We need to align X with requested_ids.

            # Create a temporary DataFrame to facilitate merge/alignment
            # (Using pandas for alignment is safer and easier than raw numpy indexing here)
            df_batch_feats = pd.DataFrame(batch_X, columns=FEATURE_NAMES)
            df_batch_feats["event_id"] = batch_ids

            # Filter rows
            df_filtered = df_batch_feats[
                df_batch_feats["event_id"].isin(requested_ids)
            ].copy()

            # Ensure the order matches requested_ids if strict ordering is needed,
            # but usually just aligning X and y is sufficient.
            # We will sort by event_id to ensure determinism.
            df_filtered = df_filtered.sort_values("event_id")

            # Extract aligned X and ids
            current_ids = df_filtered["event_id"].values
            current_X = df_filtered[FEATURE_NAMES].values.astype(np.float32)

            X_list.append(current_X)
            ids_list.append(current_ids)

            if has_targets:
                # Get targets for these specific events
                current_targets = target_map.loc[current_ids][
                    ["target_x", "target_y", "target_z"]
                ].values.astype(np.float32)
                y_list.append(current_targets)

            # Explicit garbage collection
            del df_batch_feats, df_filtered, batch_X, batch_ids
            gc.collect()

        # Concatenate all results
        if not X_list:
            return np.array([]), np.array([]), np.array([])

        final_X = np.concatenate(X_list, axis=0)
        final_ids = np.concatenate(ids_list, axis=0)

        final_y = None
        if has_targets:
            final_y = np.concatenate(y_list, axis=0)

        return final_X, final_y, final_ids

    def _get_batch_features(self, batch_id, mode, load_cached_data):
        """
        Retrieves features for a full batch, either from cache or by computing them.
        """
        cache_X_path = CACHE_DIR / f"{mode}_batch_{batch_id}_X.npy"
        cache_ids_path = CACHE_DIR / f"{mode}_batch_{batch_id}_ids.npy"

        if load_cached_data and cache_X_path.exists() and cache_ids_path.exists():
            try:
                X = np.load(cache_X_path)
                ids = np.load(cache_ids_path)
                return X, ids
            except Exception as e:
                print(f"Failed to load cache for batch {batch_id}: {e}. Recomputing.")

        # Compute from scratch
        X, ids = self._process_batch_file(batch_id, mode)

        # Save to cache
        np.save(cache_X_path, X)
        np.save(cache_ids_path, ids)

        return X, ids

    def _process_batch_file(self, batch_id, mode):
        """
        Loads raw parquet data, merges geometry, and computes features for all events.
        """
        batch_file = INPUT_DIR / mode / f"batch_{batch_id}.parquet"

        # Load batch data
        # Columns: event_id (index or col), time, sensor_id, charge, auxiliary
        df = pd.read_parquet(batch_file)

        # If event_id is the index, reset it to make it a column
        if df.index.name == "event_id":
            df = df.reset_index()

        # Map sensor geometry
        # We use numpy indexing for speed
        sensor_ids = df["sensor_id"].values

        # Get coordinates (N_pulses, 3)
        coords = self.geo_map[sensor_ids]

        # Add coordinates to dataframe (or keep as numpy arrays)
        # To use groupby efficiently, we can assign them back to df,
        # or iterate over the groupby object and index into coords.
        # Assigning to DF is cleaner for code readability.
        df["x"] = coords[:, 0]
        df["y"] = coords[:, 1]
        df["z"] = coords[:, 2]

        # Prepare for iteration
        # Group by event_id
        # Note: This can be slow. A faster way is to sort by event_id and use numpy slicing,
        # but pandas groupby is robust and easier to implement correctly.
        grouped = df.groupby("event_id")

        features_list = []
        event_ids_list = []

        # Iterate over each event
        for event_id, group in grouped:
            # Extract numpy arrays for the event
            # group columns: event_id, time, sensor_id, charge, auxiliary, x, y, z
            t = group["time"].values
            q = group["charge"].values
            aux = group["auxiliary"].values
            pos = group[["x", "y", "z"]].values

            # Compute features
            feats = self._compute_single_event_features(t, q, aux, pos)

            features_list.append(feats)
            event_ids_list.append(event_id)

        return np.array(features_list, dtype=np.float32), np.array(
            event_ids_list, dtype=np.int64
        )

    def _compute_single_event_features(self, t, q, aux, pos):
        """
        Computes the feature vector for a single event.

        Args:
            t (np.array): Time of pulses.
            q (np.array): Charge of pulses.
            aux (np.array): Auxiliary flags.
            pos (np.array): (N, 3) Position matrix.

        Returns:
            np.array: Feature vector corresponding to FEATURE_NAMES.
        """
        # 1. Pulse Aggregates
        q_sum = np.sum(q)
        q_mean = np.mean(q)
        q_std = np.std(q)
        q_count = len(q)

        # 2. Temporal Features
        t_min = np.min(t)
        t_max = np.max(t)
        t_range = t_max - t_min
        t_std = np.std(t)
        aux_ratio = np.mean(aux)

        # 3. Spatial Center of Gravity (Charge Weighted)
        # Handle case where q_sum is 0 (unlikely but possible)
        if q_sum > 0:
            cog = np.average(pos, axis=0, weights=q)
        else:
            cog = np.mean(pos, axis=0)

        pos_x_mean, pos_y_mean, pos_z_mean = cog

        # Weighted Standard Deviation of Position
        # centered positions
        pos_centered = pos - cog

        # Weighted covariance matrix: (X.T @ W @ X) / sum(w)
        # W is diag(q)
        # Efficiently: (pos_centered.T * q) @ pos_centered / q_sum
        if q_sum > 0:
            cov_matrix = (pos_centered.T * q) @ pos_centered / q_sum

            # Weighted std dev for each axis is sqrt of diagonal
            pos_std = np.sqrt(np.diag(cov_matrix))
        else:
            cov_matrix = np.cov(pos_centered.T)
            pos_std = np.sqrt(np.diag(cov_matrix))

        pos_x_std, pos_y_std, pos_z_std = pos_std

        # 4. Eigen-Features (SVD of Position Covariance)
        # eigh returns eigenvalues in ascending order
        evals, evecs = np.linalg.eigh(cov_matrix)

        # Sort descending
        evals = evals[::-1]
        evecs = evecs[:, ::-1]

        eval_1, eval_2, eval_3 = evals

        # Ratios (add epsilon to avoid div by zero)
        eps = 1e-9
        eval_ratio_12 = eval_1 / (eval_2 + eps)
        eval_ratio_13 = eval_1 / (eval_3 + eps)

        # 5. Spatiotemporal Covariance (Directionality)
        # Project positions onto the principal axis (eigenvector 1)
        # evecs[:, 0] is the principal axis
        axis_1 = evecs[:, 0]

        # Projection p_i = (r_i - r_cog) . v_1
        p_proj = pos_centered @ axis_1

        # Covariance between projection p and time t
        # Weighted covariance: E[(p - p_bar)(t - t_bar)]
        # p_bar is 0 because pos_centered is weighted centered
        # t_bar needs to be weighted mean
        if q_sum > 0:
            t_bar = np.average(t, weights=q)
            t_centered = t - t_bar

            # cov(p, t) = sum(q * p * t_centered) / sum(q)
            cov_p_t = np.sum(q * p_proj * t_centered) / q_sum

            # Raw covariances (x, t), (y, t), (z, t)
            # cov(x, t) = sum(q * x_centered * t_centered) / sum(q)
            cov_x_t = np.sum(q * pos_centered[:, 0] * t_centered) / q_sum
            cov_y_t = np.sum(q * pos_centered[:, 1] * t_centered) / q_sum
            cov_z_t = np.sum(q * pos_centered[:, 2] * t_centered) / q_sum
        else:
            cov_p_t = 0.0
            cov_x_t = 0.0
            cov_y_t = 0.0
            cov_z_t = 0.0

        # Construct feature vector matching FEATURE_NAMES order
        feature_vector = [
            q_sum,
            q_mean,
            q_std,
            q_count,
            t_range,
            t_std,
            aux_ratio,
            pos_x_mean,
            pos_y_mean,
            pos_z_mean,
            pos_x_std,
            pos_y_std,
            pos_z_std,
            eval_1,
            eval_2,
            eval_3,
            eval_ratio_12,
            eval_ratio_13,
            cov_x_t,
            cov_y_t,
            cov_z_t,
            cov_p_t,
        ]

        return np.array(feature_vector, dtype=np.float32)
