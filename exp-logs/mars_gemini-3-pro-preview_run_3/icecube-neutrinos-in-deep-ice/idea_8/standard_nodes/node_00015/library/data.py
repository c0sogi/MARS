import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config
from library.geometry import (
    load_sensor_geometry,
    compute_canonical_rotation,
    apply_rotation,
)


class IceCubeDataset(Dataset):
    """
    PyTorch Dataset for the Dual-View Attentive Graph Network.
    Provides both Raw and Canonical coordinate views for each event.
    """

    def __init__(self, X_raw, X_canon, y=None):
        """
        Args:
            X_raw (np.ndarray): Raw features [N_events, Max_Pulses, 6].
            X_canon (np.ndarray): Canonical features [N_events, Max_Pulses, 6].
            y (np.ndarray, optional): Targets [N_events, 2] (azimuth, zenith) or IDs [N_events].
        """
        self.X_raw = torch.tensor(X_raw, dtype=torch.float32)
        self.X_canon = torch.tensor(X_canon, dtype=torch.float32)

        if y is not None:
            # Check if y contains integer IDs or float targets
            if np.issubdtype(y.dtype, np.integer):
                self.y = torch.tensor(y, dtype=torch.int64)
            else:
                self.y = torch.tensor(y, dtype=torch.float32)
        else:
            self.y = None

    def __len__(self):
        return len(self.X_raw)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X_raw[idx], self.X_canon[idx], self.y[idx]
        else:
            return self.X_raw[idx], self.X_canon[idx]


def process_batch(batch_id, meta_df, sensor_map, mode="train", load_cached_data=True):
    """
    Processes a single batch of events: reads parquet, maps geometry,
    performs hybrid sampling, computes canonical transformation, and caches results.

    Args:
        batch_id (int): The batch ID to process.
        meta_df (pd.DataFrame): Metadata dataframe containing event info.
        sensor_map (dict): Dictionary mapping sensor_id to [x, y, z].
        mode (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (X_raw, X_canon, targets_or_ids)
    """
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Define cache file paths
    cache_base = os.path.join(Config.CACHE_DIR, f"{mode}_batch_{batch_id}")
    path_X_raw = f"{cache_base}_X_raw.npy"
    path_X_canon = f"{cache_base}_X_canon.npy"
    path_meta = f"{cache_base}_meta.npy"  # Stores targets for train/val, ids for test

    # 1. Try to load from cache
    if load_cached_data:
        if (
            os.path.exists(path_X_raw)
            and os.path.exists(path_X_canon)
            and os.path.exists(path_meta)
        ):
            try:
                X_raw = np.load(path_X_raw)
                X_canon = np.load(path_X_canon)
                meta_data = np.load(path_meta)
                return X_raw, X_canon, meta_data
            except Exception as e:
                print(
                    f"Warning: Failed to load cache for batch {batch_id}: {e}. Recomputing..."
                )

    # 2. Compute from scratch
    # Construct file path
    if mode in ["train", "val"]:
        file_path = os.path.join(Config.INPUT_DIR, "train", f"batch_{batch_id}.parquet")
    else:
        file_path = os.path.join(Config.INPUT_DIR, "test", f"batch_{batch_id}.parquet")

    # Load data
    # Note: event_id is usually the index in the parquet file
    batch_df = pd.read_parquet(file_path)
    if batch_df.index.name == "event_id":
        batch_df = batch_df.reset_index()

    # Filter metadata for this batch
    batch_meta = meta_df[meta_df["batch_id"] == batch_id].copy()

    # Initialize arrays
    n_events = len(batch_meta)
    n_features = 6  # x, y, z, time, charge, auxiliary

    X_raw = np.zeros((n_events, Config.MAX_PULSES, n_features), dtype=np.float32)
    X_canon = np.zeros((n_events, Config.MAX_PULSES, n_features), dtype=np.float32)

    if mode in ["train", "val"]:
        targets = np.zeros((n_events, 2), dtype=np.float32)  # azimuth, zenith
    else:
        ids = np.zeros((n_events,), dtype=np.int64)

    # Create a mapping from event_id to index in the output arrays
    # This ensures alignment between metadata and processed features
    event_ids = batch_meta["event_id"].values
    event_id_to_idx = {eid: i for i, eid in enumerate(event_ids)}

    # Group by event_id for processing
    events_group = batch_df.groupby("event_id")

    for eid, group in events_group:
        if eid not in event_id_to_idx:
            continue

        idx = event_id_to_idx[eid]

        # Extract columns
        s_ids = group["sensor_id"].values
        time = group["time"].values.astype(np.float32)
        charge = group["charge"].values.astype(np.float32)
        aux = group["auxiliary"].values.astype(np.float32)

        # Map geometry
        # Handle potential missing sensors gracefully (though dataset should be clean)
        try:
            xyz = np.array([sensor_map[sid] for sid in s_ids], dtype=np.float32)
        except KeyError:
            # Fallback: filter out unknown sensors or use 0,0,0
            valid_mask = [sid in sensor_map for sid in s_ids]
            if not any(valid_mask):
                continue
            s_ids = s_ids[valid_mask]
            time = time[valid_mask]
            charge = charge[valid_mask]
            aux = aux[valid_mask]
            xyz = np.array([sensor_map[sid] for sid in s_ids], dtype=np.float32)

        # Hybrid Sampling
        n_pulses = len(time)
        if n_pulses > Config.MAX_PULSES:
            n_high_q = Config.MAX_PULSES // 2
            n_early_t = Config.MAX_PULSES - n_high_q

            # Indices for high charge (highest values)
            idx_q = np.argsort(charge)[-n_high_q:]

            # Mask out selected indices to pick remaining from time
            mask = np.ones(n_pulses, dtype=bool)
            mask[idx_q] = False
            remaining_indices = np.where(mask)[0]

            # Indices for early time from remainder
            if len(remaining_indices) > n_early_t:
                idx_t_sub = np.argsort(time[remaining_indices])[:n_early_t]
                idx_t = remaining_indices[idx_t_sub]
            else:
                idx_t = remaining_indices

            # Combine and sort by original index (roughly time)
            indices = np.concatenate([idx_q, idx_t])
            indices = np.sort(indices)
        else:
            indices = np.arange(n_pulses)

        # Select Data
        sel_xyz = xyz[indices]
        sel_time = time[indices]
        sel_charge = charge[indices]
        sel_aux = aux[indices]

        # Feature Normalization
        # Time: relative to min time in selection, scaled to approx microseconds
        t_min = sel_time.min()
        sel_time_norm = (sel_time - t_min) / 1000.0

        # Charge: log10 transform
        sel_charge_norm = np.log10(sel_charge + 1.0)

        # Compute Canonical Transformation using library.geometry
        rotation_matrix, cog = compute_canonical_rotation(sel_xyz, sel_charge, sel_time)
        xyz_canon = apply_rotation(sel_xyz, rotation_matrix, cog)

        # Fill Arrays
        n_sel = len(indices)

        # Raw View Features: [x, y, z, t, q, aux]
        X_raw[idx, :n_sel, 0:3] = sel_xyz
        X_raw[idx, :n_sel, 3] = sel_time_norm
        X_raw[idx, :n_sel, 4] = sel_charge_norm
        X_raw[idx, :n_sel, 5] = sel_aux

        # Canonical View Features: [x', y', z', t, q, aux]
        X_canon[idx, :n_sel, 0:3] = xyz_canon
        X_canon[idx, :n_sel, 3] = sel_time_norm
        X_canon[idx, :n_sel, 4] = sel_charge_norm
        X_canon[idx, :n_sel, 5] = sel_aux

        # Store ID for test mode
        if mode == "test":
            ids[idx] = eid

    # Fill targets for train/val mode
    if mode in ["train", "val"]:
        # batch_meta is already aligned with event_id_to_idx because we built the map from it
        targets[:, 0] = batch_meta["azimuth"].values
        targets[:, 1] = batch_meta["zenith"].values
        meta_data = targets
    else:
        meta_data = ids

    # 3. Save to cache
    np.save(path_X_raw, X_raw)
    np.save(path_X_canon, X_canon)
    np.save(path_meta, meta_data)

    return X_raw, X_canon, meta_data
