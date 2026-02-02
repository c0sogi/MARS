import os
import numpy as np
import pandas as pd
import torch
from library.config import Config
from library.utils import load_sensor_geometry


def hybrid_sample_pulses(
    charge, time, sensor_id, auxiliary, x, y, z, num_pulses=Config.NUM_PULSES
):
    """
    Samples a fixed number of pulses from an event using a hybrid strategy:
    selecting the highest charge pulses and the earliest time pulses.

    Args:
        charge, time, sensor_id, auxiliary, x, y, z: Arrays of pulse data.
        num_pulses (int): Target number of pulses.

    Returns:
        np.ndarray: Sampled features of shape (num_pulses, 7).
                    Channels: [x, y, z, time, charge, auxiliary, mask]
    """
    n_total = len(charge)

    # Initialize output array
    # Channels: 0:x, 1:y, 2:z, 3:time, 4:charge, 5:aux, 6:mask
    out_feats = np.zeros((num_pulses, 7), dtype=np.float32)

    if n_total == 0:
        return out_feats

    indices = np.arange(n_total)

    if n_total <= num_pulses:
        selected_indices = indices
    else:
        # 1. Select Top N/2 by Charge
        n_charge = num_pulses // 2
        if n_charge > 0:
            # argpartition puts the k-th element in sorted position
            # We want the largest values, so we look at the end
            idx_charge = np.argpartition(charge, -n_charge)[-n_charge:]
        else:
            idx_charge = np.array([], dtype=np.int64)

        # 2. Select Earliest Time from Remaining
        n_time = num_pulses - len(idx_charge)
        if n_time > 0:
            # Create mask to find indices not yet selected
            mask = np.ones(n_total, dtype=bool)
            mask[idx_charge] = False
            remaining_indices = indices[mask]

            if len(remaining_indices) > 0:
                # If we have enough remaining, pick the earliest n_time
                if len(remaining_indices) > n_time:
                    rem_times = time[remaining_indices]
                    # Smallest times
                    rel_idx = np.argpartition(rem_times, n_time)[:n_time]
                    idx_time = remaining_indices[rel_idx]
                else:
                    idx_time = remaining_indices
            else:
                idx_time = np.array([], dtype=np.int64)
        else:
            idx_time = np.array([], dtype=np.int64)

        selected_indices = np.concatenate([idx_charge, idx_time])

    # Sort selected indices by time to preserve temporal causality for the Transformer
    sel_times = time[selected_indices]
    sort_order = np.argsort(sel_times)
    selected_indices = selected_indices[sort_order]

    n_sel = len(selected_indices)

    # Fill output array
    out_feats[:n_sel, 0] = x[selected_indices]
    out_feats[:n_sel, 1] = y[selected_indices]
    out_feats[:n_sel, 2] = z[selected_indices]
    out_feats[:n_sel, 3] = time[selected_indices]
    out_feats[:n_sel, 4] = charge[selected_indices]
    out_feats[:n_sel, 5] = auxiliary[selected_indices]
    out_feats[:n_sel, 6] = 1.0  # Mask bit

    return out_feats


def normalize_data(feats):
    """
    Normalizes pulse features in-place.

    Args:
        feats (np.ndarray): Shape (num_pulses, 7).

    Returns:
        np.ndarray: Normalized features.
    """
    mask = feats[:, 6] > 0.5
    if not np.any(mask):
        return feats

    # Spatial Normalization (Approximate detector scale ~500m)
    # Maps coordinates roughly to [-1, 1]
    feats[mask, 0] /= 500.0
    feats[mask, 1] /= 500.0
    feats[mask, 2] /= 500.0

    # Time Normalization
    # Shift event start to 0 and scale to microseconds
    t_vals = feats[mask, 3]
    t_min = np.min(t_vals)
    feats[mask, 3] = (t_vals - t_min) / 1000.0

    # Charge Normalization (Log10)
    # Add epsilon to avoid log(0)
    feats[mask, 4] = np.log10(feats[mask, 4] + 0.1)

    return feats


def compute_geometric_priors(x, y, z, time, charge, mask):
    """
    Computes a dense vector of physics-based geometric priors.
    Includes Center of Gravity, Inertia Tensor Eigenvalues, and Directional Covariance.

    Returns:
        np.ndarray: Feature vector of size 19.
    """
    valid = mask > 0.5
    # Need at least 3 points for meaningful covariance
    if np.sum(valid) < 3:
        return np.zeros(19, dtype=np.float32)

    vx, vy, vz = x[valid], y[valid], z[valid]
    vt, vq = time[valid], charge[valid]

    total_charge = np.sum(vq) + 1e-8

    # 1. Center of Gravity (COG)
    cog_x = np.sum(vx * vq) / total_charge
    cog_y = np.sum(vy * vq) / total_charge
    cog_z = np.sum(vz * vq) / total_charge

    # Centered Coordinates
    dx, dy, dz = vx - cog_x, vy - cog_y, vz - cog_z

    # 2. Inertia Tensor (Charge-Weighted Covariance Matrix)
    # Stack centered coords: (N, 3)
    coords = np.stack([dx, dy, dz], axis=1)
    # Weighted covariance: (X.T * q) @ X / Q
    cov_matrix = (coords.T * vq) @ coords / total_charge

    # Eigendecomposition
    # eigh is for symmetric matrices
    evals, evecs = np.linalg.eigh(cov_matrix)

    # Sort eigenvalues descending (eigh returns ascending)
    idx_sort = np.argsort(evals)[::-1]
    evals = evals[idx_sort]
    evecs = evecs[:, idx_sort]

    # 3. Directional Covariance
    # The principal axis (eigenvector 1) defines the line of the track.
    # We need to determine the direction (forward vs backward) using time.
    main_axis = evecs[:, 0]

    # Project positions onto the main axis
    projections = coords @ main_axis

    # Calculate covariance between projection and time
    t_mean = np.sum(vt * vq) / total_charge
    dt = vt - t_mean

    # Weighted covariance of (position_projection, time)
    cov_pt = np.sum(vq * projections * dt) / total_charge

    # If correlation is negative, the particle is moving opposite to the eigenvector.
    # Flip the vector so it aligns with time evolution.
    if cov_pt < 0:
        main_axis = -main_axis
        evecs[:, 0] = main_axis
        cov_pt = -cov_pt

    # Construct Feature Vector
    # We apply rough scaling to bring values closer to O(1) for the neural net
    priors = np.concatenate(
        [
            [cog_x / 500.0, cog_y / 500.0, cog_z / 500.0],  # 3: COG
            evals / 250000.0,  # 3: Eigenvalues
            evecs.flatten(),  # 9: Eigenvectors
            [(vt.max() - vt.min()) / 1000.0],  # 1: Duration
            [np.log10(total_charge)],  # 1: Total Charge
            [(t_mean - vt.min()) / 1000.0],  # 1: Mean Time
            [cov_pt / 500000.0],  # 1: Directional Covariance
        ]
    )

    return priors.astype(np.float32)


def preprocess_batch(
    batch_id, meta_df, sensor_geometry, output_dir=None, load_cached_data=True
):
    """
    Processes an entire batch of events: samples pulses, computes priors, and caches results.

    Args:
        batch_id (int): ID of the batch.
        meta_df (pd.DataFrame): Metadata containing event indices for this batch.
        sensor_geometry (np.ndarray): Sensor position map.
        output_dir (str): Directory to save cached files.
        load_cached_data (bool): Whether to try loading from disk first.

    Returns:
        dict: Dictionary containing 'X' (features), 'priors', 'ids', and optionally 'y' (targets).
    """
    if output_dir is None:
        output_dir = os.path.join(Config.WORKING_DIR, "cache")

    os.makedirs(output_dir, exist_ok=True)

    # Define Cache Paths
    path_X = os.path.join(output_dir, f"batch_{batch_id}_X.npy")
    path_priors = os.path.join(output_dir, f"batch_{batch_id}_priors.npy")
    path_ids = os.path.join(output_dir, f"batch_{batch_id}_ids.npy")
    path_y = os.path.join(output_dir, f"batch_{batch_id}_y.npy")

    has_targets = "azimuth" in meta_df.columns

    # 1. Try Loading from Cache
    if load_cached_data:
        if (
            os.path.exists(path_X)
            and os.path.exists(path_ids)
            and os.path.exists(path_priors)
        ):
            # If targets are expected, check if they exist
            if has_targets and not os.path.exists(path_y):
                pass  # Cache incomplete, recompute
            else:
                X = np.load(path_X)
                priors = np.load(path_priors)
                ids = np.load(path_ids)
                data = {"X": X, "priors": priors, "ids": ids}
                if has_targets:
                    data["y"] = np.load(path_y)
                return data

    # 2. Process from Scratch
    # Get file path from metadata (first row of batch)
    rel_path = meta_df.iloc[0]["file_path"]
    full_path = os.path.join(Config.INPUT_DIR, rel_path)

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Batch file not found: {full_path}")

    batch_df = pd.read_parquet(full_path)

    # Convert batch columns to numpy for fast slicing
    arr_time = batch_df["time"].values
    arr_charge = batch_df["charge"].values
    arr_aux = batch_df["auxiliary"].values.astype(np.float32)
    arr_sid = batch_df["sensor_id"].values

    # Map Sensor IDs to Geometry
    # Fancy indexing is faster than pandas map
    arr_x = sensor_geometry[arr_sid, 0]
    arr_y = sensor_geometry[arr_sid, 1]
    arr_z = sensor_geometry[arr_sid, 2]

    # Prepare Output Arrays
    n_events = len(meta_df)
    X_out = np.zeros((n_events, Config.NUM_PULSES, 7), dtype=np.float32)
    priors_out = np.zeros((n_events, 19), dtype=np.float32)
    ids_out = np.zeros(n_events, dtype=np.int64)

    if has_targets:
        y_out = np.zeros((n_events, 2), dtype=np.float32)
        vec_az = meta_df["azimuth"].values
        vec_zen = meta_df["zenith"].values

    # Extract indices from metadata for fast iteration
    first_idxs = meta_df["first_pulse_index"].values
    last_idxs = meta_df["last_pulse_index"].values
    event_ids = meta_df["event_id"].values

    # Loop over events
    for i in range(n_events):
        # Slice indices (inclusive in meta description, exclusive in python slice)
        slc = slice(first_idxs[i], last_idxs[i] + 1)

        # Extract event data
        c_evt = arr_charge[slc]
        t_evt = arr_time[slc]
        aux_evt = arr_aux[slc]
        x_evt = arr_x[slc]
        y_evt = arr_y[slc]
        z_evt = arr_z[slc]
        sid_evt = arr_sid[slc]

        # Hybrid Sampling
        feats = hybrid_sample_pulses(
            c_evt, t_evt, sid_evt, aux_evt, x_evt, y_evt, z_evt
        )

        # Compute Priors (using physical values before normalization)
        mask = feats[:, 6]
        priors = compute_geometric_priors(
            feats[:, 0], feats[:, 1], feats[:, 2], feats[:, 3], feats[:, 4], mask
        )

        # Normalize Features (in-place)
        feats_norm = normalize_data(feats)

        # Store
        X_out[i] = feats_norm
        priors_out[i] = priors
        ids_out[i] = event_ids[i]

        if has_targets:
            y_out[i, 0] = vec_az[i]
            y_out[i, 1] = vec_zen[i]

    # 3. Save to Cache
    np.save(path_X, X_out)
    np.save(path_priors, priors_out)
    np.save(path_ids, ids_out)

    result = {"X": X_out, "priors": priors_out, "ids": ids_out}

    if has_targets:
        np.save(path_y, y_out)
        result["y"] = y_out

    return result
