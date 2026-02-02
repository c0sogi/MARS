import numpy as np
import pandas as pd
import torch
import os
from library.config import Config


def load_sensor_geometry(load_cached_data=True):
    """
    Loads the sensor geometry mapping.

    Args:
        load_cached_data (bool): If True, attempts to load from cache.

    Returns:
        np.ndarray: Array of shape (N_sensors, 3) containing (x, y, z) coordinates.
    """
    cache_path = Config.WORKING_DIR / "sensor_geometry_map.npy"

    if load_cached_data and os.path.exists(cache_path):
        return np.load(cache_path)

    # Load from CSV
    if not os.path.exists(Config.SENSOR_GEOMETRY_PATH):
        raise FileNotFoundError(
            f"Sensor geometry file not found at {Config.SENSOR_GEOMETRY_PATH}"
        )

    df = pd.read_csv(Config.SENSOR_GEOMETRY_PATH)

    # Extract x, y, z
    # The dataset description states: "The row index corresponds to the sensor_idx feature of pulses."
    required_cols = ["x", "y", "z"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(
            f"Sensor geometry CSV missing required columns: {required_cols}"
        )

    geometry = df[required_cols].values.astype(np.float32)

    # Save to cache
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    np.save(cache_path, geometry)

    return geometry


def direction_to_angles(direction_vector):
    """
    Converts a direction vector (x, y, z) to (azimuth, zenith).

    Args:
        direction_vector: (N, 3) or (3,) array/tensor.

    Returns:
        azimuth, zenith: Arrays/Tensors of shape (N,) or scalars.
    """
    if isinstance(direction_vector, torch.Tensor):
        x, y, z = (
            direction_vector[..., 0],
            direction_vector[..., 1],
            direction_vector[..., 2],
        )
        # Clip z to [-1, 1] to avoid NaN in acos due to float precision
        z = torch.clamp(z, -1.0, 1.0)
        zenith = torch.acos(z)
        azimuth = torch.atan2(y, x)
        # Map azimuth to [0, 2pi]
        azimuth = torch.where(azimuth < 0, azimuth + 2 * np.pi, azimuth)
        return azimuth, zenith
    else:
        x, y, z = (
            direction_vector[..., 0],
            direction_vector[..., 1],
            direction_vector[..., 2],
        )
        z = np.clip(z, -1.0, 1.0)
        zenith = np.arccos(z)
        azimuth = np.arctan2(y, x)
        azimuth[azimuth < 0] += 2 * np.pi
        return azimuth, zenith


def angles_to_direction(azimuth, zenith):
    """
    Converts (azimuth, zenith) to a direction vector (x, y, z).

    Args:
        azimuth, zenith: Arrays/Tensors.

    Returns:
        direction_vector: (N, 3) or (3,) array/tensor.
    """
    if isinstance(azimuth, torch.Tensor):
        sin_zen = torch.sin(zenith)
        x = torch.cos(azimuth) * sin_zen
        y = torch.sin(azimuth) * sin_zen
        z = torch.cos(zenith)
        return torch.stack([x, y, z], dim=-1)
    else:
        sin_zen = np.sin(zenith)
        x = np.cos(azimuth) * sin_zen
        y = np.sin(azimuth) * sin_zen
        z = np.cos(zenith)
        return np.stack([x, y, z], axis=-1)


def compute_canonical_frame(pulse_x, pulse_y, pulse_z, pulse_time, pulse_charge):
    """
    Computes the rotation matrix R that aligns the event's principal axis with Z
    and ensures the time gradient is positive along Z.

    Args:
        pulse_x, pulse_y, pulse_z: (N,) arrays of coordinates.
        pulse_time: (N,) array of pulse times.
        pulse_charge: (N,) array of pulse charges (weights).

    Returns:
        R: (3, 3) rotation matrix.
    """
    # Ensure inputs are numpy arrays
    pulse_x = np.asarray(pulse_x)
    pulse_y = np.asarray(pulse_y)
    pulse_z = np.asarray(pulse_z)
    pulse_time = np.asarray(pulse_time)
    pulse_charge = np.asarray(pulse_charge)

    n_pulses = len(pulse_x)
    if n_pulses < 3:
        return np.eye(3, dtype=np.float32)

    positions = np.stack([pulse_x, pulse_y, pulse_z], axis=1)  # (N, 3)
    weights = pulse_charge
    sum_weights = np.sum(weights)

    if sum_weights <= 1e-6:
        return np.eye(3, dtype=np.float32)

    # 1. Weighted Center of Mass
    center = np.average(positions, axis=0, weights=weights)
    centered_pos = positions - center

    # 2. Weighted Covariance Matrix
    # C = (1/sum_w) * sum(w_i * (x_i - c)(x_i - c)^T)
    # Efficient calculation: (W * X).T @ X where W is diag(weights)
    weighted_centered = centered_pos * weights[:, np.newaxis]
    cov = (centered_pos.T @ weighted_centered) / sum_weights

    # 3. SVD / Eigendecomposition
    # Covariance is symmetric, svd and eigh are equivalent. SVD is generally robust.
    try:
        U, S, Vh = np.linalg.svd(cov)
    except np.linalg.LinAlgError:
        return np.eye(3, dtype=np.float32)

    # U columns are eigenvectors corresponding to sorted singular values (largest first)
    # Principal axis (track direction) is usually the first eigenvector for tracks
    e1 = U[:, 0]
    e2 = U[:, 1]
    e3 = U[:, 2]

    # 4. Direction Ambiguity Resolution
    # Project pulses onto the principal axis
    projections = centered_pos @ e1

    # Check correlation with time.
    # If particle moves along e1, time should increase as projection increases.
    if np.std(projections) > 1e-9 and np.std(pulse_time) > 1e-9:
        cov_pt = np.cov(projections, pulse_time)[0, 1]
        if cov_pt < 0:
            e1 = -e1
            e2 = -e2  # Flip e2 to maintain handedness (or we fix it later with det)

    # 5. Construct Rotation Matrix
    # We want to map:
    # e1 -> Z (0, 0, 1)
    # e2 -> X (1, 0, 0)
    # e3 -> Y (0, 1, 0)
    #
    # The rotation matrix R satisfies v_new = R * v_old.
    # Therefore R * e1 = [0, 0, 1]^T
    #           R * e2 = [1, 0, 0]^T
    #           R * e3 = [0, 1, 0]^T
    #
    # This implies R = [e2, e3, e1]^T (rows are e2, e3, e1)
    R = np.vstack([e2, e3, e1])

    # 6. Ensure Right-Handed Coordinate System (det(R) = 1)
    if np.linalg.det(R) < 0:
        # Flip the second axis (mapped to Y) to invert determinant
        R[1, :] = -R[1, :]

    return R.astype(np.float32)
