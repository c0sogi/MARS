import os
import numpy as np
import pandas as pd
import torch
from library.config import Config


def load_sensor_geometry(load_cached_data=True):
    """
    Loads the sensor geometry mapping from sensor_id to (x, y, z) coordinates.
    Implements caching to avoid repeated CSV parsing.

    Args:
        load_cached_data (bool): If True, attempts to load from the cached .npy file first.

    Returns:
        np.ndarray: A numpy array of shape (max_sensor_id + 1, 3) where the index
                    corresponds to the sensor_id and the value is [x, y, z].
    """
    # Ensure the working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    cache_path = os.path.join(Config.WORKING_DIR, "sensor_geometry_map.npy")

    # 1. Attempt to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            geometry_map = np.load(cache_path)
            return geometry_map
        except Exception as e:
            # Fallback to recomputing if cache is corrupt
            pass

    # 2. Compute from scratch
    if not os.path.exists(Config.SENSOR_GEOMETRY_PATH):
        raise FileNotFoundError(
            f"Sensor geometry file not found at {Config.SENSOR_GEOMETRY_PATH}"
        )

    df = pd.read_csv(Config.SENSOR_GEOMETRY_PATH)

    # Determine the mapping strategy based on available columns
    # If 'sensor_id' is present, use it as the index. Otherwise, assume row index is sensor_id.
    if "sensor_id" in df.columns:
        max_id = df["sensor_id"].max()
        # Initialize array with zeros (or NaNs)
        geometry_map = np.zeros((max_id + 1, 3), dtype=np.float32)

        ids = df["sensor_id"].values
        coords = df[["x", "y", "z"]].values
        geometry_map[ids] = coords.astype(np.float32)
    else:
        # Implicit indexing
        geometry_map = df[["x", "y", "z"]].values.astype(np.float32)

    # 3. Save to cache
    try:
        np.save(cache_path, geometry_map)
    except Exception:
        # Non-critical failure if cache cannot be written
        pass

    return geometry_map


def spherical_to_cartesian(azimuth, zenith):
    """
    Converts spherical coordinates to Cartesian unit vectors.
    Supports both NumPy arrays and PyTorch tensors.

    Args:
        azimuth: Angle in radians [0, 2pi].
        zenith: Angle in radians [0, pi].

    Returns:
        tuple: (x, y, z) components.
    """
    if isinstance(azimuth, torch.Tensor):
        sin_zen = torch.sin(zenith)
        x = torch.cos(azimuth) * sin_zen
        y = torch.sin(azimuth) * sin_zen
        z = torch.cos(zenith)
    else:
        sin_zen = np.sin(zenith)
        x = np.cos(azimuth) * sin_zen
        y = np.sin(azimuth) * sin_zen
        z = np.cos(zenith)

    return x, y, z


def cartesian_to_spherical(x, y, z):
    """
    Converts Cartesian coordinates to spherical coordinates.
    Supports both NumPy arrays and PyTorch tensors.

    Args:
        x, y, z: Components of the direction vector.

    Returns:
        tuple: (azimuth, zenith) in radians.
    """
    if isinstance(x, torch.Tensor):
        r = torch.sqrt(x**2 + y**2 + z**2)
        # Clamp value for acos stability
        z_clamped = torch.clamp(z / (r + 1e-8), -1.0, 1.0)
        zenith = torch.acos(z_clamped)
        azimuth = torch.atan2(y, x)
        # Map azimuth from [-pi, pi] to [0, 2pi]
        azimuth = torch.where(azimuth < 0, azimuth + 2 * torch.pi, azimuth)
    else:
        r = np.sqrt(x**2 + y**2 + z**2)
        z_clamped = np.clip(z / (r + 1e-8), -1.0, 1.0)
        zenith = np.arccos(z_clamped)
        azimuth = np.arctan2(y, x)
        azimuth = np.where(azimuth < 0, azimuth + 2 * np.pi, azimuth)

    return azimuth, zenith


def angular_dist_score(y_true, y_pred):
    """
    Computes the mean angular error between true and predicted directions.
    This is the competition metric.

    Args:
        y_true: Array-like of shape (N, 2) containing (azimuth, zenith).
        y_pred: Array-like of shape (N, 2) containing (azimuth, zenith).

    Returns:
        float: Mean angular error in radians.
    """
    # Detach and convert to numpy if inputs are tensors
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    az_true, zen_true = y_true[:, 0], y_true[:, 1]
    az_pred, zen_pred = y_pred[:, 0], y_pred[:, 1]

    # Convert spherical to Cartesian unit vectors
    xt, yt, zt = spherical_to_cartesian(az_true, zen_true)
    xp, yp, zp = spherical_to_cartesian(az_pred, zen_pred)

    # Compute dot product (cosine of the angle)
    dot_prod = xt * xp + yt * yp + zt * zp

    # Clip to valid range [-1, 1] for numerical stability
    dot_prod = np.clip(dot_prod, -1.0, 1.0)

    # Calculate angular error
    angular_errors = np.arccos(dot_prod)

    return np.mean(angular_errors)
