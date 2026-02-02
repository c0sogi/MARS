import numpy as np
import pandas as pd
import torch
from library.config import SENSOR_GEO_PATH


def load_geometry():
    """
    Loads the sensor geometry data from the path defined in config.

    Returns:
        pd.DataFrame: DataFrame containing x, y, z coordinates indexed by sensor_id.
    """
    df = pd.read_csv(SENSOR_GEO_PATH)
    if "sensor_id" in df.columns:
        df = df.set_index("sensor_id")
    return df


def angles_to_direction(azimuth, zenith):
    """
    Converts azimuth and zenith angles to Cartesian unit vectors (x, y, z).
    Supports both numpy arrays and torch tensors.

    Args:
        azimuth: float, numpy array, or torch tensor (radians).
        zenith: float, numpy array, or torch tensor (radians).

    Returns:
        Tuple (x, y, z) of the same type as inputs.
    """
    # Check if inputs are torch tensors
    is_torch = isinstance(azimuth, torch.Tensor) or isinstance(zenith, torch.Tensor)

    if is_torch:
        sin_zenith = torch.sin(zenith)
        x = torch.cos(azimuth) * sin_zenith
        y = torch.sin(azimuth) * sin_zenith
        z = torch.cos(zenith)
    else:
        sin_zenith = np.sin(zenith)
        x = np.cos(azimuth) * sin_zenith
        y = np.sin(azimuth) * sin_zenith
        z = np.cos(zenith)

    return x, y, z


def direction_to_angles(x, y, z):
    """
    Converts Cartesian unit vectors (x, y, z) to azimuth and zenith angles.
    Supports both numpy arrays and torch tensors.

    Args:
        x, y, z: components of the vector. Can be scalars, numpy arrays, or torch tensors.

    Returns:
        Tuple (azimuth, zenith) in radians.
        Azimuth is in [0, 2*pi], Zenith is in [0, pi].
    """
    is_torch = isinstance(x, torch.Tensor)

    if is_torch:
        # Clip z to [-1, 1] to avoid numerical errors in arccos
        z = torch.clamp(z, -1.0, 1.0)
        zenith = torch.acos(z)
        azimuth = torch.atan2(y, x)
        # atan2 returns [-pi, pi], convert to [0, 2*pi]
        azimuth = torch.where(azimuth < 0, azimuth + 2 * torch.pi, azimuth)
    else:
        z = np.clip(z, -1.0, 1.0)
        zenith = np.arccos(z)
        azimuth = np.arctan2(y, x)
        azimuth = np.where(azimuth < 0, azimuth + 2 * np.pi, azimuth)

    return azimuth, zenith


def angular_dist_score(y_true, y_pred):
    """
    Calculates the Mean Angular Error between true and predicted angles.

    Args:
        y_true: array-like of shape (N, 2) containing [azimuth, zenith].
        y_pred: array-like of shape (N, 2) containing [azimuth, zenith].

    Returns:
        float: Mean angular error in radians.
    """
    # Convert inputs to numpy arrays if they aren't already
    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true)
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred)

    # Extract angles
    az_true, zen_true = y_true[:, 0], y_true[:, 1]
    az_pred, zen_pred = y_pred[:, 0], y_pred[:, 1]

    # Convert to unit vectors
    xt, yt, zt = angles_to_direction(az_true, zen_true)
    xp, yp, zp = angles_to_direction(az_pred, zen_pred)

    # Compute dot product: u . v
    dot_prod = xt * xp + yt * yp + zt * zp

    # Clip to [-1, 1] to avoid NaNs in arccos due to float precision errors
    dot_prod = np.clip(dot_prod, -1.0, 1.0)

    # Calculate angular distance
    angular_errors = np.arccos(dot_prod)

    return np.mean(angular_errors)
