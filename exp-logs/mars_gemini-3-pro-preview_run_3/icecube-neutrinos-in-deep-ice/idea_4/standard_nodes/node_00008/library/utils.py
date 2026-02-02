import numpy as np
import pandas as pd
import torch
import library.config as config


def load_sensor_geometry():
    """
    Loads the sensor geometry data.

    Returns:
        np.ndarray: A (N_sensors, 3) array containing x, y, z coordinates.
                    The index of the array corresponds to the sensor_id.
    """
    df = pd.read_csv(config.SENSOR_GEOMETRY_PATH)
    # The task description states: "The row index corresponds to the sensor_idx feature of pulses."
    # We assume the file is sorted or indexed correctly by row number 0..N-1.
    sensor_positions = df[["x", "y", "z"]].to_numpy().astype(np.float32)
    return sensor_positions


def spherical_to_cartesian(azimuth, zenith):
    """
    Converts spherical coordinates to cartesian unit vectors.

    Args:
        azimuth (float or np.ndarray): Azimuth angle in radians [0, 2pi].
        zenith (float or np.ndarray): Zenith angle in radians [0, pi].

    Returns:
        tuple: (x, y, z) components of the unit vector.
    """
    x = np.cos(azimuth) * np.sin(zenith)
    y = np.sin(azimuth) * np.sin(zenith)
    z = np.cos(zenith)
    return x, y, z


def cartesian_to_spherical(x, y, z):
    """
    Converts cartesian vectors to spherical coordinates.

    Args:
        x, y, z (float or np.ndarray): Cartesian components.

    Returns:
        tuple: (azimuth, zenith) in radians.
    """
    # Normalize vector to be safe
    r = np.sqrt(x**2 + y**2 + z**2)
    # Avoid division by zero
    r = np.where(r == 0, 1e-6, r)

    x_norm = x / r
    y_norm = y / r
    z_norm = z / r

    # Zenith: angle from +z axis. z = cos(zenith)
    # Clip to [-1, 1] to avoid numerical errors with arccos
    z_norm = np.clip(z_norm, -1.0, 1.0)
    zenith = np.arccos(z_norm)

    # Azimuth: angle in x-y plane.
    azimuth = np.arctan2(y_norm, x_norm)

    # Convert range [-pi, pi] to [0, 2pi]
    azimuth = np.where(azimuth < 0, azimuth + 2 * np.pi, azimuth)

    return azimuth, zenith


def compute_eigen_characteristics(pos, charge):
    """
    Computes the charge-weighted covariance matrix and its eigendecomposition
    to capture the global shape/direction of the event.

    Args:
        pos (np.ndarray): (N, 3) array of pulse positions (x, y, z).
        charge (np.ndarray): (N,) array of pulse charges.

    Returns:
        np.ndarray: A flat array of size 12 containing:
                    [eval1, eval2, eval3,
                     evec1_x, evec1_y, evec1_z,
                     evec2_x, evec2_y, evec2_z,
                     evec3_x, evec3_y, evec3_z]
                    Sorted by eigenvalue descending (primary axis first).
    """
    # Handle edge cases with insufficient points
    if pos.shape[0] < 2 or charge.sum() <= 0:
        return np.zeros(12, dtype=np.float32)

    # Normalize weights
    weights = charge / charge.sum()

    # Weighted Center of Mass
    mean_pos = np.average(pos, axis=0, weights=weights)

    # Center the positions
    centered_pos = pos - mean_pos

    # Weighted Covariance Matrix
    # Cov = sum(w_i * (p_i - mu)^T * (p_i - mu)) / (1 - sum(w^2)) for reliability,
    # but for feature extraction simple weighted average is sufficient.
    # Using np.cov with aweights handles this.
    try:
        cov = np.cov(centered_pos.T, aweights=weights)
    except Exception:
        return np.zeros(12, dtype=np.float32)

    # Eigendecomposition
    # eigh is for symmetric/hermitian matrices (covariance is symmetric)
    evals, evecs = np.linalg.eigh(cov)

    # Sort descending (largest eigenvalue first -> primary axis)
    idx = evals.argsort()[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    # Flatten eigenvectors (column-wise in evecs, so we take transpose to get rows)
    # evecs[:, 0] is the first eigenvector
    flat_evecs = evecs.T.flatten()

    # Concatenate
    result = np.concatenate([evals, flat_evecs])

    # Handle NaNs if any
    if np.isnan(result).any():
        return np.zeros(12, dtype=np.float32)

    return result.astype(np.float32)


def angular_dist_score(y_true, y_pred):
    """
    Computes the mean angular distance between true and predicted directions.

    Args:
        y_true (np.ndarray): Shape (N, 2) containing [azimuth, zenith] ground truth.
        y_pred (np.ndarray): Shape (N, 2) containing [azimuth, zenith] predictions.

    Returns:
        float: Mean angular error in radians.
    """
    # Extract components
    az_true = y_true[:, 0]
    zen_true = y_true[:, 1]

    az_pred = y_pred[:, 0]
    zen_pred = y_pred[:, 1]

    # Convert to cartesian unit vectors
    # We use the implemented function but handle the array structure
    x_t, y_t, z_t = spherical_to_cartesian(az_true, zen_true)
    x_p, y_p, z_p = spherical_to_cartesian(az_pred, zen_pred)

    # Dot product of unit vectors
    # u . v = |u||v|cos(theta) = 1*1*cos(theta) -> theta = arccos(u . v)
    dot_prod = x_t * x_p + y_t * y_p + z_t * z_p

    # Clip to numerical stability range [-1, 1]
    dot_prod = np.clip(dot_prod, -1.0, 1.0)

    # Angular distance
    errors = np.arccos(dot_prod)

    return np.mean(errors)
