import numpy as np
import torch
from library.config import set_seed


def seed_everything(seed):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Wraps the configuration's set_seed function.

    Args:
        seed (int): The seed value to set.
    """
    set_seed(seed)


def angles_to_vector(azimuth, zenith):
    """
    Converts spherical coordinates (azimuth, zenith) to a Cartesian unit vector (x, y, z).

    Args:
        azimuth: Angle in radians [0, 2*pi]. Can be scalar or numpy array.
        zenith: Angle in radians [0, pi]. Can be scalar or numpy array.

    Returns:
        np.ndarray: Array of shape (N, 3) if inputs are arrays, or (3,) if scalar.
                    The vector is normalized (unit length).
    """
    # Ensure inputs are numpy arrays for consistent math operations
    azimuth = np.asarray(azimuth)
    zenith = np.asarray(zenith)

    sin_zenith = np.sin(zenith)

    x = np.cos(azimuth) * sin_zenith
    y = np.sin(azimuth) * sin_zenith
    z = np.cos(zenith)

    # Stack along the last axis to create (..., 3) vectors
    return np.stack([x, y, z], axis=-1)


def vector_to_angles(x, y, z):
    """
    Converts a Cartesian unit vector (x, y, z) to spherical coordinates (azimuth, zenith).

    Args:
        x, y, z: Components of the unit vector. Can be scalars or numpy arrays.

    Returns:
        Tuple (azimuth, zenith) in radians.
        azimuth range: [0, 2*pi]
        zenith range: [0, pi]
    """
    # Ensure inputs are numpy arrays
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    # Clip z to [-1, 1] to handle potential floating point noise
    z = np.clip(z, -1.0, 1.0)

    # Calculate zenith
    zenith = np.arccos(z)

    # Calculate azimuth
    azimuth = np.arctan2(y, x)

    # Convert azimuth range from (-pi, pi] to [0, 2*pi]
    azimuth = np.where(azimuth < 0, azimuth + 2 * np.pi, azimuth)

    return azimuth, zenith


def angular_dist_score(y_true, y_pred):
    """
    Calculates the mean angular error between true and predicted directions.

    Args:
        y_true: Array-like of shape (N, 2) containing [azimuth, zenith] for ground truth.
        y_pred: Array-like of shape (N, 2) containing [azimuth, zenith] for predictions.

    Returns:
        float: The mean angular error in radians.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Extract angles
    az_true = y_true[:, 0]
    zen_true = y_true[:, 1]

    az_pred = y_pred[:, 0]
    zen_pred = y_pred[:, 1]

    # Convert spherical coordinates to unit vectors
    # True vectors
    sin_zen_t = np.sin(zen_true)
    x_t = np.cos(az_true) * sin_zen_t
    y_t = np.sin(az_true) * sin_zen_t
    z_t = np.cos(zen_true)

    # Predicted vectors
    sin_zen_p = np.sin(zen_pred)
    x_p = np.cos(az_pred) * sin_zen_p
    y_p = np.sin(az_pred) * sin_zen_p
    z_p = np.cos(zen_pred)

    # Calculate dot product: u . v
    # Since vectors are normalized, dot product equals cos(theta)
    dot_prod = x_t * x_p + y_t * y_p + z_t * z_p

    # Clip dot product to [-1, 1] to avoid NaNs in arccos due to precision errors
    dot_prod = np.clip(dot_prod, -1.0, 1.0)

    # Calculate angular distance (theta = arccos(u . v))
    angular_errors = np.arccos(dot_prod)

    # Return mean angular error
    return np.mean(angular_errors)
