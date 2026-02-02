import json
import hashlib
import numpy as np
import torch


def get_config_hash(config_dict):
    """
    Generates a unique MD5 hash based on a dictionary of configuration parameters.
    Used to version cached datasets.

    Args:
        config_dict (dict): Dictionary containing configuration parameters.

    Returns:
        str: MD5 hash string.
    """
    # Sort keys to ensure deterministic hashing regardless of insertion order
    config_str = json.dumps(config_dict, sort_keys=True)
    return hashlib.md5(config_str.encode("utf-8")).hexdigest()


def azimuth_zenith_to_vector(azimuth, zenith):
    """
    Converts spherical coordinates to a 3D Cartesian unit vector.

    Coordinate system:
    x = cos(azimuth) * sin(zenith)
    y = sin(azimuth) * sin(zenith)
    z = cos(zenith)

    Args:
        azimuth (np.ndarray or torch.Tensor): Azimuth angle in radians.
        zenith (np.ndarray or torch.Tensor): Zenith angle in radians.

    Returns:
        np.ndarray or torch.Tensor: Array of shape (..., 3) containing (x, y, z).
    """
    if isinstance(azimuth, torch.Tensor):
        sin_z = torch.sin(zenith)
        x = torch.cos(azimuth) * sin_z
        y = torch.sin(azimuth) * sin_z
        z = torch.cos(zenith)
        return torch.stack([x, y, z], dim=-1)
    else:
        sin_z = np.sin(zenith)
        x = np.cos(azimuth) * sin_z
        y = np.sin(azimuth) * sin_z
        z = np.cos(zenith)
        return np.stack([x, y, z], axis=-1)


def vector_to_azimuth_zenith(vec):
    """
    Converts a 3D Cartesian unit vector to spherical coordinates.

    Args:
        vec (np.ndarray or torch.Tensor): Input vectors of shape (..., 3).

    Returns:
        tuple: (azimuth, zenith) in radians.
    """
    if isinstance(vec, torch.Tensor):
        x, y, z = vec[..., 0], vec[..., 1], vec[..., 2]

        # Zenith: z = cos(zenith) -> zenith = acos(z)
        # Clip to [-1, 1] to avoid NaNs due to float precision errors
        z = torch.clamp(z, -1.0, 1.0)
        zenith = torch.acos(z)

        # Azimuth: tan(azimuth) = y/x -> azimuth = atan2(y, x)
        azimuth = torch.atan2(y, x)
        # Map [-pi, pi] to [0, 2pi]
        azimuth = torch.where(azimuth < 0, azimuth + 2 * np.pi, azimuth)

        return azimuth, zenith
    else:
        x, y, z = vec[..., 0], vec[..., 1], vec[..., 2]

        # Clip to ensure numerical stability
        z = np.clip(z, -1.0, 1.0)
        zenith = np.arccos(z)

        azimuth = np.arctan2(y, x)
        # Map [-pi, pi] to [0, 2pi]
        azimuth[azimuth < 0] += 2 * np.pi

        return azimuth, zenith


def angular_dist_score(y_true, y_pred):
    """
    Computes the mean angular error between true and predicted directions.

    Args:
        y_true (np.ndarray): Array of shape (N, 2) containing true [azimuth, zenith].
        y_pred (np.ndarray): Array of shape (N, 2) containing predicted [azimuth, zenith].

    Returns:
        float: Mean angular error in radians.
    """
    # Convert angles to unit vectors
    v_true = azimuth_zenith_to_vector(y_true[:, 0], y_true[:, 1])
    v_pred = azimuth_zenith_to_vector(y_pred[:, 0], y_pred[:, 1])

    # Compute dot product: a . b = |a||b|cos(theta)
    # Since vectors are normalized, |a| = |b| = 1, so a . b = cos(theta)
    dot_prod = np.sum(v_true * v_pred, axis=1)

    # Clip to handle potential floating point errors slightly outside [-1, 1]
    dot_prod = np.clip(dot_prod, -1.0, 1.0)

    # Calculate angle
    angles = np.arccos(dot_prod)

    return np.mean(angles)
