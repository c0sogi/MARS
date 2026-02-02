import json
import hashlib
import numpy as np


def spherical_to_cartesian(azimuth, zenith):
    """
    Converts spherical coordinates (azimuth, zenith) to Cartesian coordinates (x, y, z).

    Args:
        azimuth (float or np.ndarray): Azimuth angle in radians (0 to 2*pi).
        zenith (float or np.ndarray): Zenith angle in radians (0 to pi).

    Returns:
        tuple: (x, y, z) Cartesian coordinates. Returns unit vectors.
    """
    x = np.cos(azimuth) * np.sin(zenith)
    y = np.sin(azimuth) * np.sin(zenith)
    z = np.cos(zenith)
    return x, y, z


def cartesian_to_spherical(x, y, z):
    """
    Converts Cartesian coordinates (x, y, z) to spherical coordinates (azimuth, zenith).

    Args:
        x (float or np.ndarray): x-coordinate.
        y (float or np.ndarray): y-coordinate.
        z (float or np.ndarray): z-coordinate.

    Returns:
        tuple: (azimuth, zenith) in radians.
               azimuth is in [0, 2*pi), zenith is in [0, pi].
    """
    # Normalize vector to ensure numerical stability for arccos
    r = np.sqrt(x**2 + y**2 + z**2)
    # Avoid division by zero
    r = np.where(r == 0, 1.0, r)

    z_norm = z / r
    z_norm = np.clip(z_norm, -1.0, 1.0)

    zenith = np.arccos(z_norm)

    azimuth = np.arctan2(y, x)
    # Map azimuth from (-pi, pi] to [0, 2*pi)
    azimuth = np.where(azimuth < 0, azimuth + 2 * np.pi, azimuth)

    return azimuth, zenith


def angular_dist_score(y_true, y_pred):
    """
    Calculates the mean angular error between true and predicted directions.

    Args:
        y_true (np.ndarray): Array of shape (N, 2) containing true [azimuth, zenith].
        y_pred (np.ndarray): Array of shape (N, 2) containing predicted [azimuth, zenith].

    Returns:
        float: Mean angular error in radians.
    """
    # Extract angles
    az_true = y_true[:, 0]
    zen_true = y_true[:, 1]

    az_pred = y_pred[:, 0]
    zen_pred = y_pred[:, 1]

    # Convert to unit vectors
    x_true, y_true_c, z_true = spherical_to_cartesian(az_true, zen_true)
    x_pred, y_pred_c, z_pred = spherical_to_cartesian(az_pred, zen_pred)

    # Compute dot product: u . v = |u||v| cos(theta). Since unit vectors, |u|=|v|=1.
    dot_product = x_true * x_pred + y_true_c * y_pred_c + z_true * z_pred

    # Clip to handle numerical errors slightly outside [-1, 1]
    dot_product = np.clip(dot_product, -1.0, 1.0)

    # Calculate angle
    angular_errors = np.arccos(dot_product)

    return np.mean(np.abs(angular_errors))


def get_config_hash(config_dict):
    """
    Generates a unique MD5 hash for a configuration dictionary.
    Useful for caching preprocessing steps based on parameters.

    Args:
        config_dict (dict): Dictionary containing configuration parameters.

    Returns:
        str: Hexadecimal hash string.
    """
    # Filter out keys that shouldn't affect the data cache (like num_workers or paths)
    # However, for safety, we usually hash the specific subset passed to this function.
    # We assume the caller passes the relevant dict.

    # Sort keys to ensure consistent ordering
    config_str = json.dumps(config_dict, sort_keys=True, default=str)

    # Create hash
    md5_hash = hashlib.md5(config_str.encode("utf-8")).hexdigest()

    return md5_hash
