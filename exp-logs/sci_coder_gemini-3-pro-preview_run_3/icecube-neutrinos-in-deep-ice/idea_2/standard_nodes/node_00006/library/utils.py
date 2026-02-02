import numpy as np


def spherical_to_cartesian(azimuth, zenith):
    """
    Convert spherical coordinates to Cartesian unit vector components.

    Formulas:
    x = cos(azimuth) * sin(zenith)
    y = sin(azimuth) * sin(zenith)
    z = cos(zenith)

    Args:
        azimuth (float or np.ndarray): Azimuth angle(s) in radians [0, 2*pi].
        zenith (float or np.ndarray): Zenith angle(s) in radians [0, pi].

    Returns:
        tuple: (x, y, z) components of the unit vector.
    """
    # Ensure inputs are numpy arrays
    azimuth = np.asarray(azimuth)
    zenith = np.asarray(zenith)

    sin_zenith = np.sin(zenith)

    x = np.cos(azimuth) * sin_zenith
    y = np.sin(azimuth) * sin_zenith
    z = np.cos(zenith)

    return x, y, z


def cartesian_to_spherical(x, y, z):
    """
    Convert Cartesian vector components to spherical coordinates.
    The resulting vector is normalized before conversion.

    Args:
        x (float or np.ndarray): x component.
        y (float or np.ndarray): y component.
        z (float or np.ndarray): z component.

    Returns:
        tuple: (azimuth, zenith) in radians.
               azimuth in [0, 2*pi], zenith in [0, pi].
    """
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    # Compute magnitude
    r = np.sqrt(x**2 + y**2 + z**2)

    # Handle zero magnitude to avoid division by zero
    r_safe = np.where(r == 0, 1.0, r)

    # Normalize components
    x_norm = x / r_safe
    y_norm = y / r_safe
    z_norm = z / r_safe

    # Zenith: arccos(z)
    # Clip to [-1, 1] to handle floating point errors slightly outside range
    z_norm = np.clip(z_norm, -1.0, 1.0)
    zenith = np.arccos(z_norm)

    # Azimuth: arctan2(y, x) -> returns values in [-pi, pi]
    azimuth = np.arctan2(y_norm, x_norm)

    # Convert range [-pi, pi] to [0, 2*pi]
    azimuth = np.where(azimuth < 0, azimuth + 2 * np.pi, azimuth)

    return azimuth, zenith


def compute_angular_error(true_azimuth, true_zenith, pred_azimuth, pred_zenith):
    """
    Compute the mean angular error between true and predicted directions.

    Args:
        true_azimuth (np.ndarray): Ground truth azimuth angles.
        true_zenith (np.ndarray): Ground truth zenith angles.
        pred_azimuth (np.ndarray): Predicted azimuth angles.
        pred_zenith (np.ndarray): Predicted zenith angles.

    Returns:
        float: The mean angular error in radians.
    """
    # Convert both sets of angles to unit vectors
    true_x, true_y, true_z = spherical_to_cartesian(true_azimuth, true_zenith)
    pred_x, pred_y, pred_z = spherical_to_cartesian(pred_azimuth, pred_zenith)

    # Dot product of unit vectors equals cos(angle)
    # u . v = x1*x2 + y1*y2 + z1*z2
    dot_prod = true_x * pred_x + true_y * pred_y + true_z * pred_z

    # Clip to valid cosine range [-1, 1] to prevent NaNs from floating point noise
    dot_prod = np.clip(dot_prod, -1.0, 1.0)

    # Angle is arccos of dot product
    angular_errors = np.arccos(dot_prod)

    return np.mean(angular_errors)
