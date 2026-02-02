import numpy as np


def spherical_to_cartesian(azimuth, zenith):
    """
    Converts spherical coordinates to Cartesian unit vectors.

    The coordinate system is defined as:
    x = cos(azimuth) * sin(zenith)
    y = sin(azimuth) * sin(zenith)
    z = cos(zenith)

    Args:
        azimuth (float or np.ndarray): Azimuth angle in radians [0, 2*pi].
        zenith (float or np.ndarray): Zenith angle in radians [0, pi].

    Returns:
        tuple: (x, y, z) where each element is a float or np.ndarray corresponding
               to the Cartesian coordinates. The resulting vector is a unit vector.
    """
    sin_zenith = np.sin(zenith)
    x = np.cos(azimuth) * sin_zenith
    y = np.sin(azimuth) * sin_zenith
    z = np.cos(zenith)
    return x, y, z


def cartesian_to_spherical(x, y, z):
    """
    Converts Cartesian coordinates to spherical coordinates.

    Args:
        x (float or np.ndarray): X coordinate.
        y (float or np.ndarray): Y coordinate.
        z (float or np.ndarray): Z coordinate.

    Returns:
        tuple: (azimuth, zenith) in radians.
               azimuth is in range [0, 2*pi].
               zenith is in range [0, pi].
    """
    # Calculate radius (norm)
    r = np.sqrt(x**2 + y**2 + z**2)

    # Handle potential division by zero or very small vectors by adding epsilon
    # or relying on numpy's behavior (inf/nan), but here we assume valid direction vectors.
    # We use np.divide to handle array broadcasting safely if needed,
    # though standard operators work for matching shapes.

    # Zenith: arccos(z / r)
    # Clip argument to [-1, 1] to handle floating point errors slightly outside range
    zenith = np.arccos(np.clip(z / r, -1.0, 1.0))

    # Azimuth: arctan2(y, x) returns values in [-pi, pi]
    azimuth = np.arctan2(y, x)

    # Convert azimuth to [0, 2*pi] range
    if np.isscalar(azimuth):
        if azimuth < 0:
            azimuth += 2 * np.pi
    else:
        azimuth = np.where(azimuth < 0, azimuth + 2 * np.pi, azimuth)

    return azimuth, zenith


def angular_dist_score(true_azimuth, true_zenith, pred_azimuth, pred_zenith):
    """
    Computes the mean angular distance (error) between true and predicted directions.

    The angular distance is the angle between the two unit vectors defined by the
    spherical coordinates.

    Args:
        true_azimuth (np.ndarray or float): Ground truth azimuth angles in radians.
        true_zenith (np.ndarray or float): Ground truth zenith angles in radians.
        pred_azimuth (np.ndarray or float): Predicted azimuth angles in radians.
        pred_zenith (np.ndarray or float): Predicted zenith angles in radians.

    Returns:
        float: The mean angular error in radians.
    """
    # Convert spherical to cartesian unit vectors
    # Vector u (True)
    ux, uy, uz = spherical_to_cartesian(true_azimuth, true_zenith)

    # Vector v (Predicted)
    vx, vy, vz = spherical_to_cartesian(pred_azimuth, pred_zenith)

    # Compute dot product: u . v
    # Since both are unit vectors, u . v = cos(theta)
    dot_product = ux * vx + uy * vy + uz * vz

    # Clip values to [-1, 1] to prevent NaNs from floating point errors in arccos
    dot_product = np.clip(dot_product, -1.0, 1.0)

    # Calculate the angle
    angular_dist = np.arccos(dot_product)

    # Return the mean angular distance
    return np.mean(np.abs(angular_dist))
