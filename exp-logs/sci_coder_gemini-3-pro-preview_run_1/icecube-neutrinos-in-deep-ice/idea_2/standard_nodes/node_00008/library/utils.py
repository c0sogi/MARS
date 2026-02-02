import numpy as np
import pandas as pd
import os
from library.config import GEOMETRY_PATH


def load_sensor_geometry():
    """
    Loads the sensor geometry data.

    Returns:
        pd.DataFrame: DataFrame containing x, y, z coordinates indexed by sensor_id.
    """
    if not os.path.exists(GEOMETRY_PATH):
        raise FileNotFoundError(f"Geometry file not found at {GEOMETRY_PATH}")

    df = pd.read_csv(GEOMETRY_PATH)
    # Ensure sensor_id is the index for easy lookup
    if "sensor_id" in df.columns:
        df = df.set_index("sensor_id")

    return df


def spherical_to_cartesian(azimuth, zenith):
    """
    Converts spherical coordinates to Cartesian unit vectors.

    Args:
        azimuth (np.ndarray or float): Azimuth angle in radians (0 to 2pi).
        zenith (np.ndarray or float): Zenith angle in radians (0 to pi).

    Returns:
        tuple: (x, y, z) components of the unit vector.
    """
    # Ensure inputs are numpy arrays for element-wise operations if they are lists
    azimuth = np.array(azimuth)
    zenith = np.array(zenith)

    x = np.cos(azimuth) * np.sin(zenith)
    y = np.sin(azimuth) * np.sin(zenith)
    z = np.cos(zenith)

    return x, y, z


def cartesian_to_spherical(x, y, z):
    """
    Converts Cartesian vectors to spherical coordinates.

    Args:
        x (np.ndarray or float): X component.
        y (np.ndarray or float): Y component.
        z (np.ndarray or float): Z component.

    Returns:
        tuple: (azimuth, zenith) in radians.
               Azimuth is in [0, 2pi), Zenith is in [0, pi].
    """
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    # Normalize the vector to ensure it is a unit vector
    norm = np.sqrt(x**2 + y**2 + z**2)
    # Avoid division by zero
    norm = np.where(norm == 0, 1e-8, norm)

    x_norm = x / norm
    y_norm = y / norm
    z_norm = z / norm

    # Zenith: angle from +z axis. Range [0, pi]
    # Clip to [-1, 1] to avoid numerical errors in arccos
    z_clipped = np.clip(z_norm, -1.0, 1.0)
    zenith = np.arccos(z_clipped)

    # Azimuth: angle in x-y plane. arctan2 returns (-pi, pi]
    azimuth = np.arctan2(y_norm, x_norm)

    # Convert azimuth range from (-pi, pi] to [0, 2pi)
    azimuth = np.where(azimuth < 0, azimuth + 2 * np.pi, azimuth)

    return azimuth, zenith


def angular_dist_score(y_true, y_pred):
    """
    Calculates the mean angular error between true and predicted directions.

    Args:
        y_true (pd.DataFrame or dict): Contains keys/columns 'azimuth' and 'zenith'.
        y_pred (pd.DataFrame or dict): Contains keys/columns 'azimuth' and 'zenith'.

    Returns:
        float: Mean angular error in radians.
    """
    # Extract angles
    true_az = np.array(y_true["azimuth"])
    true_zen = np.array(y_true["zenith"])
    pred_az = np.array(y_pred["azimuth"])
    pred_zen = np.array(y_pred["zenith"])

    # Convert to unit vectors
    true_x, true_y, true_z = spherical_to_cartesian(true_az, true_zen)
    pred_x, pred_y, pred_z = spherical_to_cartesian(pred_az, pred_zen)

    # Compute dot product: u . v
    # Since both are unit vectors, dot product is cos(theta)
    dot_product = (true_x * pred_x) + (true_y * pred_y) + (true_z * pred_z)

    # Clip to valid range [-1, 1] for arccos
    dot_product = np.clip(dot_product, -1.0, 1.0)

    # Calculate angular distance
    angular_dist = np.arccos(dot_product)

    # Return mean error
    return np.mean(angular_dist)
