import numpy as np
import pandas as pd
import torch
from library.config import Config


def load_sensor_geometry(path=Config.SENSOR_GEOMETRY_PATH):
    """
    Loads the sensor geometry data from the specified CSV path.

    Args:
        path (str): Path to the sensor_geometry.csv file.

    Returns:
        pd.DataFrame: DataFrame containing 'x', 'y', 'z' coordinates,
                      indexed by 'sensor_id'.
    """
    # Load the CSV
    df = pd.read_csv(path)

    # If sensor_id is present as a column, set it as index.
    # Otherwise, assume the row index corresponds to sensor_id.
    if "sensor_id" in df.columns:
        df = df.set_index("sensor_id")
    else:
        df.index.name = "sensor_id"

    # Return only the coordinate columns as float32
    return df[["x", "y", "z"]].astype(np.float32)


def angles_to_direction(azimuth, zenith):
    """
    Converts azimuth and zenith angles to a 3D unit direction vector (x, y, z).
    Supports both numpy arrays and torch tensors.

    Formulas:
        x = cos(azimuth) * sin(zenith)
        y = sin(azimuth) * sin(zenith)
        z = cos(zenith)

    Args:
        azimuth: Radians [0, 2pi]. Scalar, Array (N,), or Tensor (N,).
        zenith: Radians [0, pi]. Scalar, Array (N,), or Tensor (N,).

    Returns:
        Array or Tensor of shape (N, 3) containing (x, y, z) unit vectors.
    """
    # Determine backend (numpy or torch) based on input type
    if isinstance(azimuth, torch.Tensor) or isinstance(zenith, torch.Tensor):
        sin = torch.sin
        cos = torch.cos
        stack = torch.stack
        backend = "torch"
    else:
        sin = np.sin
        cos = np.cos
        stack = np.stack
        backend = "numpy"

    # Compute components
    x = cos(azimuth) * sin(zenith)
    y = sin(azimuth) * sin(zenith)
    z = cos(zenith)

    # Stack into vectors
    if backend == "torch":
        return stack([x, y, z], dim=-1)
    else:
        return stack([x, y, z], axis=-1)


def direction_to_angles(vectors):
    """
    Converts 3D unit vectors to azimuth and zenith angles.
    Supports both numpy arrays and torch tensors.

    Args:
        vectors: Array or Tensor of shape (N, 3) representing unit vectors (x, y, z).

    Returns:
        Tuple (azimuth, zenith) where:
            azimuth: Radians [0, 2pi]
            zenith: Radians [0, pi]
    """
    is_torch = isinstance(vectors, torch.Tensor)

    if is_torch:
        x = vectors[:, 0]
        y = vectors[:, 1]
        z = vectors[:, 2]
        norm = torch.norm
        atan2 = torch.atan2
        acos = torch.acos
        pi = np.pi
        clip = torch.clamp
        where = torch.where
    else:
        x = vectors[:, 0]
        y = vectors[:, 1]
        z = vectors[:, 2]
        norm = np.linalg.norm
        atan2 = np.arctan2
        acos = np.arccos
        pi = np.pi
        clip = np.clip
        where = np.where

    # Compute norm to ensure unit vectors (handling potential numerical drift)
    if is_torch:
        r = norm(vectors, dim=1)
        r = clip(r, min=1e-8)
    else:
        r = norm(vectors, axis=1)
        r = clip(r, 1e-8, None)

    # Zenith: arccos(z / r). Clip input to [-1, 1] for stability.
    z_scaled = z / r
    if is_torch:
        z_scaled = clip(z_scaled, -1.0, 1.0)
    else:
        z_scaled = clip(z_scaled, -1.0, 1.0)
    zenith = acos(z_scaled)

    # Azimuth: arctan2(y, x). Result is [-pi, pi].
    azimuth = atan2(y, x)

    # Convert Azimuth to [0, 2pi] range
    azimuth = where(azimuth < 0, azimuth + 2 * pi, azimuth)

    return azimuth, zenith


def angular_dist_score(y_true, y_pred):
    """
    Computes the Mean Angular Error between true and predicted angles.
    This is the evaluation metric for the task.

    Args:
        y_true: Array-like of shape (N, 2) containing true [azimuth, zenith].
        y_pred: Array-like of shape (N, 2) containing predicted [azimuth, zenith].

    Returns:
        float: The mean angular error in radians.
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Extract angles
    az_true, zen_true = y_true[:, 0], y_true[:, 1]
    az_pred, zen_pred = y_pred[:, 0], y_pred[:, 1]

    # Convert angles to 3D unit vectors
    vec_true = angles_to_direction(az_true, zen_true)
    vec_pred = angles_to_direction(az_pred, zen_pred)

    # Compute dot product between corresponding vectors
    # dot(u, v) = cos(theta)
    dot_products = np.sum(vec_true * vec_pred, axis=1)

    # Clip values to [-1, 1] to avoid NaNs in arccos due to float precision
    dot_products = np.clip(dot_products, -1.0, 1.0)

    # Calculate angle difference
    angular_errors = np.arccos(dot_products)

    # Return mean error
    return float(np.mean(np.abs(angular_errors)))
