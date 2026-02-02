import os
import numpy as np
import pandas as pd
import torch
from library.config import Config


def load_sensor_geometry(load_cached_data=True):
    """
    Loads the sensor geometry data (x, y, z coordinates for each sensor).

    Args:
        load_cached_data (bool): If True, attempts to load from a pre-saved parquet file.
                                 If False or file missing, loads from raw CSV and saves cache.

    Returns:
        pd.DataFrame: DataFrame indexed by 'sensor_id' with columns ['x', 'y', 'z'].
    """
    cache_dir = Config.IDEA_DIR
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "sensor_geometry_cache.parquet")

    # 1. Try to load cached data
    if load_cached_data and os.path.exists(cache_path):
        return pd.read_parquet(cache_path)

    # 2. Compute/Process from scratch
    geo_path = Config.GEOMETRY_PATH
    if not os.path.exists(geo_path):
        raise FileNotFoundError(f"Geometry file not found at {geo_path}")

    df = pd.read_csv(geo_path)

    # Ensure sensor_id is the index if it exists as a column
    if "sensor_id" in df.columns:
        df = df.set_index("sensor_id")

    # Select only coordinate columns and cast to float32 for memory efficiency
    df = df[["x", "y", "z"]].astype("float32")

    # 3. Save to cache
    df.to_parquet(cache_path)

    return df


def angles_to_vector(azimuth, zenith):
    """
    Converts azimuth and zenith angles to a 3D unit vector (x, y, z).
    Supports both numpy arrays and torch tensors.

    Args:
        azimuth: Azimuth angle in radians (0 to 2pi).
        zenith: Zenith angle in radians (0 to pi).

    Returns:
        (N, 3) array/tensor or (3,) array/tensor representing unit vectors.
    """
    is_torch = isinstance(azimuth, torch.Tensor)

    if is_torch:
        sin = torch.sin
        cos = torch.cos
        stack_fn = lambda x: torch.stack(x, dim=-1)
    else:
        sin = np.sin
        cos = np.cos
        stack_fn = lambda x: np.stack(x, axis=-1) if np.ndim(x[0]) > 0 else np.array(x)

    x = cos(azimuth) * sin(zenith)
    y = sin(azimuth) * sin(zenith)
    z = cos(zenith)

    return stack_fn([x, y, z])


def vector_to_angles(vectors):
    """
    Converts 3D unit vectors to azimuth and zenith angles.

    Args:
        vectors: Numpy array or Torch tensor of shape (N, 3).

    Returns:
        azimuth, zenith: Arrays/Tensors of angles in radians.
    """
    is_torch = isinstance(vectors, torch.Tensor)

    if is_torch:
        x = vectors[..., 0]
        y = vectors[..., 1]
        z = vectors[..., 2]

        # Ensure normalization
        norm = torch.sqrt(x**2 + y**2 + z**2)
        x = x / norm
        y = y / norm
        z = z / norm

        # Clip z for numerical stability in acos
        zenith = torch.acos(torch.clamp(z, -1.0, 1.0))
        azimuth = torch.atan2(y, x)

        # Map azimuth from [-pi, pi] to [0, 2pi]
        azimuth = torch.where(azimuth < 0, azimuth + 2 * np.pi, azimuth)

    else:
        x = vectors[..., 0]
        y = vectors[..., 1]
        z = vectors[..., 2]

        norm = np.linalg.norm(vectors, axis=-1)
        # Avoid division by zero
        norm = np.where(norm == 0, 1e-8, norm)

        x = x / norm
        y = y / norm
        z = z / norm

        zenith = np.arccos(np.clip(z, -1.0, 1.0))
        azimuth = np.arctan2(y, x)

        azimuth = np.where(azimuth < 0, azimuth + 2 * np.pi, azimuth)

    return azimuth, zenith


def angular_dist_score(az_true, zen_true, az_pred, zen_pred):
    """
    Computes the mean angular error between true and predicted directions.

    Args:
        az_true, zen_true: Ground truth angles (radians).
        az_pred, zen_pred: Predicted angles (radians).

    Returns:
        float: Mean angular error in radians.
    """
    # Ensure inputs are numpy arrays for metric calculation
    if isinstance(az_true, torch.Tensor):
        az_true = az_true.detach().cpu().numpy()
    if isinstance(zen_true, torch.Tensor):
        zen_true = zen_true.detach().cpu().numpy()
    if isinstance(az_pred, torch.Tensor):
        az_pred = az_pred.detach().cpu().numpy()
    if isinstance(zen_pred, torch.Tensor):
        zen_pred = zen_pred.detach().cpu().numpy()

    # Convert angles to unit vectors
    vec_true = angles_to_vector(az_true, zen_true)
    vec_pred = angles_to_vector(az_pred, zen_pred)

    # Compute dot product: v1 . v2
    # Since vectors are normalized, dot product is cosine of angle
    dot_prod = np.sum(vec_true * vec_pred, axis=-1)

    # Clip to handle numerical errors slightly outside [-1, 1]
    dot_prod = np.clip(dot_prod, -1.0, 1.0)

    # Angle is arccos of dot product
    angles = np.arccos(dot_prod)

    return float(np.mean(angles))
