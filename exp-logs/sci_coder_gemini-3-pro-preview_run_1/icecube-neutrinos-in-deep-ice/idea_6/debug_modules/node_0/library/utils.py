import os
import random
import hashlib
import json
import numpy as np
import pandas as pd
import torch
from library.config import Config


def set_seed(seed=42):
    """
    Sets fixed random seeds for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic algorithms where possible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_config_hash(params_dict):
    """
    Generates a unique hash based on a dictionary of parameters.
    Used for caching version control.
    """
    # Sort keys to ensure deterministic hashing
    try:
        params_str = json.dumps(params_dict, sort_keys=True)
    except TypeError:
        # Fallback for non-JSON serializable types by using string representation
        # This handles cases where values might be types or other objects
        params_str = str(sorted(params_dict.items()))

    return hashlib.md5(params_str.encode("utf-8")).hexdigest()


def load_sensor_geometry(path=None):
    """
    Loads the sensor geometry data.
    Returns a pandas DataFrame indexed by sensor_id.
    """
    if path is None:
        path = Config.SENSOR_GEO_PATH

    if not os.path.exists(path):
        raise FileNotFoundError(f"Sensor geometry file not found at {path}")

    df = pd.read_csv(path)

    # Ensure sensor_id is the index for easy lookups
    if "sensor_id" in df.columns:
        df = df.set_index("sensor_id")

    # Ensure columns are x, y, z
    if not all(col in df.columns for col in ["x", "y", "z"]):
        raise ValueError("Sensor geometry file must contain x, y, z columns")

    return df


def spherical_to_cartesian(azimuth, zenith):
    """
    Converts spherical coordinates (azimuth, zenith) to cartesian (x, y, z).
    Inputs can be scalars, numpy arrays, or torch tensors.

    Formulas:
    x = cos(azimuth) * sin(zenith)
    y = sin(azimuth) * sin(zenith)
    z = cos(zenith)
    """
    # Detect type to use appropriate math library
    if isinstance(azimuth, torch.Tensor) or isinstance(zenith, torch.Tensor):
        sin = torch.sin
        cos = torch.cos
        stack = torch.stack
    else:
        sin = np.sin
        cos = np.cos
        stack = lambda x, axis: np.stack(x, axis=axis)

    x = cos(azimuth) * sin(zenith)
    y = sin(azimuth) * sin(zenith)
    z = cos(zenith)

    # If inputs are scalar (numpy 0-d array or python float), return tuple
    if np.ndim(x) == 0 and not isinstance(x, torch.Tensor):
        return x, y, z
    if isinstance(x, torch.Tensor) and x.ndim == 0:
        return x, y, z

    # If inputs are arrays, stack them into (N, 3)
    return stack([x, y, z], axis=-1)


def cartesian_to_spherical(x, y, z):
    """
    Converts cartesian coordinates (x, y, z) to spherical (azimuth, zenith).
    Inputs can be scalars, numpy arrays, or torch tensors.

    Returns:
        azimuth: [0, 2pi]
        zenith: [0, pi]
    """
    # Detect type
    is_torch = (
        isinstance(x, torch.Tensor)
        or isinstance(y, torch.Tensor)
        or isinstance(z, torch.Tensor)
    )

    if is_torch:
        sqrt = torch.sqrt
        acos = torch.acos
        atan2 = torch.atan2
        clip = torch.clamp
        pi = np.pi
        where = torch.where
    else:
        sqrt = np.sqrt
        acos = np.arccos
        atan2 = np.arctan2
        clip = np.clip
        pi = np.pi
        where = np.where

    # Normalize vector
    norm = sqrt(x**2 + y**2 + z**2)

    # Avoid division by zero
    if is_torch:
        norm = torch.max(
            norm,
            torch.tensor(1e-8, device=norm.device if hasattr(norm, "device") else None),
        )
    else:
        norm = np.maximum(norm, 1e-8)

    x_n = x / norm
    y_n = y / norm
    z_n = z / norm

    # Zenith
    # Clip to [-1, 1] to avoid NaNs in arccos
    z_n = clip(z_n, -1.0, 1.0)
    zenith = acos(z_n)

    # Azimuth
    azimuth = atan2(y_n, x_n)
    # Map to [0, 2pi]
    azimuth = where(azimuth < 0, azimuth + 2 * pi, azimuth)

    return azimuth, zenith


def angular_dist_score(y_true, y_pred):
    """
    Computes the mean angular error between true and predicted directions.

    Args:
        y_true: Array-like of shape (N, 2) containing [azimuth, zenith]
        y_pred: Array-like of shape (N, 2) containing [azimuth, zenith]

    Returns:
        float: Mean angular error in radians.
    """
    # Convert to numpy if tensors
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Extract components
    az_true, zen_true = y_true[:, 0], y_true[:, 1]
    az_pred, zen_pred = y_pred[:, 0], y_pred[:, 1]

    # Convert to cartesian unit vectors
    # spherical_to_cartesian returns (N, 3) for array inputs
    vec_true = spherical_to_cartesian(az_true, zen_true)
    vec_pred = spherical_to_cartesian(az_pred, zen_pred)

    # Compute dot product: v_true . v_pred
    # sum(a*b) over last axis
    dot_product = np.sum(vec_true * vec_pred, axis=1)

    # Clip to valid range for arccos [-1, 1]
    dot_product = np.clip(dot_product, -1.0, 1.0)

    # Calculate angle
    angular_errors = np.arccos(dot_product)

    # Return mean
    return float(np.mean(np.abs(angular_errors)))
