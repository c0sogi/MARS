import os
import json
import hashlib
import random
import numpy as np
import pandas as pd
import torch
from library.config import Config


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_sensor_geometry():
    """
    Loads the sensor geometry data from the path defined in Config.

    Returns:
        pd.DataFrame: DataFrame containing sensor_id, x, y, z coordinates.
    """
    if not os.path.exists(Config.GEOMETRY_PATH):
        raise FileNotFoundError(f"Geometry file not found at {Config.GEOMETRY_PATH}")

    df = pd.read_csv(Config.GEOMETRY_PATH)
    return df


def azimuth_zenith_to_vector(azimuth, zenith):
    """
    Converts azimuth and zenith angles to a unit vector (x, y, z).
    Supports both numpy arrays and torch tensors.

    Args:
        azimuth: Angle in radians (0 to 2pi).
        zenith: Angle in radians (0 to pi).

    Returns:
        (..., 3) array/tensor of x, y, z components.
    """
    # Determine if input is torch tensor or numpy array
    is_torch = isinstance(azimuth, torch.Tensor) or isinstance(zenith, torch.Tensor)

    if is_torch:
        if not isinstance(azimuth, torch.Tensor):
            azimuth = torch.tensor(azimuth)
        if not isinstance(zenith, torch.Tensor):
            zenith = torch.tensor(zenith)

        x = torch.cos(azimuth) * torch.sin(zenith)
        y = torch.sin(azimuth) * torch.sin(zenith)
        z = torch.cos(zenith)
        return torch.stack([x, y, z], dim=-1)
    else:
        x = np.cos(azimuth) * np.sin(zenith)
        y = np.sin(azimuth) * np.sin(zenith)
        z = np.cos(zenith)
        return np.stack([x, y, z], axis=-1)


def vector_to_azimuth_zenith(vectors):
    """
    Converts 3D vectors to azimuth and zenith angles.
    Automatically normalizes input vectors to unit length before conversion.

    Args:
        vectors: (..., 3) array/tensor of x, y, z components.

    Returns:
        Tuple (azimuth, zenith) with same shape as input batch.
        Azimuth is in [0, 2pi), Zenith is in [0, pi].
    """
    is_torch = isinstance(vectors, torch.Tensor)

    if is_torch:
        # Normalize to ensure unit vectors
        norm = torch.norm(vectors, p=2, dim=-1, keepdim=True)
        # Add epsilon to avoid division by zero
        vectors_norm = vectors / (norm + 1e-8)

        x = vectors_norm[..., 0]
        y = vectors_norm[..., 1]
        z = vectors_norm[..., 2]

        # Clip for numerical stability in acos
        z = torch.clamp(z, -1.0, 1.0)

        zenith = torch.acos(z)
        azimuth = torch.atan2(y, x)

        # Convert range [-pi, pi] to [0, 2pi)
        azimuth = torch.where(azimuth < 0, azimuth + 2 * np.pi, azimuth)

        return azimuth, zenith
    else:
        # Normalize
        norm = np.linalg.norm(vectors, axis=-1, keepdims=True)
        vectors_norm = vectors / (norm + 1e-8)

        x = vectors_norm[..., 0]
        y = vectors_norm[..., 1]
        z = vectors_norm[..., 2]

        z = np.clip(z, -1.0, 1.0)

        zenith = np.arccos(z)
        azimuth = np.arctan2(y, x)

        # Convert range [-pi, pi] to [0, 2pi)
        azimuth = np.where(azimuth < 0, azimuth + 2 * np.pi, azimuth)

        return azimuth, zenith


def get_config_hash():
    """
    Generates a unique MD5 hash based on the current configuration parameters
    that affect data processing (sampling, scaling, seeds).

    Returns:
        str: Hex digest of the configuration hash.
    """
    # Dictionary of parameters that affect data generation/caching
    config_state = {
        "MAX_PULSES": Config.MAX_PULSES,
        "EARLY_PULSES": Config.EARLY_PULSES,
        "TIME_SCALE": Config.TIME_SCALE,
        "COORD_SCALE": Config.COORD_SCALE,
        "SEED": Config.SEED,
    }

    # Serialize to JSON with sort_keys=True to ensure deterministic string
    config_str = json.dumps(config_state, sort_keys=True)

    # Compute MD5
    return hashlib.md5(config_str.encode("utf-8")).hexdigest()
