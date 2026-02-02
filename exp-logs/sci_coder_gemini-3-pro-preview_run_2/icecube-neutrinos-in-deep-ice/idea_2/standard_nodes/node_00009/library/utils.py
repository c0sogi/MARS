import os
import sys
import logging
import random
import numpy as np
import pandas as pd
import torch
from library.config import SENSOR_GEO_PATH, SEED


def seed_everything(seed: int = SEED):
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


def setup_logger(log_file: str):
    """
    Configures and returns a logger that writes to both console and a file.

    Args:
        log_file (str): Path to the log file.

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Ensure directory exists
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Clear existing handlers to prevent duplicate logging
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # File Handler
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Stream Handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def load_sensor_geometry(path: str = SENSOR_GEO_PATH) -> pd.DataFrame:
    """
    Loads the sensor geometry CSV file.

    Args:
        path (str): Path to the sensor_geometry.csv file.

    Returns:
        pd.DataFrame: DataFrame containing sensor geometry with 'sensor_id' as index.
                      Columns: ['x', 'y', 'z'] (float32)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Geometry file not found at: {path}")

    df = pd.read_csv(path)

    # Set sensor_id as index for efficient lookups
    if "sensor_id" in df.columns:
        df.set_index("sensor_id", inplace=True)

    # Ensure coordinates are float32
    for col in ["x", "y", "z"]:
        if col in df.columns:
            df[col] = df[col].astype(np.float32)

    return df


def vector_to_angles(vectors):
    """
    Converts 3D direction vectors to azimuth and zenith angles.
    Supports both NumPy arrays and PyTorch tensors.

    Args:
        vectors: Input array/tensor of shape (N, 3) representing (x, y, z).

    Returns:
        azimuth, zenith: Arrays/Tensors of shape (N,) containing angles in radians.
                         Azimuth range: [0, 2*pi]
                         Zenith range: [0, pi]
    """
    is_torch = isinstance(vectors, torch.Tensor)

    if is_torch:
        # Extract components
        x = vectors[:, 0]
        y = vectors[:, 1]
        z = vectors[:, 2]

        # Normalize vectors to unit length
        norm = torch.sqrt(x**2 + y**2 + z**2)
        norm = torch.clamp(norm, min=1e-8)  # Avoid division by zero

        x = x / norm
        y = y / norm
        z = z / norm

        # Calculate Zenith: arccos(z)
        # Clip z to [-1, 1] to handle numerical errors
        z = torch.clamp(z, -1.0, 1.0)
        zenith = torch.acos(z)

        # Calculate Azimuth: arctan2(y, x)
        azimuth = torch.atan2(y, x)

        # Convert Azimuth from [-pi, pi] to [0, 2*pi]
        azimuth = torch.where(azimuth < 0, azimuth + 2 * torch.pi, azimuth)

        return azimuth, zenith

    else:
        # NumPy implementation
        x = vectors[:, 0]
        y = vectors[:, 1]
        z = vectors[:, 2]

        norm = np.linalg.norm(vectors, axis=1)
        norm = np.maximum(norm, 1e-8)

        x = x / norm
        y = y / norm
        z = z / norm

        z = np.clip(z, -1.0, 1.0)
        zenith = np.arccos(z)

        azimuth = np.arctan2(y, x)
        azimuth = np.where(azimuth < 0, azimuth + 2 * np.pi, azimuth)

        return azimuth, zenith
