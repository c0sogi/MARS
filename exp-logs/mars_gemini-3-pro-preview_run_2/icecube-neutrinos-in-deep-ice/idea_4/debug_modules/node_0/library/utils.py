import numpy as np
import logging
import sys
import os


def setup_logger(
    name: str = "logger", log_file: str = None, level: int = logging.INFO
) -> logging.Logger:
    """
    Sets up a logger that outputs to console and optionally to a file.

    Args:
        name: Name of the logger.
        log_file: Path to the log file. If None, no file logging is performed.
        level: Logging level (default: logging.INFO).

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers to avoid duplicates if function is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    if log_file:
        # Ensure directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def spherical_to_cartesian(azimuth, zenith):
    """
    Converts spherical coordinates (azimuth, zenith) to Cartesian unit vectors (x, y, z).

    Args:
        azimuth: Array-like or scalar, azimuth angle in radians [0, 2*pi].
        zenith: Array-like or scalar, zenith angle in radians [0, pi].

    Returns:
        x, y, z: Cartesian coordinates (normalized).
    """
    # Ensure inputs are numpy arrays for vectorized operations
    azimuth = np.asarray(azimuth)
    zenith = np.asarray(zenith)

    sin_zenith = np.sin(zenith)

    x = np.cos(azimuth) * sin_zenith
    y = np.sin(azimuth) * sin_zenith
    z = np.cos(zenith)

    return x, y, z


def cartesian_to_spherical(x, y, z):
    """
    Converts Cartesian coordinates (x, y, z) to spherical coordinates (azimuth, zenith).
    Handles normalization of the input vector to ensure unit length before conversion.

    Args:
        x, y, z: Array-like or scalar, Cartesian coordinates.

    Returns:
        azimuth: Angle in radians [0, 2*pi].
        zenith: Angle in radians [0, pi].
    """
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    # Normalize the vector
    r = np.sqrt(x**2 + y**2 + z**2)

    # Initialize normalized arrays
    # Use a small epsilon or mask to avoid division by zero
    mask = r > 0

    x_norm = np.zeros_like(x, dtype=float)
    y_norm = np.zeros_like(y, dtype=float)
    z_norm = np.zeros_like(z, dtype=float)

    # Handle scalar vs array input for normalization
    if r.ndim == 0:
        if r > 0:
            x_norm = x / r
            y_norm = y / r
            z_norm = z / r
    else:
        x_norm[mask] = x[mask] / r[mask]
        y_norm[mask] = y[mask] / r[mask]
        z_norm[mask] = z[mask] / r[mask]

    # Zenith: arccos(z)
    # Clip z to [-1, 1] to avoid numerical errors slightly outside range
    z_norm = np.clip(z_norm, -1.0, 1.0)
    zenith = np.arccos(z_norm)

    # Azimuth: arctan2(y, x) -> [-pi, pi]
    azimuth = np.arctan2(y_norm, x_norm)

    # Convert azimuth range from [-pi, pi] to [0, 2*pi]
    if azimuth.ndim == 0:
        if azimuth < 0:
            azimuth += 2 * np.pi
    else:
        azimuth[azimuth < 0] += 2 * np.pi

    return azimuth, zenith
