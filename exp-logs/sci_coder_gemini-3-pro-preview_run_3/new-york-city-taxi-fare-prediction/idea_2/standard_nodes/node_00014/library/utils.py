import os
import random
import numpy as np
import torch
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculates the great-circle distance between two points on the earth (specified in decimal degrees).

    Args:
        lat1 (float or np.ndarray): Latitude of the first point(s).
        lon1 (float or np.ndarray): Longitude of the first point(s).
        lat2 (float or np.ndarray): Latitude of the second point(s).
        lon2 (float or np.ndarray): Longitude of the second point(s).

    Returns:
        float or np.ndarray: Distance between the points in kilometers.
    """
    # Radius of the Earth in kilometers
    R = 6371.0

    # Convert degrees to radians
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    # Haversine formula
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance = R * c
    return distance


def rotate_coordinates(x, y, angle_degrees):
    """
    Rotates 2D coordinates by a given angle.
    Useful for aligning map coordinates with the Manhattan street grid.

    Args:
        x (float or np.ndarray): X-coordinates (e.g., Longitude).
        y (float or np.ndarray): Y-coordinates (e.g., Latitude).
        angle_degrees (float): The rotation angle in degrees.

    Returns:
        tuple: (x_rotated, y_rotated)
    """
    theta = np.radians(angle_degrees)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # Rotation matrix application
    # x' = x*cos(theta) - y*sin(theta)
    # y' = x*sin(theta) + y*cos(theta)
    x_rot = x * cos_theta - y * sin_theta
    y_rot = x * sin_theta + y * cos_theta

    return x_rot, y_rot
