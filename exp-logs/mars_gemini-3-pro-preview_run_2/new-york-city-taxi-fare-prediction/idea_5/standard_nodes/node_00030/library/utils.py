import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed: int = Config.SEED):
    """
    Sets the random seed for various libraries to ensure reproducibility.

    Args:
        seed (int): The random seed to use. Defaults to Config.SEED.
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


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculates the Haversine (great-circle) distance between two points
    on the earth (specified in decimal degrees).

    Args:
        lat1, lon1: Latitude and Longitude of the first point (float or array-like).
        lat2, lon2: Latitude and Longitude of the second point (float or array-like).

    Returns:
        Distance between the two points in kilometers (float or array-like).
    """
    # Radius of earth in kilometers
    R = 6371.0

    # Convert decimal degrees to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    # Haversine formula
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance = R * c
    return distance


def manhattan_distance(lat1, lon1, lat2, lon2):
    """
    Calculates the Manhattan (L1) distance between two points.
    This is the sum of the absolute differences of their coordinates.

    Args:
        lat1, lon1: Latitude and Longitude of the first point.
        lat2, lon2: Latitude and Longitude of the second point.

    Returns:
        Manhattan distance in degrees (float or array-like).
    """
    return np.abs(lat1 - lat2) + np.abs(lon1 - lon2)
