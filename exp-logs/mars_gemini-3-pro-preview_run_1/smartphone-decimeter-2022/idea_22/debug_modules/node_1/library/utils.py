import numpy as np
import torch
import math
import os
from library.config import Config

# WGS84 Ellipsoid Constants derived from Config
A = Config.WGS84_A
B = Config.WGS84_B
E2 = 1 - (B**2 / A**2)  # Squared eccentricity


def get_radii_of_curvature(lat_rad):
    """
    Computes the Meridional Radius (Rm) and Prime Vertical Radius (Rn)
    at a given latitude using WGS84 ellipsoid constants.

    Args:
        lat_rad (float or np.array): Latitude in radians.

    Returns:
        tuple: (Rm, Rn) in meters.
    """
    sin_lat = np.sin(lat_rad)
    sin2_lat = sin_lat**2

    # Prime Vertical Radius
    Rn = A / np.sqrt(1 - E2 * sin2_lat)

    # Meridional Radius
    Rm = A * (1 - E2) / np.power(1 - E2 * sin2_lat, 1.5)

    return Rm, Rn


def wgs84_to_cartesian(lat, lon, ref_lat, ref_lon):
    """
    Converts WGS84 coordinates to local Cartesian offsets (East, North)
    relative to a reference point.

    Args:
        lat (float or np.array): Target latitude in degrees.
        lon (float or np.array): Target longitude in degrees.
        ref_lat (float or np.array): Reference latitude in degrees.
        ref_lon (float or np.array): Reference longitude in degrees.

    Returns:
        tuple: (east, north) in meters.
    """
    # Convert to radians
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    ref_lat_rad = np.deg2rad(ref_lat)
    ref_lon_rad = np.deg2rad(ref_lon)

    # Calculate deltas
    delta_lat = lat_rad - ref_lat_rad
    delta_lon = lon_rad - ref_lon_rad

    # Get radii of curvature at the reference latitude
    Rm, Rn = get_radii_of_curvature(ref_lat_rad)

    # Calculate North and East offsets
    north = delta_lat * Rm
    east = delta_lon * Rn * np.cos(ref_lat_rad)

    return east, north


def cartesian_to_wgs84(east, north, ref_lat, ref_lon):
    """
    Converts local Cartesian offsets (East, North) back to WGS84 coordinates.

    Args:
        east (float or np.array): East offset in meters.
        north (float or np.array): North offset in meters.
        ref_lat (float or np.array): Reference latitude in degrees.
        ref_lon (float or np.array): Reference longitude in degrees.

    Returns:
        tuple: (latitude, longitude) in degrees.
    """
    ref_lat_rad = np.deg2rad(ref_lat)

    # Get radii of curvature at the reference latitude
    Rm, Rn = get_radii_of_curvature(ref_lat_rad)

    # Calculate delta radians
    delta_lat_rad = north / Rm
    delta_lon_rad = east / (Rn * np.cos(ref_lat_rad))

    # Convert back to degrees
    lat = ref_lat + np.rad2deg(delta_lat_rad)
    lon = ref_lon + np.rad2deg(delta_lon_rad)

    return lat, lon


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculates the great-circle distance between two points on the Earth surface.

    Args:
        lat1, lon1: First point coordinates in degrees.
        lat2, lon2: Second point coordinates in degrees.

    Returns:
        float or np.array: Distance in meters.
    """
    R = 6371000.0  # Mean Earth radius in meters

    phi1 = np.deg2rad(lat1)
    phi2 = np.deg2rad(lat2)
    delta_phi = np.deg2rad(lat2 - lat1)
    delta_lambda = np.deg2rad(lon2 - lon1)

    a = (
        np.sin(delta_phi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    )

    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance = R * c
    return distance


def save_checkpoint(model, optimizer, epoch, best_score, path):
    """
    Saves the model state, optimizer state, and training metadata.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer.
        epoch (int): Current epoch.
        best_score (float): Best validation score so far.
        path (str): Path to save the checkpoint.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_score": best_score,
    }
    torch.save(state, path)
    # print(f"Checkpoint saved to {path}")


def load_checkpoint(model, optimizer, path, device="cpu"):
    """
    Loads a checkpoint into the model and optimizer.

    Args:
        model (torch.nn.Module): The model instance.
        optimizer (torch.optim.Optimizer): The optimizer instance.
        path (str): Path to the checkpoint file.
        device (str): Device to map location to.

    Returns:
        tuple: (start_epoch, best_score)
    """
    if not os.path.exists(path):
        print(f"No checkpoint found at {path}")
        return 0, float("inf")

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    best_score = checkpoint.get("best_score", float("inf"))

    print(f"Loaded checkpoint from {path} (Epoch {epoch}, Score {best_score:.4f})")
    return epoch, best_score
