import os
import sys
import logging
import numpy as np
import torch
from library.config import Config


def get_config_hash():
    """
    Wrapper for Config.get_config_hash() to ensure consistency.
    Returns a unique hash string based on the current configuration.
    """
    return Config.get_config_hash()


def setup_logger(log_file, level=logging.INFO):
    """
    Sets up a logger that writes to both console and a file.

    Args:
        log_file (str): Path to the log file.
        level (int): Logging level (default: logging.INFO).

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Create the directory for the log file if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    # Create a custom logger
    # Use file path as name to allow multiple independent loggers
    logger = logging.getLogger(name=log_file)
    logger.setLevel(level)

    # Avoid adding handlers multiple times if the logger is reused
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create handlers
    c_handler = logging.StreamHandler(sys.stdout)
    f_handler = logging.FileHandler(log_file, mode="w")

    c_handler.setLevel(level)
    f_handler.setLevel(level)

    # Create formatters and add them to handlers
    format_str = "%(asctime)s - %(levelname)s - %(message)s"
    c_format = logging.Formatter(format_str)
    f_format = logging.Formatter(format_str)

    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger


def wls_to_meters(wls_lat, wls_lon, gt_lat, gt_lon):
    """
    Converts Ground Truth Lat/Lon to relative meters (dLat, dLon)
    based on the WLS Baseline Lat/Lon using simple element-wise scaling.

    Args:
        wls_lat: Baseline Latitude (degrees), array-like
        wls_lon: Baseline Longitude (degrees), array-like
        gt_lat: Ground Truth Latitude (degrees), array-like
        gt_lon: Ground Truth Longitude (degrees), array-like

    Returns:
        d_lat_m: Delta Latitude in meters
        d_lon_m: Delta Longitude in meters
    """
    # Ensure inputs are numpy arrays for vectorized operations
    wls_lat = np.array(wls_lat)
    wls_lon = np.array(wls_lon)
    gt_lat = np.array(gt_lat)
    gt_lon = np.array(gt_lon)

    d_lat_deg = gt_lat - wls_lat
    d_lon_deg = gt_lon - wls_lon

    d_lat_m = d_lat_deg * Config.DEG_TO_M_LAT
    # Scale longitude difference by cosine of latitude
    # Use wls_lat as the reference for the scaling factor
    d_lon_m = d_lon_deg * Config.DEG_TO_M_LAT * np.cos(np.radians(wls_lat))

    return d_lat_m, d_lon_m


def meters_to_wls(wls_lat, wls_lon, d_lat_m, d_lon_m):
    """
    Converts relative meters (dLat, dLon) back to absolute Lat/Lon
    based on the WLS Baseline.

    Args:
        wls_lat: Baseline Latitude (degrees), array-like
        wls_lon: Baseline Longitude (degrees), array-like
        d_lat_m: Predicted Delta Latitude in meters, array-like
        d_lon_m: Predicted Delta Longitude in meters, array-like

    Returns:
        pred_lat: Predicted Latitude (degrees)
        pred_lon: Predicted Longitude (degrees)
    """
    wls_lat = np.array(wls_lat)
    wls_lon = np.array(wls_lon)
    d_lat_m = np.array(d_lat_m)
    d_lon_m = np.array(d_lon_m)

    d_lat_deg = d_lat_m / Config.DEG_TO_M_LAT

    # Avoid division by zero or extreme values near poles
    cos_lat = np.cos(np.radians(wls_lat))
    cos_lat = np.clip(cos_lat, 1e-6, 1.0)

    d_lon_deg = d_lon_m / (Config.DEG_TO_M_LAT * cos_lat)

    pred_lat = wls_lat + d_lat_deg
    pred_lon = wls_lon + d_lon_deg

    return pred_lat, pred_lon


def haversine_loss(y_true, y_pred, wls_lat, wls_lon):
    """
    Calculates the Haversine distance between true and predicted coordinates.
    This function first reconstructs the absolute coordinates from the
    relative meter predictions and then computes the distance.

    Can handle both numpy arrays and torch tensors (converts to numpy).

    Args:
        y_true: Ground Truth residuals (dLat_m, dLon_m) - shape (N, 2)
        y_pred: Predicted residuals (dLat_m, dLon_m) - shape (N, 2)
        wls_lat: Baseline Latitude (degrees) - shape (N,)
        wls_lon: Baseline Longitude (degrees) - shape (N,)

    Returns:
        distances: Array of haversine distances in meters
    """
    # Ensure inputs are numpy arrays
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(wls_lat, torch.Tensor):
        wls_lat = wls_lat.detach().cpu().numpy()
    if isinstance(wls_lon, torch.Tensor):
        wls_lon = wls_lon.detach().cpu().numpy()

    # Reconstruct absolute coordinates
    # y_true[:, 0] is dLat_m, y_true[:, 1] is dLon_m
    true_lat, true_lon = meters_to_wls(wls_lat, wls_lon, y_true[:, 0], y_true[:, 1])
    pred_lat, pred_lon = meters_to_wls(wls_lat, wls_lon, y_pred[:, 0], y_pred[:, 1])

    # Haversine formula
    R = 6371000  # Radius of Earth in meters

    phi1 = np.radians(true_lat)
    phi2 = np.radians(pred_lat)
    dphi = np.radians(pred_lat - true_lat)
    dlambda = np.radians(pred_lon - true_lon)

    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    # Clip 'a' to [0, 1] to avoid domain errors in sqrt due to float precision
    a = np.clip(a, 0, 1)

    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distances = R * c
    return distances
