import os
import random
import numpy as np
import torch
import pandas as pd


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculates the Haversine distance between two sets of coordinates.
    Inputs can be scalars or numpy arrays.

    Args:
        lat1, lon1: Latitude and Longitude of the first point(s) in degrees.
        lat2, lon2: Latitude and Longitude of the second point(s) in degrees.

    Returns:
        float or np.ndarray: Distance in meters.
    """
    R = 6371000  # Radius of Earth in meters

    # Convert degrees to radians
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c


def ecef_to_lla(x, y, z):
    """
    Converts ECEF coordinates (meters) to Latitude, Longitude, Altitude (WGS84).

    Args:
        x, y, z: ECEF coordinates in meters (scalars or numpy arrays).

    Returns:
        tuple: (Latitude in degrees, Longitude in degrees, Altitude in meters)
    """
    # WGS84 ellipsoid constants
    a = 6378137.0
    f = 1.0 / 298.257223563
    b = a * (1.0 - f)
    e2 = (a**2 - b**2) / a**2
    ep2 = (a**2 - b**2) / b**2

    p = np.sqrt(x**2 + y**2)
    th = np.arctan2(a * z, b * p)

    lon = np.arctan2(y, x)
    lat = np.arctan2(z + ep2 * b * np.sin(th) ** 3, p - e2 * a * np.cos(th) ** 3)

    N = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    alt = p / np.cos(lat) - N

    # Convert radians to degrees
    lat = np.degrees(lat)
    lon = np.degrees(lon)

    return lat, lon, alt


def get_drive_id(file_path):
    """
    Extracts the drive_id from a file path.
    Assumes path structure like: .../drive_id/phone_name/...

    Args:
        file_path (str): The file path.

    Returns:
        str: The extracted drive_id.
    """
    parts = os.path.normpath(file_path).split(os.sep)
    # Search for the part that looks like a date-based drive ID if strict indexing fails
    # But based on the metadata generation script, it's consistently 3 levels up from the file
    if len(parts) >= 3:
        return parts[-3]
    return None


def get_phone_name(file_path):
    """
    Extracts the phone_name from a file path.

    Args:
        file_path (str): The file path.

    Returns:
        str: The extracted phone_name.
    """
    parts = os.path.normpath(file_path).split(os.sep)
    if len(parts) >= 2:
        return parts[-2]
    return None


def compute_metric(df_pred, df_gt):
    """
    Computes the competition metric: mean of the 50th and 95th percentile distance errors,
    averaged across phones.

    Args:
        df_pred (pd.DataFrame): Predictions with ['tripId', 'UnixTimeMillis', 'LatitudeDegrees', 'LongitudeDegrees']
        df_gt (pd.DataFrame): Ground truth with ['tripId', 'UnixTimeMillis', 'LatitudeDegrees', 'LongitudeDegrees']

    Returns:
        float: The calculated score.
    """
    # Create copies to avoid side effects
    pred = df_pred.copy()
    gt = df_gt.copy()

    # Ensure tripId exists or construct it
    if (
        "tripId" not in pred.columns
        and "drive_id" in pred.columns
        and "phone_name" in pred.columns
    ):
        pred["tripId"] = (
            pred["drive_id"].astype(str) + "-" + pred["phone_name"].astype(str)
        )

    if (
        "tripId" not in gt.columns
        and "drive_id" in gt.columns
        and "phone_name" in gt.columns
    ):
        gt["tripId"] = gt["drive_id"].astype(str) + "-" + gt["phone_name"].astype(str)

    # Rename columns for clarity in merge
    # Check if standard names exist, if not assume they might be named differently or handle key error
    pred_lat_col = (
        "LatitudeDegrees" if "LatitudeDegrees" in pred.columns else "lat_pred"
    )
    pred_lon_col = (
        "LongitudeDegrees" if "LongitudeDegrees" in pred.columns else "lon_pred"
    )

    gt_lat_col = "LatitudeDegrees" if "LatitudeDegrees" in gt.columns else "lat_gt"
    gt_lon_col = "LongitudeDegrees" if "LongitudeDegrees" in gt.columns else "lon_gt"

    pred = pred.rename(columns={pred_lat_col: "lat_pred", pred_lon_col: "lon_pred"})
    gt = gt.rename(columns={gt_lat_col: "lat_gt", gt_lon_col: "lon_gt"})

    # Merge on tripId and UnixTimeMillis
    merged = pd.merge(pred, gt, on=["tripId", "UnixTimeMillis"], how="inner")

    if len(merged) == 0:
        return np.nan

    # Calculate distances
    merged["dist"] = haversine_distance(
        merged["lat_pred"], merged["lon_pred"], merged["lat_gt"], merged["lon_gt"]
    )

    # Calculate metric per phone (tripId)
    scores = []
    for _, group in merged.groupby("tripId"):
        dists = group["dist"].values
        p50 = np.percentile(dists, 50)
        p95 = np.percentile(dists, 95)
        scores.append((p50 + p95) / 2)

    return np.mean(scores)
