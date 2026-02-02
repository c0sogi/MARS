import numpy as np
import pandas as pd
import logging
import sys


def get_logger(name="idea_5"):
    """
    Creates a standard logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def ecef_to_lla(x, y, z):
    """
    Convert Earth-Centered, Earth-Fixed (ECEF) coordinates to
    Latitude, Longitude, Altitude (LLA).

    Args:
        x, y, z: ECEF coordinates in meters (numpy arrays or scalars)

    Returns:
        lat, lon, alt: Latitude (deg), Longitude (deg), Altitude (m)
    """
    # WGS84 ellipsoid constants
    a = 6378137.0
    e = 8.1819190842622e-2

    b = np.sqrt(a**2 * (1 - e**2))
    ep = np.sqrt((a**2 - b**2) / b**2)

    p = np.sqrt(x**2 + y**2)
    th = np.arctan2(a * z, b * p)

    lon = np.arctan2(y, x)
    lat = np.arctan2(
        (z + ep**2 * b * np.sin(th) ** 3), (p - e**2 * a * np.cos(th) ** 3)
    )

    # Calculate altitude
    N = a / np.sqrt(1 - e**2 * np.sin(lat) ** 2)
    alt = p / np.cos(lat) - N

    # Convert to degrees
    lat = np.degrees(lat)
    lon = np.degrees(lon)

    return lat, lon, alt


def lla_to_enu(lat, lon, alt, lat_ref, lon_ref, alt_ref):
    """
    Convert LLA coordinates to East-North-Up (ENU) local frame relative to a reference point.
    Uses a local flat earth approximation.

    Args:
        lat, lon, alt: Target coordinates (deg, deg, m)
        lat_ref, lon_ref, alt_ref: Reference coordinates (deg, deg, m)

    Returns:
        east, north, up: ENU coordinates in meters
    """
    # Constants
    DEG_TO_RAD = np.pi / 180.0

    # Coordinate differences
    d_lat = lat - lat_ref
    d_lon = lon - lon_ref

    # Conversion factors based on reference latitude
    lat_rad = lat_ref * DEG_TO_RAD

    # Meters per degree latitude (approximate)
    m_per_deg_lat = (
        111132.954 - 559.822 * np.cos(2 * lat_rad) + 1.175 * np.cos(4 * lat_rad)
    )

    # Meters per degree longitude (approximate)
    m_per_deg_lon = 111412.84 * np.cos(lat_rad) - 93.5 * np.cos(3 * lat_rad)

    north = d_lat * m_per_deg_lat
    east = d_lon * m_per_deg_lon
    up = alt - alt_ref

    return east, north, up


def enu_to_lla(east, north, up, lat_ref, lon_ref, alt_ref):
    """
    Convert ENU coordinates back to LLA relative to a reference point.

    Args:
        east, north, up: ENU coordinates in meters
        lat_ref, lon_ref, alt_ref: Reference coordinates (deg, deg, m)

    Returns:
        lat, lon, alt: LLA coordinates
    """
    DEG_TO_RAD = np.pi / 180.0

    lat_rad = lat_ref * DEG_TO_RAD

    # Meters per degree (same approximation as forward)
    m_per_deg_lat = (
        111132.954 - 559.822 * np.cos(2 * lat_rad) + 1.175 * np.cos(4 * lat_rad)
    )
    m_per_deg_lon = 111412.84 * np.cos(lat_rad) - 93.5 * np.cos(3 * lat_rad)

    d_lat = north / m_per_deg_lat
    d_lon = east / m_per_deg_lon

    lat = lat_ref + d_lat
    lon = lon_ref + d_lon
    alt = alt_ref + up

    return lat, lon, alt


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees).
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371000  # Radius of earth in meters
    return c * r


def compute_metric(df):
    """
    Computes the competition metric: mean of the 50th and 95th percentile distance errors.

    Args:
        df: DataFrame containing 'tripId', 'lat_pred', 'lon_pred', 'lat_gt', 'lon_gt'

    Returns:
        score: The computed metric
    """
    # Ensure distance is calculated
    if "dist" not in df.columns:
        df = df.copy()
        df["dist"] = haversine_distance(
            df["lat_pred"], df["lon_pred"], df["lat_gt"], df["lon_gt"]
        )

    score_list = []
    # Group by tripId to calculate percentiles per trip
    for trip_id, group in df.groupby("tripId"):
        dists = group["dist"].values
        p50 = np.percentile(dists, 50)
        p95 = np.percentile(dists, 95)
        score_list.append((p50 + p95) / 2)

    # Final score is the mean across all trips
    return np.mean(score_list)
