import os
import numpy as np
import pandas as pd

# WGS84 Ellipsoid Constants
A = 6378137.0
E = 8.1819190842622e-2
M_PER_DEG_LAT = 111319.9


def ecef_to_lla(x, y, z):
    """
    Convert Earth-Centered Earth-Fixed (ECEF) coordinates to Latitude, Longitude, Altitude.
    Vectorized implementation using WGS84 constants.

    Args:
        x, y, z: ECEF coordinates in meters (scalars or numpy arrays).

    Returns:
        lat, lon, alt: Latitude (degrees), Longitude (degrees), Altitude (meters).
    """
    # Ensure inputs are numpy arrays for vectorized operations
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    asq = A**2
    esq = E**2

    b = np.sqrt(asq * (1 - esq))
    bsq = b**2
    ep = np.sqrt((asq - bsq) / bsq)

    p = np.sqrt(x**2 + y**2)
    th = np.arctan2(A * z, b * p)

    lon = np.arctan2(y, x)
    lat = np.arctan2((z + ep**2 * b * np.sin(th) ** 3), (p - esq * A * np.cos(th) ** 3))

    # Calculate altitude
    # N is the radius of curvature in the prime vertical
    N = A / np.sqrt(1 - esq * np.sin(lat) ** 2)
    alt = p / np.cos(lat) - N

    # Convert radians to degrees
    lat = np.degrees(lat)
    lon = np.degrees(lon)

    return lat, lon, alt


def lla_to_enu(lat, lon, alt, ref_lat, ref_lon, ref_alt):
    """
    Convert LLA coordinates to Local Metric ENU (East, North, Up) relative to a reference point.
    Used to calculate residuals (targets) in meters.

    Args:
        lat, lon, alt: Target coordinates (degrees, degrees, meters).
        ref_lat, ref_lon, ref_alt: Reference/Baseline coordinates (degrees, degrees, meters).

    Returns:
        east, north, up: Distance in meters from reference to target.
    """
    lat_rad = np.radians(ref_lat)
    m_per_deg_lon = M_PER_DEG_LAT * np.cos(lat_rad)

    delta_lat = lat - ref_lat
    delta_lon = lon - ref_lon

    north = delta_lat * M_PER_DEG_LAT
    east = delta_lon * m_per_deg_lon
    up = alt - ref_alt

    return east, north, up


def enu_to_lla(east, north, up, ref_lat, ref_lon, ref_alt):
    """
    Convert Local Metric ENU coordinates back to LLA relative to a reference point.
    Used to reconstruct absolute position from predicted residuals.

    Args:
        east, north, up: Residuals in meters.
        ref_lat, ref_lon, ref_alt: Reference/Baseline coordinates.

    Returns:
        lat, lon, alt: Reconstructed coordinates.
    """
    lat_rad = np.radians(ref_lat)
    m_per_deg_lon = M_PER_DEG_LAT * np.cos(lat_rad)

    delta_lat = north / M_PER_DEG_LAT
    delta_lon = east / m_per_deg_lon

    lat = ref_lat + delta_lat
    lon = ref_lon + delta_lon
    alt = ref_alt + up

    return lat, lon, alt


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points on the earth (specified in decimal degrees).

    Args:
        lat1, lon1: First point coordinates in degrees.
        lat2, lon2: Second point coordinates in degrees.

    Returns:
        Distance in meters.
    """
    R = 6371000  # Radius of earth in meters

    # Convert decimal degrees to radians
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c


def save_submission(df, path):
    """
    Save the submission dataframe to a CSV file, ensuring the directory exists.

    Args:
        df: pandas DataFrame containing the submission.
        path: File path to save the CSV.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Submission saved to {path}")
