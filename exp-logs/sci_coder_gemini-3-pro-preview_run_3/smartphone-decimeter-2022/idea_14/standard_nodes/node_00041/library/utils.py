import numpy as np
import pickle
import os

# --- WGS84 Ellipsoid Constants ---
A = 6378137.0  # Semi-major axis (meters)
B = 6356752.31424518  # Semi-minor axis (meters)
F = (A - B) / A  # Flattening
E_SQR = F * (2 - F)  # Eccentricity squared


def geodetic_to_ecef(lat, lon, alt):
    """
    Convert geodetic coordinates (Latitude, Longitude, Altitude) to ECEF (X, Y, Z).

    Args:
        lat: Latitude in degrees.
        lon: Longitude in degrees.
        alt: Altitude in meters.

    Returns:
        x, y, z: ECEF coordinates in meters.
    """
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    N = A / np.sqrt(1 - E_SQR * np.sin(lat_rad) ** 2)

    x = (N + alt) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (N + alt) * np.cos(lat_rad) * np.sin(lon_rad)
    z = (N * (1 - E_SQR) + alt) * np.sin(lat_rad)

    return x, y, z


def ecef_to_geodetic(x, y, z):
    """
    Convert ECEF coordinates (X, Y, Z) to geodetic (Latitude, Longitude, Altitude).
    Uses Ferrari's solution for high precision.

    Args:
        x, y, z: ECEF coordinates in meters.

    Returns:
        lat, lon, alt: Geodetic coordinates (degrees, degrees, meters).
    """
    # Ellipsoid constants
    a = A
    b = B
    e2 = E_SQR
    ep2 = (a**2 - b**2) / b**2

    p = np.sqrt(x**2 + y**2)
    theta = np.arctan2(z * a, p * b)

    lon = np.arctan2(y, x)
    lat = np.arctan2(z + ep2 * b * np.sin(theta) ** 3, p - e2 * a * np.cos(theta) ** 3)

    N = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    alt = p / np.cos(lat) - N

    return np.degrees(lat), np.degrees(lon), alt


def ecef_to_enu(x, y, z, lat0, lon0, alt0):
    """
    Convert ECEF coordinates to Local East-North-Up (ENU) coordinates relative to a reference point.

    Args:
        x, y, z: Target ECEF coordinates.
        lat0, lon0, alt0: Reference geodetic coordinates.

    Returns:
        e, n, u: ENU coordinates in meters.
    """
    x0, y0, z0 = geodetic_to_ecef(lat0, lon0, alt0)

    dx = x - x0
    dy = y - y0
    dz = z - z0

    lat0_rad = np.radians(lat0)
    lon0_rad = np.radians(lon0)

    sin_lat = np.sin(lat0_rad)
    cos_lat = np.cos(lat0_rad)
    sin_lon = np.sin(lon0_rad)
    cos_lon = np.cos(lon0_rad)

    e = -sin_lon * dx + cos_lon * dy
    n = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
    u = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz

    return e, n, u


def enu_to_ecef(e, n, u, lat0, lon0, alt0):
    """
    Convert Local ENU coordinates to ECEF coordinates relative to a reference point.

    Args:
        e, n, u: ENU coordinates in meters.
        lat0, lon0, alt0: Reference geodetic coordinates.

    Returns:
        x, y, z: ECEF coordinates in meters.
    """
    x0, y0, z0 = geodetic_to_ecef(lat0, lon0, alt0)

    lat0_rad = np.radians(lat0)
    lon0_rad = np.radians(lon0)

    sin_lat = np.sin(lat0_rad)
    cos_lat = np.cos(lat0_rad)
    sin_lon = np.sin(lon0_rad)
    cos_lon = np.cos(lon0_rad)

    # Inverse rotation (Transpose of the rotation matrix used in ECEF to ENU)
    dx = -sin_lon * e - sin_lat * cos_lon * n + cos_lat * cos_lon * u
    dy = cos_lon * e - sin_lat * sin_lon * n + cos_lat * sin_lon * u
    dz = cos_lat * n + sin_lat * u

    x = x0 + dx
    y = y0 + dy
    z = z0 + dz

    return x, y, z


def enu_to_geodetic(e, n, u, lat0, lon0, alt0):
    """
    Convert Local ENU coordinates to Geodetic coordinates.
    Wrapper that chains ENU -> ECEF -> Geodetic.

    Args:
        e, n, u: ENU coordinates in meters.
        lat0, lon0, alt0: Reference geodetic coordinates.

    Returns:
        lat, lon, alt: Geodetic coordinates.
    """
    x, y, z = enu_to_ecef(e, n, u, lat0, lon0, alt0)
    return ecef_to_geodetic(x, y, z)


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points on the earth (specified in decimal degrees).

    Args:
        lat1, lon1: First point coordinates.
        lat2, lon2: Second point coordinates.

    Returns:
        Distance in meters.
    """
    R = 6371000.0  # Earth radius in meters

    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c


def save_pickle(obj, path):
    """
    Save an object to a pickle file.

    Args:
        obj: Object to save.
        path: Destination file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    """
    Load an object from a pickle file.

    Args:
        path: Source file path.

    Returns:
        Loaded object.
    """
    with open(path, "rb") as f:
        return pickle.load(f)
