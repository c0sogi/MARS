import numpy as np
from library.config import Config


def llh_to_ecef(lat, lon, alt):
    """
    Convert Latitude, Longitude, Altitude to ECEF (X, Y, Z) coordinates.

    Args:
        lat: Latitude in degrees (float or numpy array)
        lon: Longitude in degrees (float or numpy array)
        alt: Altitude in meters (float or numpy array)

    Returns:
        x, y, z: ECEF coordinates in meters
    """
    a = Config.WGS84_A
    b = Config.WGS84_B
    e_sq = 1 - (b**2 / a**2)

    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)

    N = a / np.sqrt(1 - e_sq * sin_lat**2)

    x = (N + alt) * cos_lat * cos_lon
    y = (N + alt) * cos_lat * sin_lon
    z = (N * (1 - e_sq) + alt) * sin_lat

    return x, y, z


def ecef_to_llh(x, y, z):
    """
    Convert ECEF (X, Y, Z) to Latitude, Longitude, Altitude.
    Uses Ferrari's method for high precision conversion.

    Args:
        x, y, z: ECEF coordinates in meters (float or numpy array)

    Returns:
        lat, lon, alt: Latitude (deg), Longitude (deg), Altitude (m)
    """
    a = Config.WGS84_A
    b = Config.WGS84_B

    e_sq = 1 - (b**2 / a**2)
    ep_sq = (a**2 - b**2) / b**2

    p = np.sqrt(x**2 + y**2)
    th = np.arctan2(a * z, b * p)

    lon = np.arctan2(y, x)

    sin_th = np.sin(th)
    cos_th = np.cos(th)

    lat = np.arctan2(z + ep_sq * b * sin_th**3, p - e_sq * a * cos_th**3)

    sin_lat = np.sin(lat)
    N = a / np.sqrt(1 - e_sq * sin_lat**2)
    alt = p / np.cos(lat) - N

    return np.degrees(lat), np.degrees(lon), alt


def llh_to_enu(lat, lon, alt, ref_lat, ref_lon, ref_alt):
    """
    Convert LLH coordinates to ENU (East, North, Up) relative to a reference point.

    Args:
        lat, lon, alt: Target coordinates (deg, deg, m)
        ref_lat, ref_lon, ref_alt: Reference coordinates (deg, deg, m)

    Returns:
        e, n, u: ENU coordinates in meters
    """
    x, y, z = llh_to_ecef(lat, lon, alt)
    xr, yr, zr = llh_to_ecef(ref_lat, ref_lon, ref_alt)

    dx = x - xr
    dy = y - yr
    dz = z - zr

    ref_lat_rad = np.radians(ref_lat)
    ref_lon_rad = np.radians(ref_lon)

    sin_lat = np.sin(ref_lat_rad)
    cos_lat = np.cos(ref_lat_rad)
    sin_lon = np.sin(ref_lon_rad)
    cos_lon = np.cos(ref_lon_rad)

    e = -sin_lon * dx + cos_lon * dy
    n = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
    u = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz

    return e, n, u


def enu_to_llh(e, n, u, ref_lat, ref_lon, ref_alt):
    """
    Convert ENU coordinates back to LLH relative to a reference point.

    Args:
        e, n, u: ENU coordinates in meters
        ref_lat, ref_lon, ref_alt: Reference coordinates (deg, deg, m)

    Returns:
        lat, lon, alt: Target coordinates (deg, deg, m)
    """
    ref_lat_rad = np.radians(ref_lat)
    ref_lon_rad = np.radians(ref_lon)

    sin_lat = np.sin(ref_lat_rad)
    cos_lat = np.cos(ref_lat_rad)
    sin_lon = np.sin(ref_lon_rad)
    cos_lon = np.cos(ref_lon_rad)

    dx = -sin_lon * e - sin_lat * cos_lon * n + cos_lat * cos_lon * u
    dy = cos_lon * e - sin_lat * sin_lon * n + cos_lat * sin_lon * u
    dz = cos_lat * n + sin_lat * u

    xr, yr, zr = llh_to_ecef(ref_lat, ref_lon, ref_alt)

    x = xr + dx
    y = yr + dy
    z = zr + dz

    return ecef_to_llh(x, y, z)


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees).

    Args:
        lat1, lon1: First point coordinates
        lat2, lon2: Second point coordinates

    Returns:
        distance: Distance in meters
    """
    R = 6371000.0  # Earth radius in meters

    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance = R * c
    return distance
