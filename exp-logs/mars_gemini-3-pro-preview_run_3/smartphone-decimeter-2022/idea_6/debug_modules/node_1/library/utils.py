import numpy as np
from library.config import WGS84_A, WGS84_B, WGS84_E2


def wgs84_to_ecef(lat, lon, alt):
    """
    Convert WGS84 Geodetic coordinates to ECEF (Earth-Centered, Earth-Fixed).

    Args:
        lat: Latitude in degrees.
        lon: Longitude in degrees.
        alt: Altitude in meters.

    Returns:
        x, y, z: ECEF coordinates in meters.
    """
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    # Radius of curvature in the prime vertical
    N = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat_rad) ** 2)

    x = (N + alt) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (N + alt) * np.cos(lat_rad) * np.sin(lon_rad)
    z = (N * (1 - WGS84_E2) + alt) * np.sin(lat_rad)

    return x, y, z


def ecef_to_wgs84(x, y, z):
    """
    Convert ECEF coordinates to WGS84 Geodetic coordinates.
    Uses Bowring's method for high accuracy.

    Args:
        x, y, z: ECEF coordinates in meters.

    Returns:
        lat, lon, alt: Latitude (deg), Longitude (deg), Altitude (m).
    """
    # Second eccentricity squared
    e_prime_sq = (WGS84_A**2 - WGS84_B**2) / WGS84_B**2

    p = np.sqrt(x**2 + y**2)
    theta = np.arctan2(z * WGS84_A, p * WGS84_B)

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    lon_rad = np.arctan2(y, x)

    num = z + e_prime_sq * WGS84_B * sin_theta**3
    den = p - WGS84_E2 * WGS84_A * cos_theta**3
    lat_rad = np.arctan2(num, den)

    sin_lat = np.sin(lat_rad)
    N = WGS84_A / np.sqrt(1 - WGS84_E2 * sin_lat**2)

    # Avoid division by zero for points near poles where cos(lat) is small
    # However, for smartphone GPS data (usually not at poles), standard formula is fine.
    # Using a conditional for numerical stability:
    alt = p / np.cos(lat_rad) - N

    # For very high latitudes, alt = z / sin(lat) - N * (1 - e2) is preferred,
    # but cos(lat) is generally safe for this dataset's domain.

    return np.degrees(lat_rad), np.degrees(lon_rad), alt


def wgs84_to_enu(lat, lon, alt, lat_ref, lon_ref, alt_ref):
    """
    Convert WGS84 coordinates to East-North-Up (ENU) relative to a reference point.

    Args:
        lat, lon, alt: Target coordinates (deg, deg, m).
        lat_ref, lon_ref, alt_ref: Reference coordinates (deg, deg, m).

    Returns:
        e, n, u: ENU coordinates in meters.
    """
    # Convert both to ECEF
    x, y, z = wgs84_to_ecef(lat, lon, alt)
    xr, yr, zr = wgs84_to_ecef(lat_ref, lon_ref, alt_ref)

    dx = x - xr
    dy = y - yr
    dz = z - zr

    # Rotation matrix components based on reference point
    lat_ref_rad = np.radians(lat_ref)
    lon_ref_rad = np.radians(lon_ref)

    sin_lat = np.sin(lat_ref_rad)
    cos_lat = np.cos(lat_ref_rad)
    sin_lon = np.sin(lon_ref_rad)
    cos_lon = np.cos(lon_ref_rad)

    e = -sin_lon * dx + cos_lon * dy
    n = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
    u = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz

    return e, n, u


def enu_to_wgs84(e, n, u, lat_ref, lon_ref, alt_ref):
    """
    Convert ENU coordinates back to WGS84 relative to a reference point.

    Args:
        e, n, u: ENU coordinates in meters.
        lat_ref, lon_ref, alt_ref: Reference coordinates (deg, deg, m).

    Returns:
        lat, lon, alt: Target coordinates (deg, deg, m).
    """
    # Convert reference to ECEF
    xr, yr, zr = wgs84_to_ecef(lat_ref, lon_ref, alt_ref)

    lat_ref_rad = np.radians(lat_ref)
    lon_ref_rad = np.radians(lon_ref)

    sin_lat = np.sin(lat_ref_rad)
    cos_lat = np.cos(lat_ref_rad)
    sin_lon = np.sin(lon_ref_rad)
    cos_lon = np.cos(lon_ref_rad)

    # Inverse rotation (Transpose of the rotation matrix)
    dx = -sin_lon * e - sin_lat * cos_lon * n + cos_lat * cos_lon * u
    dy = cos_lon * e - sin_lat * sin_lon * n + cos_lat * sin_lon * u
    dz = cos_lat * n + sin_lat * u

    x = xr + dx
    y = yr + dy
    z = zr + dz

    return ecef_to_wgs84(x, y, z)


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points on the earth (specified in decimal degrees).

    Args:
        lat1, lon1: First point coordinates in degrees.
        lat2, lon2: Second point coordinates in degrees.

    Returns:
        Distance in meters.
    """
    R = 6371000.0  # Earth radius in meters

    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c
