import numpy as np
from library.config import WGS84_A, WGS84_B, WGS84_E2


def lla2ecef(lat_deg, lon_deg, alt_m):
    """
    Convert Latitude, Longitude, Altitude to ECEF coordinates.

    Args:
        lat_deg: Latitude in degrees (float or numpy array)
        lon_deg: Longitude in degrees (float or numpy array)
        alt_m: Altitude in meters (float or numpy array)

    Returns:
        x, y, z: ECEF coordinates in meters
    """
    lat_rad = np.deg2rad(lat_deg)
    lon_rad = np.deg2rad(lon_deg)

    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)

    # Radius of curvature in the prime vertical
    N = WGS84_A / np.sqrt(1 - WGS84_E2 * sin_lat**2)

    x = (N + alt_m) * cos_lat * cos_lon
    y = (N + alt_m) * cos_lat * sin_lon
    z = (N * (1 - WGS84_E2) + alt_m) * sin_lat

    return x, y, z


def ecef2lla(x, y, z):
    """
    Convert ECEF coordinates to Latitude, Longitude, Altitude.
    Uses Bowring's method for high precision.

    Args:
        x, y, z: ECEF coordinates in meters (float or numpy array)

    Returns:
        lat_deg, lon_deg, alt_m
    """
    # Second eccentricity squared
    e2_prime = (WGS84_A**2 - WGS84_B**2) / WGS84_B**2

    p = np.sqrt(x**2 + y**2)
    theta = np.arctan2(z * WGS84_A, p * WGS84_B)

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    lat_rad = np.arctan2(
        z + e2_prime * WGS84_B * sin_theta**3, p - WGS84_E2 * WGS84_A * cos_theta**3
    )
    lon_rad = np.arctan2(y, x)

    sin_lat = np.sin(lat_rad)
    N = WGS84_A / np.sqrt(1 - WGS84_E2 * sin_lat**2)

    alt_m = p / np.cos(lat_rad) - N

    lat_deg = np.rad2deg(lat_rad)
    lon_deg = np.rad2deg(lon_rad)

    return lat_deg, lon_deg, alt_m


def ecef2enu(x, y, z, lat0_deg, lon0_deg, alt0_m):
    """
    Convert ECEF coordinates to local ENU coordinates relative to a reference point.

    Args:
        x, y, z: Target ECEF coordinates (float or numpy array)
        lat0_deg, lon0_deg, alt0_m: Reference point LLA coordinates

    Returns:
        e, n, u: East, North, Up coordinates in meters
    """
    # Convert reference point to ECEF
    x0, y0, z0 = lla2ecef(lat0_deg, lon0_deg, alt0_m)

    # Difference vector
    dx = x - x0
    dy = y - y0
    dz = z - z0

    # Rotation matrix components
    lat0_rad = np.deg2rad(lat0_deg)
    lon0_rad = np.deg2rad(lon0_deg)

    sin_lat = np.sin(lat0_rad)
    cos_lat = np.cos(lat0_rad)
    sin_lon = np.sin(lon0_rad)
    cos_lon = np.cos(lon0_rad)

    # Rotate
    e = -sin_lon * dx + cos_lon * dy
    n = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
    u = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz

    return e, n, u


def enu2ecef(e, n, u, lat0_deg, lon0_deg, alt0_m):
    """
    Convert local ENU coordinates to ECEF coordinates relative to a reference point.

    Args:
        e, n, u: ENU coordinates in meters (float or numpy array)
        lat0_deg, lon0_deg, alt0_m: Reference point LLA coordinates

    Returns:
        x, y, z: ECEF coordinates in meters
    """
    # Convert reference point to ECEF
    x0, y0, z0 = lla2ecef(lat0_deg, lon0_deg, alt0_m)

    # Rotation matrix components
    lat0_rad = np.deg2rad(lat0_deg)
    lon0_rad = np.deg2rad(lon0_deg)

    sin_lat = np.sin(lat0_rad)
    cos_lat = np.cos(lat0_rad)
    sin_lon = np.sin(lon0_rad)
    cos_lon = np.cos(lon0_rad)

    # Inverse rotation (transpose of the rotation matrix used in ecef2enu)
    dx = -sin_lon * e - sin_lat * cos_lon * n + cos_lat * cos_lon * u
    dy = cos_lon * e - sin_lat * sin_lon * n + cos_lat * sin_lon * u
    dz = cos_lat * n + sin_lat * u

    x = x0 + dx
    y = y0 + dy
    z = z0 + dz

    return x, y, z


def calculate_los_vector(user_ecef, sat_ecef):
    """
    Calculate the Line-of-Sight (LOS) unit vector from user to satellite.

    Args:
        user_ecef: Tuple or array of (x, y, z) for user position
        sat_ecef: Tuple or array of (x, y, z) for satellite position

    Returns:
        los_vector: Normalized unit vector (x, y, z) pointing to satellite
    """
    # Ensure inputs are numpy arrays
    u_pos = np.array(user_ecef)
    s_pos = np.array(sat_ecef)

    # Vector from user to satellite
    diff = s_pos - u_pos

    # Compute distance (norm)
    dist = np.linalg.norm(diff, axis=-1, keepdims=True)

    # Handle division by zero if positions are identical (unlikely in practice)
    # Using np.divide with where clause or just standard division if arrays are safe
    # Adding epsilon to avoid nan is a safe practice, though dist should be large (~20000km)

    los_vector = diff / dist

    return los_vector
