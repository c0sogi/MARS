import numpy as np
from library.config import Config

# -------------------------------------------------------------------------
# Constants derived from Config
# -------------------------------------------------------------------------
_a = Config.WGS84_A
_f = Config.WGS84_F
_b = Config.WGS84_B
_e2 = 2 * _f - _f**2  # Square of first eccentricity


def WGS84_to_ECEF(lat, lon, alt):
    """
    Convert WGS84 geodetic coordinates to ECEF Cartesian coordinates.

    Args:
        lat (float or np.array): Latitude in degrees.
        lon (float or np.array): Longitude in degrees.
        alt (float or np.array): Altitude in meters.

    Returns:
        tuple: (x, y, z) in meters.
    """
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)

    # Prime vertical radius of curvature
    N = _a / np.sqrt(1 - _e2 * sin_lat**2)

    x = (N + alt) * cos_lat * cos_lon
    y = (N + alt) * cos_lat * sin_lon
    z = (N * (1 - _e2) + alt) * sin_lat

    return x, y, z


def ECEF_to_WGS84(x, y, z):
    """
    Convert ECEF Cartesian coordinates to WGS84 geodetic coordinates.
    Uses Heiskanen and Moritz's iterative method.

    Args:
        x (float or np.array): X coordinate in meters.
        y (float or np.array): Y coordinate in meters.
        z (float or np.array): Z coordinate in meters.

    Returns:
        tuple: (lat, lon, alt) in degrees and meters.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    lon_rad = np.arctan2(y, x)
    p = np.sqrt(x**2 + y**2)

    # Initial guess for latitude
    lat_rad = np.arctan2(z, p * (1 - _f))

    # Iteratively update latitude
    # 5 iterations is usually sufficient for mm precision
    for _ in range(5):
        sin_lat = np.sin(lat_rad)
        N = _a / np.sqrt(1 - _e2 * sin_lat**2)
        lat_rad = np.arctan2(z + _e2 * N * sin_lat, p)

    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    N = _a / np.sqrt(1 - _e2 * sin_lat**2)

    alt = p / cos_lat - N

    # Handle poles (cos_lat near 0)
    # If scalar
    if np.ndim(alt) == 0:
        if abs(lat_rad) > 1.5:  # Close to pi/2
            alt = z / sin_lat - N * (1 - _e2)
    else:
        # Vectorized pole handling
        pole_mask = np.abs(lat_rad) > 1.5
        alt[pole_mask] = z[pole_mask] / sin_lat[pole_mask] - N[pole_mask] * (1 - _e2)

    lat = np.degrees(lat_rad)
    lon = np.degrees(lon_rad)

    return lat, lon, alt


def ECEF_to_ENU(x, y, z, lat0, lon0, alt0):
    """
    Convert ECEF coordinates to local ENU coordinates relative to an anchor point.

    Args:
        x, y, z: Target ECEF coordinates (meters).
        lat0, lon0, alt0: Anchor WGS84 coordinates (degrees, meters).

    Returns:
        tuple: (e, n, u) in meters.
    """
    # Convert anchor to ECEF
    x0, y0, z0 = WGS84_to_ECEF(lat0, lon0, alt0)

    # Delta ECEF vector
    dx = x - x0
    dy = y - y0
    dz = z - z0

    # Rotation matrix elements
    lat0_rad = np.radians(lat0)
    lon0_rad = np.radians(lon0)

    sin_lat = np.sin(lat0_rad)
    cos_lat = np.cos(lat0_rad)
    sin_lon = np.sin(lon0_rad)
    cos_lon = np.cos(lon0_rad)

    # Rotate
    # E = -sin(lon)*dx + cos(lon)*dy
    # N = -sin(lat)cos(lon)*dx - sin(lat)sin(lon)*dy + cos(lat)*dz
    # U = cos(lat)cos(lon)*dx + cos(lat)sin(lon)*dy + sin(lat)*dz

    e = -sin_lon * dx + cos_lon * dy
    n = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
    u = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz

    return e, n, u


def ENU_to_ECEF(e, n, u, lat0, lon0, alt0):
    """
    Convert local ENU coordinates to ECEF coordinates relative to an anchor point.

    Args:
        e, n, u: Local ENU coordinates (meters).
        lat0, lon0, alt0: Anchor WGS84 coordinates (degrees, meters).

    Returns:
        tuple: (x, y, z) in meters.
    """
    # Convert anchor to ECEF
    x0, y0, z0 = WGS84_to_ECEF(lat0, lon0, alt0)

    # Rotation matrix elements
    lat0_rad = np.radians(lat0)
    lon0_rad = np.radians(lon0)

    sin_lat = np.sin(lat0_rad)
    cos_lat = np.cos(lat0_rad)
    sin_lon = np.sin(lon0_rad)
    cos_lon = np.cos(lon0_rad)

    # Inverse Rotation (Transpose)
    # dx = -sin(lon)*E - sin(lat)cos(lon)*N + cos(lat)cos(lon)*U
    # dy = cos(lon)*E - sin(lat)sin(lon)*N + cos(lat)sin(lon)*U
    # dz = cos(lat)*N + sin(lat)*U

    dx = -sin_lon * e - sin_lat * cos_lon * n + cos_lat * cos_lon * u
    dy = cos_lon * e - sin_lat * sin_lon * n + cos_lat * sin_lon * u
    dz = cos_lat * n + sin_lat * u

    x = x0 + dx
    y = y0 + dy
    z = z0 + dz

    return x, y, z


def ENU_to_WGS84(e, n, u, lat0, lon0, alt0):
    """
    Convert local ENU coordinates to WGS84 geodetic coordinates.
    Useful for converting predicted residuals back to global submission format.

    Args:
        e, n, u: Local ENU coordinates (meters).
        lat0, lon0, alt0: Anchor WGS84 coordinates (degrees, meters).

    Returns:
        tuple: (lat, lon, alt) in degrees and meters.
    """
    x, y, z = ENU_to_ECEF(e, n, u, lat0, lon0, alt0)
    return ECEF_to_WGS84(x, y, z)


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees).

    Args:
        lat1, lon1: First point coordinates.
        lat2, lon2: Second point coordinates.

    Returns:
        Distance in meters.
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2

    # Clip to handle potential floating point errors slightly outside [0, 1]
    a = np.clip(a, 0.0, 1.0)

    c = 2 * np.arcsin(np.sqrt(a))

    # Radius of earth in kilometers is 6371
    # Return distance in meters
    r = 6371000.0
    return c * r
