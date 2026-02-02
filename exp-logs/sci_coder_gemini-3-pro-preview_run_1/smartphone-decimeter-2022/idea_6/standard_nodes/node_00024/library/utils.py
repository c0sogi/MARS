import numpy as np

# =============================================================================
# WGS84 ELLIPSOID CONSTANTS
# =============================================================================
A = 6378137.0  # Semi-major axis (meters)
B = 6356752.314245  # Semi-minor axis (meters)
F = 1 / 298.257223563  # Flattening
E2 = F * (2 - F)  # Eccentricity squared


# =============================================================================
# METRIC FUNCTIONS
# =============================================================================
def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees).

    Args:
        lat1, lon1: First point coordinates (scalar or array)
        lat2, lon2: Second point coordinates (scalar or array)

    Returns:
        Distance in meters.
    """
    # Convert decimal degrees to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    )

    # Clip to valid range [0, 1] to handle floating point errors
    a = np.clip(a, 0, 1)

    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371000.0  # Average Earth radius in meters

    return c * r


# =============================================================================
# COORDINATE TRANSFORMATION FUNCTIONS
# =============================================================================
def geodetic_to_ecef(lat, lon, alt):
    """
    Convert geodetic coordinates (WGS84) to Earth-Centered, Earth-Fixed (ECEF).

    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees
        alt: Altitude in meters

    Returns:
        x, y, z: ECEF coordinates in meters
    """
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    # Radius of curvature in the prime vertical
    N = A / np.sqrt(1 - E2 * np.sin(lat_rad) ** 2)

    x = (N + alt) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (N + alt) * np.cos(lat_rad) * np.sin(lon_rad)
    z = (N * (1 - E2) + alt) * np.sin(lat_rad)

    return x, y, z


def ecef_to_enu(x, y, z, lat_ref, lon_ref, alt_ref):
    """
    Convert ECEF coordinates to East-North-Up (ENU) relative to a reference point.

    Args:
        x, y, z: Target ECEF coordinates
        lat_ref, lon_ref, alt_ref: Reference geodetic coordinates

    Returns:
        e, n, u: ENU coordinates in meters
    """
    # Convert reference point to ECEF
    xr, yr, zr = geodetic_to_ecef(lat_ref, lon_ref, alt_ref)

    dx = x - xr
    dy = y - yr
    dz = z - zr

    lat_rad = np.radians(lat_ref)
    lon_rad = np.radians(lon_ref)

    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)

    # Rotation matrix application
    e = -sin_lon * dx + cos_lon * dy
    n = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
    u = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz

    return e, n, u


def geodetic_to_enu(lat, lon, alt, lat_ref, lon_ref, alt_ref):
    """
    Convert Geodetic coordinates to ENU directly.

    Args:
        lat, lon, alt: Target geodetic coordinates
        lat_ref, lon_ref, alt_ref: Reference geodetic coordinates

    Returns:
        e, n, u: ENU coordinates in meters
    """
    x, y, z = geodetic_to_ecef(lat, lon, alt)
    return ecef_to_enu(x, y, z, lat_ref, lon_ref, alt_ref)


def enu_to_ecef(e, n, u, lat_ref, lon_ref, alt_ref):
    """
    Convert ENU coordinates back to ECEF.

    Args:
        e, n, u: ENU coordinates
        lat_ref, lon_ref, alt_ref: Reference geodetic coordinates

    Returns:
        x, y, z: ECEF coordinates
    """
    lat_rad = np.radians(lat_ref)
    lon_rad = np.radians(lon_ref)

    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)

    # Reference ECEF
    xr, yr, zr = geodetic_to_ecef(lat_ref, lon_ref, alt_ref)

    # Inverse rotation
    dx = -sin_lon * e - sin_lat * cos_lon * n + cos_lat * cos_lon * u
    dy = cos_lon * e - sin_lat * sin_lon * n + cos_lat * sin_lon * u
    dz = cos_lat * n + sin_lat * u

    return xr + dx, yr + dy, zr + dz


def ecef_to_geodetic(x, y, z):
    """
    Convert ECEF coordinates to Geodetic (lat, lon, alt) using an iterative method.

    Args:
        x, y, z: ECEF coordinates

    Returns:
        lat, lon, alt: Geodetic coordinates (degrees, degrees, meters)
    """
    # Ensure inputs are numpy arrays for vectorized operations
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    p = np.sqrt(x**2 + y**2)

    # Longitude is direct
    lon = np.arctan2(y, x)

    # Initial guess for latitude
    # Assume h=0 for initialization
    lat = np.arctan2(z, p * (1 - E2))

    # Iterative solution for latitude and altitude
    # 5 iterations is typically sufficient for sub-millimeter precision
    h = np.zeros_like(lat)

    for _ in range(5):
        sin_lat = np.sin(lat)
        N = A / np.sqrt(1 - E2 * sin_lat**2)
        h = p / np.cos(lat) - N
        lat = np.arctan2(z, p * (1 - E2 * N / (N + h)))

    return np.degrees(lat), np.degrees(lon), h


def enu_to_geodetic(e, n, u, lat_ref, lon_ref, alt_ref):
    """
    Convert ENU coordinates back to Geodetic.

    Args:
        e, n, u: ENU coordinates
        lat_ref, lon_ref, alt_ref: Reference geodetic coordinates

    Returns:
        lat, lon, alt: Geodetic coordinates
    """
    x, y, z = enu_to_ecef(e, n, u, lat_ref, lon_ref, alt_ref)
    return ecef_to_geodetic(x, y, z)
