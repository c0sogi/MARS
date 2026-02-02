import numpy as np

# -------------------------------------------------------------------------
# WGS84 Ellipsoid Constants
# -------------------------------------------------------------------------
A = 6378137.0  # Semi-major axis (meters)
F = 1 / 298.257223563  # Flattening
B = A * (1 - F)  # Semi-minor axis (meters)
E2 = 2 * F - F**2  # First eccentricity squared


def wgs84_to_ecef(lat, lon, alt):
    """
    Convert WGS84 Geodetic coordinates to Earth-Centered Earth-Fixed (ECEF) Cartesian coordinates.

    Args:
        lat (float or np.array): Latitude in degrees.
        lon (float or np.array): Longitude in degrees.
        alt (float or np.array): Altitude in meters.

    Returns:
        tuple: (x, y, z) ECEF coordinates in meters.
    """
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)

    # Radius of curvature in the prime vertical
    N = A / np.sqrt(1 - E2 * sin_lat**2)

    x = (N + alt) * cos_lat * cos_lon
    y = (N + alt) * cos_lat * sin_lon
    z = (N * (1 - E2) + alt) * sin_lat

    return x, y, z


def ecef_to_wgs84(x, y, z):
    """
    Convert ECEF coordinates to WGS84 Geodetic coordinates.
    Uses a direct method suitable for Earth-based coordinates.

    Args:
        x (float or np.array): X coordinate in meters.
        y (float or np.array): Y coordinate in meters.
        z (float or np.array): Z coordinate in meters.

    Returns:
        tuple: (lat, lon, alt) in degrees and meters.
    """
    # Distance from Z-axis
    p = np.sqrt(x**2 + y**2)

    # Longitude
    lon = np.arctan2(y, x)

    # Latitude and Altitude (Iterative method or direct approximation)
    # Using Bowring's method or similar direct approximation is efficient
    theta = np.arctan2(z * A, p * B)

    e_prime_sq = E2 / (1 - E2)

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    lat = np.arctan2(z + e_prime_sq * B * sin_theta**3, p - E2 * A * cos_theta**3)

    sin_lat = np.sin(lat)
    N = A / np.sqrt(1 - E2 * sin_lat**2)

    alt = p / np.cos(lat) - N

    return np.degrees(lat), np.degrees(lon), alt


def ecef_to_enu(x, y, z, lat0, lon0, alt0):
    """
    Convert ECEF coordinates to a local East-North-Up (ENU) frame relative to a reference point.

    Args:
        x, y, z (float or np.array): Target ECEF coordinates.
        lat0, lon0, alt0 (float): Reference point Geodetic coordinates (degrees, meters).

    Returns:
        tuple: (e, n, u) coordinates in meters.
    """
    # Convert reference point to ECEF
    x0, y0, z0 = wgs84_to_ecef(lat0, lon0, alt0)

    # Delta ECEF vector
    dx = x - x0
    dy = y - y0
    dz = z - z0

    # Rotation Matrix parameters
    lat0_rad = np.radians(lat0)
    lon0_rad = np.radians(lon0)

    sin_lat = np.sin(lat0_rad)
    cos_lat = np.cos(lat0_rad)
    sin_lon = np.sin(lon0_rad)
    cos_lon = np.cos(lon0_rad)

    # Rotate vector from ECEF to ENU
    # Row 1: East
    e = -sin_lon * dx + cos_lon * dy
    # Row 2: North
    n = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
    # Row 3: Up
    u = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz

    return e, n, u


def enu_to_ecef(e, n, u, lat0, lon0, alt0):
    """
    Convert local ENU coordinates to ECEF relative to a reference point.

    Args:
        e, n, u (float or np.array): ENU coordinates in meters.
        lat0, lon0, alt0 (float): Reference point Geodetic coordinates.

    Returns:
        tuple: (x, y, z) ECEF coordinates.
    """
    # Convert reference point to ECEF
    x0, y0, z0 = wgs84_to_ecef(lat0, lon0, alt0)

    lat0_rad = np.radians(lat0)
    lon0_rad = np.radians(lon0)

    sin_lat = np.sin(lat0_rad)
    cos_lat = np.cos(lat0_rad)
    sin_lon = np.sin(lon0_rad)
    cos_lon = np.cos(lon0_rad)

    # Inverse Rotation (Transpose of the rotation matrix used in ecef_to_enu)
    # Col 1 of original is Row 1 here
    dx = -sin_lon * e - sin_lat * cos_lon * n + cos_lat * cos_lon * u
    dy = cos_lon * e - sin_lat * sin_lon * n + cos_lat * sin_lon * u
    dz = cos_lat * n + sin_lat * u

    x = x0 + dx
    y = y0 + dy
    z = z0 + dz

    return x, y, z


def geodetic_to_enu(lat, lon, alt, lat0, lon0, alt0):
    """
    Convenience function to convert Geodetic coordinates directly to local ENU.

    Args:
        lat, lon, alt (float or np.array): Target Geodetic coordinates.
        lat0, lon0, alt0 (float): Reference point Geodetic coordinates.

    Returns:
        tuple: (e, n, u) coordinates in meters.
    """
    x, y, z = wgs84_to_ecef(lat, lon, alt)
    return ecef_to_enu(x, y, z, lat0, lon0, alt0)


def enu_to_geodetic(e, n, u, lat0, lon0, alt0):
    """
    Convenience function to convert local ENU coordinates directly to Geodetic.

    Args:
        e, n, u (float or np.array): ENU coordinates in meters.
        lat0, lon0, alt0 (float): Reference point Geodetic coordinates.

    Returns:
        tuple: (lat, lon, alt) Geodetic coordinates.
    """
    x, y, z = enu_to_ecef(e, n, u, lat0, lon0, alt0)
    return ecef_to_wgs84(x, y, z)
