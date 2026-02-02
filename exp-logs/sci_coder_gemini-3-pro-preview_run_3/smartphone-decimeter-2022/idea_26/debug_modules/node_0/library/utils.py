import numpy as np
from library.config import WGS84_A, WGS84_F, WGS84_B, WGS84_E2


def geodetic_to_ecef(lat, lon, alt):
    """
    Convert geodetic coordinates (latitude, longitude, altitude) to ECEF coordinates.

    Parameters:
    -----------
    lat : float or np.array
        Latitude in degrees.
    lon : float or np.array
        Longitude in degrees.
    alt : float or np.array
        Altitude in meters.

    Returns:
    --------
    x, y, z : float or np.array
        ECEF coordinates in meters.
    """
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)

    N = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat_rad) ** 2)

    x = (N + alt) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (N + alt) * np.cos(lat_rad) * np.sin(lon_rad)
    z = (N * (1 - WGS84_E2) + alt) * np.sin(lat_rad)

    return x, y, z


def ecef_to_geodetic(x, y, z):
    """
    Convert ECEF coordinates to geodetic coordinates (lat, lon, alt).
    Uses Heikkinen's method for stability.

    Parameters:
    -----------
    x, y, z : float or np.array
        ECEF coordinates in meters.

    Returns:
    --------
    lat, lon, alt : float or np.array
        Latitude (deg), Longitude (deg), Altitude (m).
    """
    a = WGS84_A
    b = WGS84_B
    e2 = WGS84_E2
    ep2 = (a**2 - b**2) / b**2

    p = np.sqrt(x**2 + y**2)
    theta = np.arctan2(z * a, p * b)

    lon = np.arctan2(y, x)

    lat = np.arctan2(z + ep2 * b * np.sin(theta) ** 3, p - e2 * a * np.cos(theta) ** 3)

    N = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    alt = p / np.cos(lat) - N

    return np.rad2deg(lat), np.rad2deg(lon), alt


def rotation_matrix(lat, lon):
    """
    Construct the rotation matrix from ECEF to ENU frame at a specific geodetic location.

    Parameters:
    -----------
    lat : float
        Reference latitude in degrees.
    lon : float
        Reference longitude in degrees.

    Returns:
    --------
    R : np.array (3, 3)
        Rotation matrix such that d_enu = R @ d_ecef.
    """
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)

    slat = np.sin(lat_rad)
    clat = np.cos(lat_rad)
    slon = np.sin(lon_rad)
    clon = np.cos(lon_rad)

    # Row 0: East unit vector
    # Row 1: North unit vector
    # Row 2: Up unit vector
    R = np.array(
        [
            [-slon, cos_lon, 0],
            [-slat * cos_lon, -slat * sin_lon, clat],
            [clat * cos_lon, clat * sin_lon, slat],
        ]
    )

    return R


def ecef_to_enu(x, y, z, lat0, lon0, alt0):
    """
    Convert ECEF coordinates to local ENU coordinates relative to a reference point.
    Supports both scalar and array inputs for coordinates and references.

    Parameters:
    -----------
    x, y, z : float or np.array
        Target ECEF coordinates.
    lat0, lon0, alt0 : float or np.array
        Reference geodetic coordinates.

    Returns:
    --------
    e, n, u : float or np.array
        East, North, Up coordinates in meters.
    """
    # Convert reference to ECEF
    x0, y0, z0 = geodetic_to_ecef(lat0, lon0, alt0)

    dx = x - x0
    dy = y - y0
    dz = z - z0

    lat0_rad = np.deg2rad(lat0)
    lon0_rad = np.deg2rad(lon0)

    slat = np.sin(lat0_rad)
    clat = np.cos(lat0_rad)
    slon = np.sin(lon0_rad)
    clon = np.cos(lon0_rad)

    # Vectorized rotation application
    # East: -sin(lon)*dx + cos(lon)*dy
    e = -slon * dx + clon * dy

    # North: -sin(lat)cos(lon)*dx - sin(lat)sin(lon)*dy + cos(lat)*dz
    n = -slat * clon * dx - slat * slon * dy + clat * dz

    # Up: cos(lat)cos(lon)*dx + cos(lat)sin(lon)*dy + sin(lat)*dz
    u = clat * clon * dx + clat * slon * dy + slat * dz

    return e, n, u


def enu_to_ecef(e, n, u, lat0, lon0, alt0):
    """
    Convert local ENU coordinates back to ECEF.

    Parameters:
    -----------
    e, n, u : float or np.array
        ENU coordinates in meters.
    lat0, lon0, alt0 : float or np.array
        Reference geodetic coordinates.

    Returns:
    --------
    x, y, z : float or np.array
        ECEF coordinates.
    """
    x0, y0, z0 = geodetic_to_ecef(lat0, lon0, alt0)

    lat0_rad = np.deg2rad(lat0)
    lon0_rad = np.deg2rad(lon0)

    slat = np.sin(lat0_rad)
    clat = np.cos(lat0_rad)
    slon = np.sin(lon0_rad)
    clon = np.cos(lon0_rad)

    # Inverse rotation (Transpose of R)
    # dx = -slon*e - slat*clon*n + clat*clon*u
    dx = -slon * e - slat * clon * n + clat * clon * u

    # dy = clon*e - slat*slon*n + clat*slon*u
    dy = clon * e - slat * slon * n + clat * slon * u

    # dz = 0*e + clat*n + slat*u
    dz = clat * n + slat * u

    x = x0 + dx
    y = y0 + dy
    z = z0 + dz

    return x, y, z
