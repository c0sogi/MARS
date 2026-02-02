import numpy as np
from library.config import Config


def lla_to_ecef(lat_deg, lon_deg, alt_m):
    """
    Convert Geodetic coordinates (Latitude, Longitude, Altitude) to ECEF (X, Y, Z).
    Vectorized implementation.

    Args:
        lat_deg: Latitude in degrees (numpy array or scalar).
        lon_deg: Longitude in degrees (numpy array or scalar).
        alt_m: Altitude in meters (numpy array or scalar).

    Returns:
        x, y, z: ECEF coordinates in meters.
    """
    lat_rad = np.radians(lat_deg)
    lon_rad = np.radians(lon_deg)

    a = Config.WGS84_A
    e2 = Config.WGS84_E2

    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)

    N = a / np.sqrt(1 - e2 * sin_lat**2)

    x = (N + alt_m) * cos_lat * cos_lon
    y = (N + alt_m) * cos_lat * sin_lon
    z = (N * (1 - e2) + alt_m) * sin_lat

    return x, y, z


def ecef_to_lla(x, y, z):
    """
    Convert ECEF coordinates (X, Y, Z) to Geodetic (Latitude, Longitude, Altitude).
    Uses Ferrari's solution. Vectorized implementation.

    Args:
        x, y, z: ECEF coordinates in meters (numpy arrays or scalars).

    Returns:
        lat, lon, alt: Latitude (deg), Longitude (deg), Altitude (m).
    """
    a = Config.WGS84_A
    e2 = Config.WGS84_E2

    # Derived constants
    e = np.sqrt(e2)
    b = np.sqrt(a**2 * (1 - e2))
    ep2 = (a**2 - b**2) / b**2

    p = np.sqrt(x**2 + y**2)
    th = np.arctan2(a * z, b * p)

    sin_th = np.sin(th)
    cos_th = np.cos(th)

    lon_rad = np.arctan2(y, x)
    lat_rad = np.arctan2(z + ep2 * b * sin_th**3, p - e2 * a * cos_th**3)

    sin_lat = np.sin(lat_rad)
    N = a / np.sqrt(1 - e2 * sin_lat**2)
    alt = p / np.cos(lat_rad) - N

    # Handle poles (cos(lat) ~ 0)
    # For simplicity in this context, we assume non-polar regions or handle numerical stability by numpy
    # A robust check could be added, but standard datasets are usually mid-latitude.

    lat_deg = np.degrees(lat_rad)
    lon_deg = np.degrees(lon_rad)

    return lat_deg, lon_deg, alt


def ecef_to_enu(x, y, z, ref_lat, ref_lon, ref_alt):
    """
    Convert ECEF coordinates to Local East-North-Up (ENU) relative to a reference point.

    Args:
        x, y, z: Target ECEF coordinates.
        ref_lat, ref_lon, ref_alt: Reference point Geodetic coordinates.

    Returns:
        e, n, u: East, North, Up coordinates in meters.
    """
    # Convert reference point to ECEF
    ref_x, ref_y, ref_z = lla_to_ecef(ref_lat, ref_lon, ref_alt)

    dx = x - ref_x
    dy = y - ref_y
    dz = z - ref_z

    ref_lat_rad = np.radians(ref_lat)
    ref_lon_rad = np.radians(ref_lon)

    sin_lat = np.sin(ref_lat_rad)
    cos_lat = np.cos(ref_lat_rad)
    sin_lon = np.sin(ref_lon_rad)
    cos_lon = np.cos(ref_lon_rad)

    # Rotation matrix multiplication
    e = -sin_lon * dx + cos_lon * dy
    n = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
    u = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz

    return e, n, u


def enu_to_ecef(e, n, u, ref_lat, ref_lon, ref_alt):
    """
    Convert Local ENU coordinates back to ECEF.

    Args:
        e, n, u: Local ENU coordinates in meters.
        ref_lat, ref_lon, ref_alt: Reference point Geodetic coordinates.

    Returns:
        x, y, z: ECEF coordinates.
    """
    ref_x, ref_y, ref_z = lla_to_ecef(ref_lat, ref_lon, ref_alt)

    ref_lat_rad = np.radians(ref_lat)
    ref_lon_rad = np.radians(ref_lon)

    sin_lat = np.sin(ref_lat_rad)
    cos_lat = np.cos(ref_lat_rad)
    sin_lon = np.sin(ref_lon_rad)
    cos_lon = np.cos(ref_lon_rad)

    # Inverse rotation
    dx = -sin_lon * e - sin_lat * cos_lon * n + cos_lat * cos_lon * u
    dy = cos_lon * e - sin_lat * sin_lon * n + cos_lat * sin_lon * u
    dz = cos_lat * n + sin_lat * u

    x = ref_x + dx
    y = ref_y + dy
    z = ref_z + dz

    return x, y, z


def lla_to_enu_relative(lat, lon, alt, ref_lat, ref_lon, ref_alt):
    """
    Convert Geodetic coordinates to Local ENU relative to a reference point.
    This is the primary transformation for input features.

    Args:
        lat, lon, alt: Target Geodetic coordinates.
        ref_lat, ref_lon, ref_alt: Reference Geodetic coordinates.

    Returns:
        e, n, u: East, North, Up coordinates in meters.
    """
    x, y, z = lla_to_ecef(lat, lon, alt)
    return ecef_to_enu(x, y, z, ref_lat, ref_lon, ref_alt)


def enu_to_lla_relative(e, n, u, ref_lat, ref_lon, ref_alt):
    """
    Convert Local ENU coordinates back to Geodetic relative to a reference point.
    This is used to reconstruct the final Lat/Lon predictions.

    Args:
        e, n, u: Local ENU coordinates in meters.
        ref_lat, ref_lon, ref_alt: Reference Geodetic coordinates.

    Returns:
        lat, lon, alt: Geodetic coordinates.
    """
    x, y, z = enu_to_ecef(e, n, u, ref_lat, ref_lon, ref_alt)
    return ecef_to_lla(x, y, z)


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the Great Circle distance between two points on the Earth.

    Args:
        lat1, lon1: First point coordinates in degrees.
        lat2, lon2: Second point coordinates in degrees.

    Returns:
        distance: Distance in meters.
    """
    R = 6371000.0  # Earth radius in meters

    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c


def flat_earth_projection(lat, lon, ref_lat, ref_lon):
    """
    Converts lat/lon to local meters (North, East) relative to a reference point
    using a flat earth approximation.
    Cite solution_lesson_node_00020: Minimize Coordinate Transformations.
    """
    R = 6371000.0
    dlat = np.radians(lat - ref_lat)
    dlon = np.radians(lon - ref_lon)

    # North offset in meters
    north_m = dlat * R

    # East offset in meters (adjusted for latitude)
    east_m = dlon * R * np.cos(np.radians(ref_lat))

    return north_m, east_m


def inverse_flat_earth_projection(north_m, east_m, ref_lat, ref_lon):
    """
    Converts local meters (North, East) back to lat/lon.
    """
    R = 6371000.0

    dlat = np.degrees(north_m / R)
    dlon = np.degrees(east_m / (R * np.cos(np.radians(ref_lat))))

    lat = ref_lat + dlat
    lon = ref_lon + dlon

    return lat, lon
