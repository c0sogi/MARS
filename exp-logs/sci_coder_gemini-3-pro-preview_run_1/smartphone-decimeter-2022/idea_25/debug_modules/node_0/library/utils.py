import numpy as np

# -------------------------------------------------------------------------
# WGS84 Ellipsoid Constants
# -------------------------------------------------------------------------
WGS84_A = 6378137.0  # Semi-major axis
WGS84_F = 1.0 / 298.257223563  # Flattening
WGS84_B = WGS84_A * (1.0 - WGS84_F)  # Semi-minor axis
WGS84_E2 = 2 * WGS84_F - WGS84_F**2  # Eccentricity squared


def lla_to_ecef(lat_deg, lon_deg, alt):
    """
    Convert Latitude, Longitude, Altitude to ECEF coordinates.

    Args:
        lat_deg: Latitude in degrees (numpy array or scalar)
        lon_deg: Longitude in degrees (numpy array or scalar)
        alt: Altitude in meters (numpy array or scalar)

    Returns:
        x, y, z: ECEF coordinates in meters
    """
    lat_rad = np.radians(lat_deg)
    lon_rad = np.radians(lon_deg)

    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)

    N = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat**2)

    x = (N + alt) * cos_lat * cos_lon
    y = (N + alt) * cos_lat * sin_lon
    z = (N * (1.0 - WGS84_E2) + alt) * sin_lat

    return x, y, z


def ecef_to_lla(x, y, z):
    """
    Convert ECEF coordinates to Latitude, Longitude, Altitude using iterative method.

    Args:
        x, y, z: ECEF coordinates in meters (numpy array or scalar)

    Returns:
        lat, lon, alt: Latitude (deg), Longitude (deg), Altitude (m)
    """
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    lon_rad = np.arctan2(y, x)
    p = np.sqrt(x**2 + y**2)

    # Initial approximation
    lat_rad = np.arctan2(z, p * (1 - WGS84_E2))
    h = 0.0

    # Iteratively update latitude and altitude
    # 5 iterations is usually sufficient for cm-level precision
    for _ in range(5):
        sin_lat = np.sin(lat_rad)
        N = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat**2)
        h = p / np.cos(lat_rad) - N
        lat_rad = np.arctan2(z, p * (1.0 - WGS84_E2 * N / (N + h)))

    lat_deg = np.degrees(lat_rad)
    lon_deg = np.degrees(lon_rad)

    return lat_deg, lon_deg, h


def ecef_to_enu(x, y, z, ref_lat, ref_lon, ref_alt):
    """
    Convert ECEF coordinates to local ENU coordinates relative to a reference point.

    Args:
        x, y, z: Target ECEF coordinates
        ref_lat, ref_lon, ref_alt: Reference point LLA

    Returns:
        e, n, u: East, North, Up coordinates in meters
    """
    # Convert reference point to ECEF
    ref_x, ref_y, ref_z = lla_to_ecef(ref_lat, ref_lon, ref_alt)

    # Delta ECEF
    dx = x - ref_x
    dy = y - ref_y
    dz = z - ref_z

    # Rotation Matrix elements
    lat_rad = np.radians(ref_lat)
    lon_rad = np.radians(ref_lon)

    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)

    # Rotate
    e = -sin_lon * dx + cos_lon * dy
    n = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
    u = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz

    return e, n, u


def enu_to_ecef(e, n, u, ref_lat, ref_lon, ref_alt):
    """
    Convert local ENU coordinates to ECEF coordinates relative to a reference point.

    Args:
        e, n, u: ENU coordinates in meters
        ref_lat, ref_lon, ref_alt: Reference point LLA

    Returns:
        x, y, z: ECEF coordinates
    """
    # Convert reference point to ECEF
    ref_x, ref_y, ref_z = lla_to_ecef(ref_lat, ref_lon, ref_alt)

    # Rotation Matrix elements
    lat_rad = np.radians(ref_lat)
    lon_rad = np.radians(ref_lon)

    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)

    # Inverse Rotation (Transpose)
    # R_inv = [ -sin_lon, -sin_lat*cos_lon, cos_lat*cos_lon ]
    #         [  cos_lon, -sin_lat*sin_lon, cos_lat*sin_lon ]
    #         [        0,          cos_lat,         sin_lat ]

    dx = -sin_lon * e - sin_lat * cos_lon * n + cos_lat * cos_lon * u
    dy = cos_lon * e - sin_lat * sin_lon * n + cos_lat * sin_lon * u
    dz = cos_lat * n + sin_lat * u

    x = ref_x + dx
    y = ref_y + dy
    z = ref_z + dz

    return x, y, z


def wgs84_to_enu(lat, lon, alt, ref_lat, ref_lon, ref_alt):
    """
    Wrapper to convert WGS84 LLA to ENU relative to a reference.
    """
    x, y, z = lla_to_ecef(lat, lon, alt)
    return ecef_to_enu(x, y, z, ref_lat, ref_lon, ref_alt)


def enu_to_wgs84(e, n, u, ref_lat, ref_lon, ref_alt):
    """
    Wrapper to convert ENU to WGS84 LLA relative to a reference.
    """
    x, y, z = enu_to_ecef(e, n, u, ref_lat, ref_lon, ref_alt)
    return ecef_to_lla(x, y, z)


def calc_haversine_error(lat_pred, lon_pred, lat_true, lon_true):
    """
    Calculate the Haversine distance error in meters.

    Args:
        lat_pred, lon_pred: Predicted coordinates (degrees)
        lat_true, lon_true: Ground truth coordinates (degrees)

    Returns:
        distance: Distance in meters (numpy array or scalar)
    """
    R = 6371000.0  # Earth radius in meters

    phi1 = np.radians(lat_pred)
    phi2 = np.radians(lat_true)
    dphi = np.radians(lat_true - lat_pred)
    dlambda = np.radians(lon_true - lon_pred)

    a = (
        np.sin(dphi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    )

    # Clip a to [0, 1] to avoid numerical errors
    a = np.clip(a, 0.0, 1.0)

    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))

    return R * c
