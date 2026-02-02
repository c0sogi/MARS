import numpy as np

# WGS84 Ellipsoid Constants
WGS84_A = 6378137.0  # Semi-major axis (meters)
WGS84_F = 1 / 298.257223563  # Flattening factor
WGS84_B = WGS84_A * (1 - WGS84_F)  # Semi-minor axis
WGS84_E2 = 2 * WGS84_F - WGS84_F**2  # First eccentricity squared


def lla_to_ecef(lat, lon, alt):
    """
    Convert Latitude, Longitude, Altitude (LLA) to Earth-Centered, Earth-Fixed (ECEF) coordinates.

    Args:
        lat (np.array): Latitude in degrees.
        lon (np.array): Longitude in degrees.
        alt (np.array): Altitude in meters.

    Returns:
        tuple: (x, y, z) in ECEF coordinates (meters).
    """
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    N = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat_rad) ** 2)

    x = (N + alt) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (N + alt) * np.cos(lat_rad) * np.sin(lon_rad)
    z = (N * (1 - WGS84_E2) + alt) * np.sin(lat_rad)

    return x, y, z


def ecef_to_enu(x, y, z, ref_lat, ref_lon, ref_alt):
    """
    Convert ECEF coordinates to local East-North-Up (ENU) coordinates relative to a reference point.

    Args:
        x, y, z (np.array): ECEF coordinates of points.
        ref_lat, ref_lon, ref_alt (float): LLA coordinates of the reference point.

    Returns:
        tuple: (e, n, u) in ENU coordinates (meters).
    """
    # Convert reference point to ECEF
    ref_x, ref_y, ref_z = lla_to_ecef(ref_lat, ref_lon, ref_alt)

    # Difference vector
    dx = x - ref_x
    dy = y - ref_y
    dz = z - ref_z

    # Rotation matrix parameters
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


def wgs84_to_enu(lat, lon, alt, ref_lat, ref_lon, ref_alt):
    """
    Convert WGS84 LLA coordinates to local ENU coordinates.

    Args:
        lat, lon, alt (np.array): Target coordinates.
        ref_lat, ref_lon, ref_alt (float): Reference origin coordinates.

    Returns:
        tuple: (east, north, up) in meters.
    """
    x, y, z = lla_to_ecef(lat, lon, alt)
    return ecef_to_enu(x, y, z, ref_lat, ref_lon, ref_alt)


def flat_earth_scaling(lat, lon, ref_lat, ref_lon):
    """
    Approximate relative position in meters using flat earth scaling.
    This is computationally cheaper than WGS84->ENU and sufficient for small residuals.

    Args:
        lat (np.array): Target latitude.
        lon (np.array): Target longitude.
        ref_lat (float or np.array): Reference latitude.
        ref_lon (float or np.array): Reference longitude.

    Returns:
        tuple: (delta_east_meters, delta_north_meters)
    """
    # Constants
    METERS_PER_DEGREE_LAT = 111319.9

    delta_lat = lat - ref_lat
    delta_lon = lon - ref_lon

    delta_north = delta_lat * METERS_PER_DEGREE_LAT
    # Scale longitude difference by cosine of latitude
    delta_east = delta_lon * METERS_PER_DEGREE_LAT * np.cos(np.radians(ref_lat))

    return delta_east, delta_north


def haversine_loss(y_true_lat, y_true_lon, y_pred_lat, y_pred_lon):
    """
    Calculate the Haversine distance (great-circle distance) between two points.

    Args:
        y_true_lat (np.array): Ground truth latitude.
        y_true_lon (np.array): Ground truth longitude.
        y_pred_lat (np.array): Predicted latitude.
        y_pred_lon (np.array): Predicted longitude.

    Returns:
        np.array: Distance in meters.
    """
    R = 6371000  # Earth radius in meters

    phi1 = np.radians(y_true_lat)
    phi2 = np.radians(y_pred_lat)
    dphi = np.radians(y_pred_lat - y_true_lat)
    dlambda = np.radians(y_pred_lon - y_true_lon)

    a = (
        np.sin(dphi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    )

    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    meters = R * c
    return meters
