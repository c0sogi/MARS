import numpy as np
from library.config import WGS84_A, WGS84_F, HUBER_DELTA


def wgs84_to_ecef(lat, lon, alt):
    """
    Convert WGS84 geodetic coordinates to ECEF (Earth-Centered, Earth-Fixed) coordinates.

    Args:
        lat (float or np.array): Latitude in degrees.
        lon (float or np.array): Longitude in degrees.
        alt (float or np.array): Altitude in meters.

    Returns:
        tuple: (x, y, z) in meters.
    """
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)

    a = WGS84_A
    f = WGS84_F
    e2 = 2 * f - f**2

    N = a / np.sqrt(1 - e2 * np.sin(lat_rad) ** 2)

    x = (N + alt) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (N + alt) * np.cos(lat_rad) * np.sin(lon_rad)
    z = (N * (1 - e2) + alt) * np.sin(lat_rad)

    return x, y, z


def ecef_to_wgs84(x, y, z):
    """
    Convert ECEF coordinates to WGS84 geodetic coordinates.
    Uses Ferrari's solution for high precision.

    Args:
        x (float or np.array): ECEF X coordinate in meters.
        y (float or np.array): ECEF Y coordinate in meters.
        z (float or np.array): ECEF Z coordinate in meters.

    Returns:
        tuple: (lat, lon, alt) in degrees and meters.
    """
    a = WGS84_A
    f = WGS84_F
    b = a * (1 - f)
    e2 = 2 * f - f**2
    ep2 = (a**2 - b**2) / b**2

    p = np.sqrt(x**2 + y**2)
    theta = np.arctan2(z * a, p * b)

    lon_rad = np.arctan2(y, x)
    lat_rad = np.arctan2(
        z + ep2 * b * np.sin(theta) ** 3, p - e2 * a * np.cos(theta) ** 3
    )

    N = a / np.sqrt(1 - e2 * np.sin(lat_rad) ** 2)
    alt = p / np.cos(lat_rad) - N

    # Handle poles where cos(lat) is close to 0
    # For exact poles, p -> 0.
    # A robust way for alt is using z: alt = z / sin(lat) - N * (1-e2)
    # But for GNSS tracks, p/cos(lat) is generally stable enough except at very high latitudes.
    # We apply a simple mask if needed, but standard formula usually suffices for vehicle data.

    return np.rad2deg(lat_rad), np.rad2deg(lon_rad), alt


def ecef_to_enu(x, y, z, lat0, lon0, alt0):
    """
    Convert ECEF coordinates to local ENU (East, North, Up) coordinates
    relative to a reference point.

    Args:
        x, y, z (float or np.array): Target ECEF coordinates.
        lat0, lon0, alt0 (float): Reference WGS84 coordinates.

    Returns:
        tuple: (e, n, u) in meters.
    """
    # Convert reference point to ECEF
    x0, y0, z0 = wgs84_to_ecef(lat0, lon0, alt0)

    # Deltas
    dx = x - x0
    dy = y - y0
    dz = z - z0

    # Rotation matrix parameters
    phi = np.deg2rad(lat0)
    lam = np.deg2rad(lon0)

    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    sin_lam = np.sin(lam)
    cos_lam = np.cos(lam)

    # Rotation
    e = -sin_lam * dx + cos_lam * dy
    n = -sin_phi * cos_lam * dx - sin_phi * sin_lam * dy + cos_phi * dz
    u = cos_phi * cos_lam * dx + cos_phi * sin_lam * dy + sin_phi * dz

    return e, n, u


def enu_to_ecef(e, n, u, lat0, lon0, alt0):
    """
    Convert local ENU coordinates to ECEF coordinates relative to a reference point.

    Args:
        e, n, u (float or np.array): Local ENU coordinates.
        lat0, lon0, alt0 (float): Reference WGS84 coordinates.

    Returns:
        tuple: (x, y, z) in meters.
    """
    # Convert reference point to ECEF
    x0, y0, z0 = wgs84_to_ecef(lat0, lon0, alt0)

    # Rotation matrix parameters
    phi = np.deg2rad(lat0)
    lam = np.deg2rad(lon0)

    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    sin_lam = np.sin(lam)
    cos_lam = np.cos(lam)

    # Inverse Rotation
    dx = -sin_lam * e - sin_phi * cos_lam * n + cos_phi * cos_lam * u
    dy = cos_lam * e - sin_phi * sin_lam * n + cos_phi * sin_lam * u
    dz = cos_phi * n + sin_phi * u

    return x0 + dx, y0 + dy, z0 + dz


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points on the earth.

    Args:
        lat1, lon1: First point coordinates in degrees.
        lat2, lon2: Second point coordinates in degrees.

    Returns:
        float or np.array: Distance in meters.
    """
    R = 6371000.0  # Earth radius in meters

    phi1 = np.deg2rad(lat1)
    phi2 = np.deg2rad(lat2)
    dphi = np.deg2rad(lat2 - lat1)
    dlam = np.deg2rad(lon2 - lon1)

    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c


def huber_loss(residuals, delta=HUBER_DELTA):
    """
    Compute Huber Loss for residuals.
    L(a) = 0.5 * a^2            if |a| <= delta
         = delta * (|a| - 0.5 * delta)  otherwise

    Args:
        residuals (np.array): Difference between predicted and target values.
        delta (float): Threshold for transition from quadratic to linear loss.

    Returns:
        np.array: Loss values.
    """
    abs_r = np.abs(residuals)
    quadratic = np.minimum(abs_r, delta)
    linear = abs_r - quadratic
    return 0.5 * quadratic**2 + delta * linear
