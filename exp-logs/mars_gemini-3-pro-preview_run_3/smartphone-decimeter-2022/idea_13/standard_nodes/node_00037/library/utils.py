import numpy as np
from library.config import haversine_distance, ecef_to_enu

# =============================================================================
# CONSTANTS (WGS84 Ellipsoid)
# =============================================================================
WGS84_A = 6378137.0  # Semi-major axis
WGS84_B = 6356752.3142  # Semi-minor axis
WGS84_F = (WGS84_A - WGS84_B) / WGS84_A  # Flattening
WGS84_E2 = WGS84_F * (2 - WGS84_F)  # First eccentricity squared
WGS84_EP2 = (WGS84_A**2 - WGS84_B**2) / WGS84_B**2  # Second eccentricity squared


# =============================================================================
# COORDINATE TRANSFORMATIONS
# =============================================================================


def lla_to_ecef(lat_deg, lon_deg, alt_m):
    """
    Convert Latitude, Longitude, Altitude to ECEF coordinates.

    Args:
        lat_deg: Latitude in degrees (float or numpy array)
        lon_deg: Longitude in degrees (float or numpy array)
        alt_m: Altitude in meters (float or numpy array)

    Returns:
        x, y, z: ECEF coordinates in meters
    """
    lat_rad = np.radians(lat_deg)
    lon_rad = np.radians(lon_deg)

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


def ecef_to_lla(x, y, z):
    """
    Convert ECEF coordinates to Latitude, Longitude, Altitude.
    Uses the closed-form solution (e.g., Ferrari's method approximation).

    Args:
        x, y, z: ECEF coordinates in meters (float or numpy array)

    Returns:
        lat (deg), lon (deg), alt (m)
    """
    p = np.sqrt(x**2 + y**2)
    theta = np.arctan2(z * WGS84_A, p * WGS84_B)

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    lon_rad = np.arctan2(y, x)

    lat_rad = np.arctan2(
        z + WGS84_EP2 * WGS84_B * sin_theta**3, p - WGS84_E2 * WGS84_A * cos_theta**3
    )

    sin_lat = np.sin(lat_rad)
    N = WGS84_A / np.sqrt(1 - WGS84_E2 * sin_lat**2)

    # Altitude calculation
    # Use standard formula, robust enough for non-polar regions
    alt_m = p / np.cos(lat_rad) - N

    return np.degrees(lat_rad), np.degrees(lon_rad), alt_m


def lla_to_enu(lat_deg, lon_deg, alt_m, ref_lat, ref_lon, ref_alt):
    """
    Convert target LLA coordinates to ENU relative to a reference LLA point.
    Useful for calculating local metric residuals (East, North, Up) from Ground Truth.

    Args:
        lat_deg, lon_deg, alt_m: Target coordinates
        ref_lat, ref_lon, ref_alt: Reference coordinates (origin of ENU frame)

    Returns:
        e, n, u: East, North, Up coordinates in meters
    """
    # 1. Convert target LLA to ECEF
    x, y, z = lla_to_ecef(lat_deg, lon_deg, alt_m)

    # 2. Convert ECEF to ENU using the reference point
    # Note: ecef_to_enu expects reference in LLA
    e, n, u = ecef_to_enu(x, y, z, ref_lat, ref_lon, ref_alt)

    return e, n, u
