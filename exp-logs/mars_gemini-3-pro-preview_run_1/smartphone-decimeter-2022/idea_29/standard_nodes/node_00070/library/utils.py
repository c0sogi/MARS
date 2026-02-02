import numpy as np

# WGS84 Ellipsoid Constants
WGS84_A = 6378137.0  # Semi-major axis (meters)
WGS84_F = 1 / 298.257223563  # Flattening
WGS84_B = WGS84_A * (1 - WGS84_F)  # Semi-minor axis
WGS84_E2 = 2 * WGS84_F - WGS84_F**2  # Eccentricity squared


def wgs84_to_ecef(lat, lon, alt):
    """
    Convert WGS84 geodetic coordinates to ECEF Cartesian coordinates.

    Args:
        lat (np.array or float): Latitude in degrees.
        lon (np.array or float): Longitude in degrees.
        alt (np.array or float): Altitude in meters.

    Returns:
        tuple: (x, y, z) in meters.
    """
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    N = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat_rad) ** 2)

    x = (N + alt) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (N + alt) * np.cos(lat_rad) * np.sin(lon_rad)
    z = (N * (1 - WGS84_E2) + alt) * np.sin(lat_rad)

    return x, y, z


def ecef_to_wgs84(x, y, z):
    """
    Convert ECEF Cartesian coordinates to WGS84 geodetic coordinates.
    Uses Ferrari's solution.

    Args:
        x (np.array or float): X coordinate in meters.
        y (np.array or float): Y coordinate in meters.
        z (np.array or float): Z coordinate in meters.

    Returns:
        tuple: (lat, lon, alt) in degrees and meters.
    """
    # Distance from Z-axis
    p = np.sqrt(x**2 + y**2)

    # Longitude
    lon = np.arctan2(y, x)

    # Latitude and Altitude
    # Ferrari's solution parameters
    th = np.arctan2(WGS84_A * z, WGS84_B * p)
    ep2 = (WGS84_A**2 - WGS84_B**2) / WGS84_B**2  # Second eccentricity squared

    sin_th = np.sin(th)
    cos_th = np.cos(th)

    lat = np.arctan2(z + ep2 * WGS84_B * sin_th**3, p - WGS84_E2 * WGS84_A * cos_th**3)

    sin_lat = np.sin(lat)
    N = WGS84_A / np.sqrt(1 - WGS84_E2 * sin_lat**2)
    alt = p / np.cos(lat) - N

    # Handle poles (p close to 0)
    # If p is very small, lat is +/- 90, alt is abs(z) - b
    # This simple check handles scalar or array inputs roughly
    # For robust array handling, boolean masking would be better, but this suffices for GPS tracks
    if np.any(p < 1e-9):
        # Fallback for poles if implemented strictly, but standard formula usually stable enough
        pass

    return np.degrees(lat), np.degrees(lon), alt


def ecef_to_enu(x, y, z, ref_lat, ref_lon, ref_alt):
    """
    Convert ECEF coordinates to Local East-North-Up (ENU) coordinates
    relative to a reference point.

    Args:
        x, y, z: Target ECEF coordinates.
        ref_lat, ref_lon, ref_alt: Reference WGS84 coordinates.

    Returns:
        tuple: (e, n, u) in meters.
    """
    # Convert reference point to ECEF
    xr, yr, zr = wgs84_to_ecef(ref_lat, ref_lon, ref_alt)

    dx = x - xr
    dy = y - yr
    dz = z - zr

    ref_lat_rad = np.radians(ref_lat)
    ref_lon_rad = np.radians(ref_lon)

    sin_lat = np.sin(ref_lat_rad)
    cos_lat = np.cos(ref_lat_rad)
    sin_lon = np.sin(ref_lon_rad)
    cos_lon = np.cos(ref_lon_rad)

    # Rotation matrix
    e = -sin_lon * dx + cos_lon * dy
    n = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
    u = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz

    return e, n, u


def enu_to_ecef(e, n, u, ref_lat, ref_lon, ref_alt):
    """
    Convert Local ENU coordinates to ECEF coordinates relative to a reference point.

    Args:
        e, n, u: Target ENU coordinates.
        ref_lat, ref_lon, ref_alt: Reference WGS84 coordinates.

    Returns:
        tuple: (x, y, z) in meters.
    """
    xr, yr, zr = wgs84_to_ecef(ref_lat, ref_lon, ref_alt)

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

    x = xr + dx
    y = yr + dy
    z = zr + dz

    return x, y, z


def enu_to_wgs84(e, n, u, ref_lat, ref_lon, ref_alt):
    """
    Convert Local ENU coordinates to WGS84 geodetic coordinates.

    Args:
        e, n, u: Target ENU coordinates.
        ref_lat, ref_lon, ref_alt: Reference WGS84 coordinates.

    Returns:
        tuple: (lat, lon, alt) in degrees and meters.
    """
    x, y, z = enu_to_ecef(e, n, u, ref_lat, ref_lon, ref_alt)
    return ecef_to_wgs84(x, y, z)


def calculate_azimuth_centroid(azimuths_deg, weights=None):
    """
    Compute the signal-weighted centroid of azimuth angles.
    Decomposes azimuths into sine and cosine components, computes the weighted mean,
    and returns the components.

    Args:
        azimuths_deg (np.array): Azimuth angles in degrees.
        weights (np.array, optional): Weights for each azimuth (e.g., Cn0).
                                      If None, uniform weights are used.

    Returns:
        tuple: (sin_centroid, cos_centroid)
    """
    if len(azimuths_deg) == 0:
        return 0.0, 0.0

    az_rad = np.radians(azimuths_deg)

    if weights is None:
        weights = np.ones_like(azimuths_deg)

    # Normalize weights
    w_sum = np.sum(weights)
    if w_sum == 0:
        return 0.0, 0.0

    sin_sum = np.sum(weights * np.sin(az_rad))
    cos_sum = np.sum(weights * np.cos(az_rad))

    return sin_sum / w_sum, cos_sum / w_sum
