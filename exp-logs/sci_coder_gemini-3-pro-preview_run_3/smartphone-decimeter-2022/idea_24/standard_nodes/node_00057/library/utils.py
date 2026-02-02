import numpy as np

# --- WGS84 Ellipsoid Constants ---
WGS84_A = 6378137.0  # Semi-major axis (meters)
WGS84_F = 1 / 298.257223563  # Flattening
WGS84_B = WGS84_A * (1 - WGS84_F)  # Semi-minor axis
WGS84_E2 = 2 * WGS84_F - WGS84_F**2  # Eccentricity squared


def llh_to_ecef(lat, lon, alt):
    """
    Convert Latitude, Longitude, Altitude (WGS84) to ECEF coordinates.

    Args:
        lat: Latitude in degrees.
        lon: Longitude in degrees.
        alt: Altitude in meters.

    Returns:
        x, y, z: ECEF coordinates in meters.
    """
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)

    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)

    # Radius of curvature in the prime vertical
    N = WGS84_A / np.sqrt(1 - WGS84_E2 * sin_lat**2)

    x = (N + alt) * cos_lat * cos_lon
    y = (N + alt) * cos_lat * sin_lon
    z = (N * (1 - WGS84_E2) + alt) * sin_lat

    return x, y, z


def ecef_to_llh(x, y, z):
    """
    Convert ECEF coordinates to Latitude, Longitude, Altitude (WGS84).
    Uses a robust iterative method or closed-form approximation.

    Args:
        x, y, z: ECEF coordinates in meters.

    Returns:
        lat, lon, alt: Latitude (deg), Longitude (deg), Altitude (m).
    """
    # Distance from Z-axis
    p = np.sqrt(x**2 + y**2)

    # Longitude
    lon = np.arctan2(y, x)

    # Latitude (Iterative method)
    # Initial guess
    lat = np.arctan2(z, p * (1 - WGS84_E2))

    # Iterate to converge
    for _ in range(5):
        sin_lat = np.sin(lat)
        N = WGS84_A / np.sqrt(1 - WGS84_E2 * sin_lat**2)
        alt = p / np.cos(lat) - N
        lat = np.arctan2(z, p * (1 - WGS84_E2 * (N / (N + alt))))

    lat_deg = np.rad2deg(lat)
    lon_deg = np.rad2deg(lon)

    return lat_deg, lon_deg, alt


def ecef_to_enu(x, y, z, ref_lat, ref_lon, ref_alt):
    """
    Convert ECEF coordinates to local ENU (East, North, Up) frame.

    Args:
        x, y, z: Target ECEF coordinates (meters).
        ref_lat, ref_lon, ref_alt: Reference point LLH (deg, deg, m).

    Returns:
        e, n, u: ENU coordinates in meters relative to reference.
    """
    # Convert reference point to ECEF
    ref_x, ref_y, ref_z = llh_to_ecef(ref_lat, ref_lon, ref_alt)

    # Difference vector
    dx = x - ref_x
    dy = y - ref_y
    dz = z - ref_z

    # Rotation matrix components
    lat_rad = np.deg2rad(ref_lat)
    lon_rad = np.deg2rad(ref_lon)

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
    Convert local ENU coordinates to ECEF frame.

    Args:
        e, n, u: ENU coordinates (meters).
        ref_lat, ref_lon, ref_alt: Reference point LLH (deg, deg, m).

    Returns:
        x, y, z: ECEF coordinates in meters.
    """
    # Convert reference point to ECEF
    ref_x, ref_y, ref_z = llh_to_ecef(ref_lat, ref_lon, ref_alt)

    # Rotation matrix components
    lat_rad = np.deg2rad(ref_lat)
    lon_rad = np.deg2rad(ref_lon)

    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)

    # Inverse Rotation (Transpose)
    # R = [ -sin_lon,           cos_lon,          0      ]
    #     [ -sin_lat*cos_lon,  -sin_lat*sin_lon,  cos_lat]
    #     [  cos_lat*cos_lon,   cos_lat*sin_lon,  sin_lat]
    #
    # dx = R.T * [e, n, u]^T

    dx = -sin_lon * e - sin_lat * cos_lon * n + cos_lat * cos_lon * u
    dy = cos_lon * e - sin_lat * sin_lon * n + cos_lat * sin_lon * u
    dz = cos_lat * n + sin_lat * u

    x = ref_x + dx
    y = ref_y + dy
    z = ref_z + dz

    return x, y, z


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees).

    Args:
        lat1, lon1: First point coordinates in degrees.
        lat2, lon2: Second point coordinates in degrees.

    Returns:
        Distance in meters.
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.deg2rad, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371000  # Radius of earth in meters
    return c * r
