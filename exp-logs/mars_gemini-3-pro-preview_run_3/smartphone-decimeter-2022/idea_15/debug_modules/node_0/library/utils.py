import numpy as np

# WGS84 Ellipsoid Constants
WGS84_A = 6378137.0  # Semi-major axis
WGS84_F = 1 / 298.257223563  # Flattening
WGS84_E2 = 2 * WGS84_F - WGS84_F**2  # Eccentricity squared


def geodetic_to_ecef(lat, lon, alt):
    """
    Convert Geodetic coordinates (Latitude, Longitude, Altitude) to ECEF coordinates.

    Args:
        lat (float or np.array): Latitude in degrees.
        lon (float or np.array): Longitude in degrees.
        alt (float or np.array): Altitude in meters.

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


def ecef_to_geodetic(x, y, z):
    """
    Convert ECEF coordinates to Geodetic coordinates (Latitude, Longitude, Altitude).
    Uses Heiskanen and Moritz iterative method.

    Args:
        x (float or np.array): ECEF X coordinate in meters.
        y (float or np.array): ECEF Y coordinate in meters.
        z (float or np.array): ECEF Z coordinate in meters.

    Returns:
        tuple: (lat, lon, alt) in degrees and meters.
    """
    # Longitude is straightforward
    lon = np.arctan2(y, x)

    # Iterative solution for Latitude and Altitude
    p = np.sqrt(x**2 + y**2)

    # Initial guess
    lat = np.arctan2(z, p * (1 - WGS84_E2))

    # Iterate
    # Usually converges in 3-4 iterations for GNSS altitudes
    for _ in range(5):
        N = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat) ** 2)
        h = p / np.cos(lat) - N
        lat = np.arctan2(z, p * (1 - WGS84_E2 * N / (N + h)))

    lat_deg = np.degrees(lat)
    lon_deg = np.degrees(lon)

    return lat_deg, lon_deg, h


def get_rotation_matrix(lat_rad, lon_rad):
    """
    Compute the rotation matrix from ECEF to ENU frame at a given reference lat/lon.

    Args:
        lat_rad (float or np.array): Reference latitude in radians.
        lon_rad (float or np.array): Reference longitude in radians.

    Returns:
        np.array: Rotation matrix (3x3) or (N, 3, 3).
    """
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)

    # Row 1: East
    r11 = -sin_lon
    r12 = cos_lon
    r13 = np.zeros_like(sin_lon)

    # Row 2: North
    r21 = -sin_lat * cos_lon
    r22 = -sin_lat * sin_lon
    r23 = cos_lat

    # Row 3: Up
    r31 = cos_lat * cos_lon
    r32 = cos_lat * sin_lon
    r33 = sin_lat

    # Stack into matrix
    # If inputs are arrays, we want (N, 3, 3)
    if np.ndim(lat_rad) > 0:
        R = np.zeros((len(lat_rad), 3, 3))
        R[:, 0, 0] = r11
        R[:, 0, 1] = r12
        R[:, 0, 2] = r13
        R[:, 1, 0] = r21
        R[:, 1, 1] = r22
        R[:, 1, 2] = r23
        R[:, 2, 0] = r31
        R[:, 2, 1] = r32
        R[:, 2, 2] = r33
    else:
        R = np.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]])

    return R


def ecef_to_enu(x, y, z, ref_x, ref_y, ref_z):
    """
    Convert ECEF coordinates to Local Tangent Plane (ENU) coordinates relative to a reference.

    Args:
        x, y, z (float or np.array): Target ECEF coordinates.
        ref_x, ref_y, ref_z (float or np.array): Reference ECEF coordinates (origin of ENU).

    Returns:
        tuple: (east, north, up) in meters.
    """
    # Calculate reference lat/lon for rotation
    ref_lat, ref_lon, _ = ecef_to_geodetic(ref_x, ref_y, ref_z)
    ref_lat_rad = np.radians(ref_lat)
    ref_lon_rad = np.radians(ref_lon)

    # Delta vector in ECEF
    dx = x - ref_x
    dy = y - ref_y
    dz = z - ref_z

    # Rotation Matrix
    R = get_rotation_matrix(ref_lat_rad, ref_lon_rad)

    # Apply rotation
    if np.ndim(x) > 0:
        # Vectorized matrix multiplication
        # R is (N, 3, 3), D is (N, 3)
        # We want (N, 3) result
        # Reshape D to (N, 3, 1) for matmul -> (N, 3, 1) -> flatten
        D = np.stack([dx, dy, dz], axis=1)[..., np.newaxis]
        enu = np.matmul(R, D).squeeze(-1)
        e, n, u = enu[:, 0], enu[:, 1], enu[:, 2]
    else:
        # Scalar
        D = np.array([dx, dy, dz])
        enu = R @ D
        e, n, u = enu[0], enu[1], enu[2]

    return e, n, u


def enu_to_ecef(e, n, u, ref_x, ref_y, ref_z):
    """
    Convert ENU coordinates back to ECEF coordinates given the reference origin.

    Args:
        e, n, u (float or np.array): ENU coordinates.
        ref_x, ref_y, ref_z (float or np.array): Reference ECEF coordinates.

    Returns:
        tuple: (x, y, z) in ECEF.
    """
    ref_lat, ref_lon, _ = ecef_to_geodetic(ref_x, ref_y, ref_z)
    ref_lat_rad = np.radians(ref_lat)
    ref_lon_rad = np.radians(ref_lon)

    R = get_rotation_matrix(ref_lat_rad, ref_lon_rad)

    # Transpose rotation matrix for inverse transform
    # R is orthogonal, so Inverse = Transpose
    if np.ndim(e) > 0:
        # R is (N, 3, 3) -> Transpose last two dims -> (N, 3, 3)
        R_inv = np.transpose(R, (0, 2, 1))
        ENU = np.stack([e, n, u], axis=1)[..., np.newaxis]
        d_ecef = np.matmul(R_inv, ENU).squeeze(-1)
        dx, dy, dz = d_ecef[:, 0], d_ecef[:, 1], d_ecef[:, 2]
    else:
        R_inv = R.T
        ENU = np.array([e, n, u])
        d_ecef = R_inv @ ENU
        dx, dy, dz = d_ecef[0], d_ecef[1], d_ecef[2]

    return ref_x + dx, ref_y + dy, ref_z + dz


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees).

    Args:
        lat1, lon1: First point coordinates.
        lat2, lon2: Second point coordinates.

    Returns:
        float or np.array: Distance in meters.
    """
    # Radius of earth in kilometers. Use 6371km or derive from WGS84 mean radius
    # Task metric usually implies standard haversine radius 6371000m or WGS84 ellipsoid distance
    # Simple Haversine uses sphere.
    R = 6371000.0

    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c
