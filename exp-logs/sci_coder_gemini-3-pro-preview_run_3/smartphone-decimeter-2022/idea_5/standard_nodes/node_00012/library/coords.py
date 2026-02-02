import numpy as np
from library.config import Config


def geodetic_to_ecef(lat, lon, alt):
    """
    Convert Geodetic coordinates (Latitude, Longitude, Altitude) to ECEF coordinates.

    Args:
        lat (float or np.ndarray): Latitude in degrees.
        lon (float or np.ndarray): Longitude in degrees.
        alt (float or np.ndarray): Altitude in meters.

    Returns:
        tuple: (x, y, z) in meters.
    """
    a = Config.WGS84_A
    e2 = Config.WGS84_E2

    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)

    # Radius of curvature in the prime vertical
    N = a / np.sqrt(1 - e2 * sin_lat**2)

    x = (N + alt) * cos_lat * cos_lon
    y = (N + alt) * cos_lat * sin_lon
    z = (N * (1 - e2) + alt) * sin_lat

    return x, y, z


def ecef_to_geodetic(x, y, z):
    """
    Convert ECEF coordinates to Geodetic coordinates (Latitude, Longitude, Altitude).
    Uses an iterative method for high precision.

    Args:
        x (float or np.ndarray): ECEF X coordinate in meters.
        y (float or np.ndarray): ECEF Y coordinate in meters.
        z (float or np.ndarray): ECEF Z coordinate in meters.

    Returns:
        tuple: (lat, lon, alt) in degrees and meters.
    """
    a = Config.WGS84_A
    e2 = Config.WGS84_E2

    # Longitude is straightforward
    lon = np.arctan2(y, x)

    # Distance from Z-axis
    p = np.sqrt(x**2 + y**2)

    # Iterative solution for Latitude and Altitude
    # Initial guess
    lat = np.arctan2(z, p * (1 - e2))
    h = 0.0

    # Iterate to converge
    for _ in range(5):
        sin_lat = np.sin(lat)
        N = a / np.sqrt(1 - e2 * sin_lat**2)
        h = p / np.cos(lat) - N
        lat = np.arctan2(z, p * (1 - e2 * (N / (N + h))))

    return np.degrees(lat), np.degrees(lon), h


def get_rotation_matrix(lat, lon):
    """
    Calculate the rotation matrix to convert ECEF vectors to ENU frame
    at a specific reference lat/lon.

    Args:
        lat (float or np.ndarray): Reference latitude in degrees.
        lon (float or np.ndarray): Reference longitude in degrees.

    Returns:
        np.ndarray: Rotation matrix (3x3) or (N, 3, 3).
    """
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)

    # Row 1: East unit vector
    r11 = -sin_lon
    r12 = cos_lon
    r13 = np.zeros_like(lon_rad)

    # Row 2: North unit vector
    r21 = -sin_lat * cos_lon
    r22 = -sin_lat * sin_lon
    r23 = cos_lat

    # Row 3: Up unit vector
    r31 = cos_lat * cos_lon
    r32 = cos_lat * sin_lon
    r33 = sin_lat

    if np.ndim(lat) == 0:
        R = np.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]])
    else:
        # Stack for vectorized operations: shape (N, 3, 3)
        # Transpose to get correct shape from list of arrays
        R = np.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]]).transpose(
            2, 0, 1
        )

    return R


def ecef_to_enu(x, y, z, ref_lat, ref_lon, ref_alt):
    """
    Convert ECEF coordinates to Local Tangent Plane (ENU) coordinates
    relative to a reference point.

    Args:
        x, y, z: Target ECEF coordinates.
        ref_lat, ref_lon, ref_alt: Reference Geodetic coordinates.

    Returns:
        tuple: (east, north, up) in meters.
    """
    # Convert reference point to ECEF
    ref_x, ref_y, ref_z = geodetic_to_ecef(ref_lat, ref_lon, ref_alt)

    dx = x - ref_x
    dy = y - ref_y
    dz = z - ref_z

    # Get rotation matrix
    R = get_rotation_matrix(ref_lat, ref_lon)

    if np.ndim(x) == 0:
        # Scalar case
        d_vec = np.array([dx, dy, dz])
        enu = R @ d_vec
        return enu[0], enu[1], enu[2]
    else:
        # Vectorized case
        # R shape: (N, 3, 3), d_vec shape: (N, 3, 1)
        d_vec = np.stack([dx, dy, dz], axis=-1)[..., np.newaxis]
        enu = np.matmul(R, d_vec).squeeze(-1)
        return enu[:, 0], enu[:, 1], enu[:, 2]


def enu_to_ecef(east, north, up, ref_lat, ref_lon, ref_alt):
    """
    Convert ENU coordinates to ECEF coordinates relative to a reference point.

    Args:
        east, north, up: Local ENU coordinates in meters.
        ref_lat, ref_lon, ref_alt: Reference Geodetic coordinates.

    Returns:
        tuple: (x, y, z) ECEF coordinates in meters.
    """
    # Convert reference point to ECEF
    ref_x, ref_y, ref_z = geodetic_to_ecef(ref_lat, ref_lon, ref_alt)

    # Get rotation matrix (Forward: ECEF -> ENU)
    R = get_rotation_matrix(ref_lat, ref_lon)

    if np.ndim(east) == 0:
        # Scalar case
        enu_vec = np.array([east, north, up])
        # Inverse rotation is transpose for rotation matrices
        d_ecef = R.T @ enu_vec
        return ref_x + d_ecef[0], ref_y + d_ecef[1], ref_z + d_ecef[2]
    else:
        # Vectorized case
        # R shape: (N, 3, 3). Transpose last two dims for inverse
        R_inv = np.transpose(R, (0, 2, 1))
        enu_vec = np.stack([east, north, up], axis=-1)[..., np.newaxis]
        d_ecef = np.matmul(R_inv, enu_vec).squeeze(-1)

        return ref_x + d_ecef[:, 0], ref_y + d_ecef[:, 1], ref_z + d_ecef[:, 2]
