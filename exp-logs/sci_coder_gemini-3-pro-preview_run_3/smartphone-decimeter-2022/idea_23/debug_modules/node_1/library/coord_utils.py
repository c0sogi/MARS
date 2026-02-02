import numpy as np

# WGS84 Ellipsoid Constants
WGS84_A = 6378137.0  # Semi-major axis (meters)
WGS84_F = 1 / 298.257223563  # Flattening
WGS84_B = WGS84_A * (1 - WGS84_F)  # Semi-minor axis
WGS84_E2 = 1 - (WGS84_B**2 / WGS84_A**2)  # Square of first eccentricity


def wgs84_to_ecef(lat, lon, alt):
    """
    Convert WGS84 geodetic coordinates to ECEF Cartesian coordinates.

    Args:
        lat (float or np.array): Latitude in degrees.
        lon (float or np.array): Longitude in degrees.
        alt (float or np.array): Altitude in meters.

    Returns:
        tuple: (x, y, z) in meters.
    """
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    # Radius of curvature in the prime vertical
    N = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat_rad) ** 2)

    x = (N + alt) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (N + alt) * np.cos(lat_rad) * np.sin(lon_rad)
    z = (N * (1 - WGS84_E2) + alt) * np.sin(lat_rad)

    return x, y, z


def get_enu_rotation_matrix(ref_lat, ref_lon):
    """
    Compute the rotation matrix from ECEF to ENU frame centered at (ref_lat, ref_lon).

    Args:
        ref_lat (float): Reference latitude in degrees.
        ref_lon (float): Reference longitude in degrees.

    Returns:
        np.array: 3x3 Rotation matrix.
    """
    lat_rad = np.radians(ref_lat)
    lon_rad = np.radians(ref_lon)

    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)

    # Row 1: East vector
    r11 = -sin_lon
    r12 = cos_lon
    r13 = 0

    # Row 2: North vector
    r21 = -sin_lat * cos_lon
    r22 = -sin_lat * sin_lon
    r23 = cos_lat

    # Row 3: Up vector
    r31 = cos_lat * cos_lon
    r32 = cos_lat * sin_lon
    r33 = sin_lat

    return np.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]])


def ecef_to_enu(x, y, z, ref_lat, ref_lon, ref_alt):
    """
    Convert ECEF coordinates to local ENU coordinates.

    Args:
        x (float or np.array): ECEF X coordinate in meters.
        y (float or np.array): ECEF Y coordinate in meters.
        z (float or np.array): ECEF Z coordinate in meters.
        ref_lat (float): Reference latitude in degrees (origin of ENU).
        ref_lon (float): Reference longitude in degrees (origin of ENU).
        ref_alt (float): Reference altitude in meters (origin of ENU).

    Returns:
        tuple: (east, north, up) in meters.
    """
    # Get reference point in ECEF
    ref_x, ref_y, ref_z = wgs84_to_ecef(ref_lat, ref_lon, ref_alt)

    # Difference vector
    dx = x - ref_x
    dy = y - ref_y
    dz = z - ref_z

    # Rotation matrix
    R = get_enu_rotation_matrix(ref_lat, ref_lon)

    # Apply rotation
    # If inputs are arrays, we need to handle shapes correctly
    # R is (3, 3), d is (3, N) or (3,)

    if np.ndim(x) > 0:
        # Stack inputs to shape (3, N)
        d_vec = np.vstack((dx, dy, dz))
        enu = R @ d_vec
        return enu[0], enu[1], enu[2]
    else:
        d_vec = np.array([dx, dy, dz])
        enu = R @ d_vec
        return enu[0], enu[1], enu[2]
