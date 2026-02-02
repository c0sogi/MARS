import numpy as np

# -------------------------------------------------------------------------
# WGS84 Ellipsoid Constants
# -------------------------------------------------------------------------
A = 6378137.0  # Semi-major axis
F = 1 / 298.257223563  # Flattening
B = A * (1 - F)  # Semi-minor axis
E2 = 2 * F - F**2  # Square of eccentricity
E2_PRIME = E2 / (1 - E2)  # Square of second eccentricity


def wgs84_to_ecef(lat, lon, alt):
    """
    Convert WGS84 Geodetic coordinates to ECEF.

    Args:
        lat: Latitude in degrees (float or np.array)
        lon: Longitude in degrees (float or np.array)
        alt: Altitude in meters (float or np.array)

    Returns:
        x, y, z: ECEF coordinates in meters
    """
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    n = A / np.sqrt(1 - E2 * np.sin(lat_rad) ** 2)

    x = (n + alt) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (n + alt) * np.cos(lat_rad) * np.sin(lon_rad)
    z = (n * (1 - E2) + alt) * np.sin(lat_rad)

    return x, y, z


def ecef_to_wgs84(x, y, z):
    """
    Convert ECEF coordinates to WGS84 Geodetic coordinates.
    Using Heikkinen's method (non-iterative).

    Args:
        x, y, z: ECEF coordinates in meters (float or np.array)

    Returns:
        lat, lon, alt: Latitude and Longitude in degrees, Altitude in meters
    """
    p = np.sqrt(x**2 + y**2)
    theta = np.arctan2(z * A, p * B)

    lon = np.arctan2(y, x)

    lat_num = z + E2_PRIME * B * np.sin(theta) ** 3
    lat_den = p - E2 * A * np.cos(theta) ** 3
    lat = np.arctan2(lat_num, lat_den)

    n = A / np.sqrt(1 - E2 * np.sin(lat) ** 2)
    alt = p / np.cos(lat) - n

    return np.degrees(lat), np.degrees(lon), alt


def get_rotation_matrix(lat, lon):
    """
    Calculate the rotation matrix from ECEF to ENU at a reference latitude and longitude.

    Args:
        lat: Reference latitude in degrees
        lon: Reference longitude in degrees

    Returns:
        R: Rotation matrix (3x3 or Nx3x3 if input is array)
    """
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)

    # If inputs are arrays, we need to handle shapes to return (N, 3, 3)
    if np.ndim(lat) > 0:
        zeros = np.zeros_like(lat)
        # Construct matrix components
        r1 = np.stack([-sin_lon, cos_lon, zeros], axis=1)  # (N, 3)
        r2 = np.stack(
            [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat], axis=1
        )  # (N, 3)
        r3 = np.stack([cos_lat * cos_lon, cos_lat * sin_lon, sin_lat], axis=1)  # (N, 3)

        # Stack to form (N, 3, 3)
        R = np.stack([r1, r2, r3], axis=1)
        return R
    else:
        return np.array(
            [
                [-sin_lon, cos_lon, 0],
                [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
                [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat],
            ]
        )


def wgs84_to_enu(lat, lon, lat_ref, lon_ref, alt_ref=0):
    """
    Convert WGS84 coordinates to Local Tangent Plane (ENU) coordinates.

    Args:
        lat, lon: Target coordinates in degrees
        lat_ref, lon_ref: Reference coordinates in degrees
        alt_ref: Reference altitude in meters (default 0)

    Returns:
        e, n, u: East, North, Up coordinates in meters
    """
    # Convert target and reference to ECEF
    x, y, z = wgs84_to_ecef(lat, lon, alt_ref)
    xr, yr, zr = wgs84_to_ecef(lat_ref, lon_ref, alt_ref)

    dx, dy, dz = x - xr, y - yr, z - zr

    R = get_rotation_matrix(lat_ref, lon_ref)

    if np.ndim(lat) > 0:
        # Batch matrix multiplication
        # dx is (N,), R is (N, 3, 3)
        # We need dx as (N, 3, 1)
        d_vec = np.stack([dx, dy, dz], axis=1)[..., np.newaxis]
        enu = np.matmul(R, d_vec).squeeze(-1)  # (N, 3)
        return enu[:, 0], enu[:, 1], enu[:, 2]
    else:
        d_vec = np.array([dx, dy, dz])
        enu = R @ d_vec
        return enu[0], enu[1], enu[2]


def enu_to_wgs84(east, north, up, lat_ref, lon_ref, alt_ref=0):
    """
    Convert ENU coordinates back to WGS84 relative to a reference.

    Args:
        east, north, up: Local coordinates in meters
        lat_ref, lon_ref: Reference coordinates in degrees
        alt_ref: Reference altitude in meters

    Returns:
        lat, lon, alt: WGS84 coordinates
    """
    xr, yr, zr = wgs84_to_ecef(lat_ref, lon_ref, alt_ref)

    R = get_rotation_matrix(lat_ref, lon_ref)

    if np.ndim(east) > 0:
        # Transpose R for inverse rotation (R^T)
        # R is (N, 3, 3), R^T is (N, 3, 3) where we swap last two dims
        R_inv = np.transpose(R, (0, 2, 1))

        enu_vec = np.stack([east, north, up], axis=1)[..., np.newaxis]
        d_ecef = np.matmul(R_inv, enu_vec).squeeze(-1)  # (N, 3)

        x = xr + d_ecef[:, 0]
        y = yr + d_ecef[:, 1]
        z = zr + d_ecef[:, 2]
    else:
        R_inv = R.T
        enu_vec = np.array([east, north, up])
        d_ecef = R_inv @ enu_vec

        x = xr + d_ecef[0]
        y = yr + d_ecef[1]
        z = zr + d_ecef[2]

    return ecef_to_wgs84(x, y, z)


def calculate_azimuth_elevation(sat_pos, rec_pos):
    """
    Compute Azimuth and Elevation of satellites.

    Args:
        sat_pos: Satellite ECEF positions (N, 3)
        rec_pos: Receiver ECEF positions (N, 3)

    Returns:
        azimuth: Azimuth in degrees (0 to 360)
        elevation: Elevation in degrees (-90 to 90)
    """
    # 1. Calculate difference vector
    diff = sat_pos - rec_pos  # (N, 3)

    # 2. Get Receiver Lat/Lon for ENU rotation
    # Assuming rec_pos is ECEF
    lat, lon, _ = ecef_to_wgs84(rec_pos[:, 0], rec_pos[:, 1], rec_pos[:, 2])

    # 3. Rotate difference vector to ENU
    R = get_rotation_matrix(lat, lon)  # (N, 3, 3)

    # Reshape diff for matmul: (N, 3, 1)
    diff_exp = diff[..., np.newaxis]
    enu = np.matmul(R, diff_exp).squeeze(-1)  # (N, 3)

    e = enu[:, 0]
    n = enu[:, 1]
    u = enu[:, 2]

    # 4. Calculate Azimuth (degrees from North, clockwise)
    # arctan2(e, n) gives angle from North (n axis) towards East (e axis)
    az = np.degrees(np.arctan2(e, n))
    az = np.mod(az, 360)

    # 5. Calculate Elevation
    # angle between vector and horizontal plane
    r = np.sqrt(e**2 + n**2)
    el = np.degrees(np.arctan2(u, r))

    return az, el
