import numpy as np
from library.config import WGS84_A, WGS84_B, WGS84_F


def wgs84_to_ecef(lat, lon, alt):
    """
    Convert WGS84 geodetic coordinates to ECEF coordinates.

    Args:
        lat (float or np.array): Latitude in degrees.
        lon (float or np.array): Longitude in degrees.
        alt (float or np.array): Altitude in meters.

    Returns:
        tuple: (x, y, z) in ECEF coordinates (meters).
    """
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    # Calculate e^2 (first eccentricity squared)
    # e^2 = 2f - f^2
    e2 = 2 * WGS84_F - WGS84_F**2

    # Radius of curvature in the prime vertical
    N = WGS84_A / np.sqrt(1 - e2 * np.sin(lat_rad) ** 2)

    x = (N + alt) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (N + alt) * np.cos(lat_rad) * np.sin(lon_rad)
    z = (N * (1 - e2) + alt) * np.sin(lat_rad)

    return x, y, z


def ecef_to_enu(x, y, z, ref_lat, ref_lon, ref_alt):
    """
    Convert ECEF coordinates to local East-North-Up (ENU) coordinates
    relative to a reference point.

    Args:
        x, y, z (float or np.array): Target ECEF coordinates.
        ref_lat, ref_lon, ref_alt (float): Reference WGS84 coordinates.

    Returns:
        tuple: (east, north, up) in meters.
    """
    # Convert reference point to ECEF
    ref_x, ref_y, ref_z = wgs84_to_ecef(ref_lat, ref_lon, ref_alt)

    # Difference vector
    dx = x - ref_x
    dy = y - ref_y
    dz = z - ref_z

    # Rotation matrix components
    lat_rad = np.radians(ref_lat)
    lon_rad = np.radians(ref_lon)

    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)

    # Rotation matrix multiplication
    # Row 1 (East): -sin_lon, cos_lon, 0
    east = -sin_lon * dx + cos_lon * dy

    # Row 2 (North): -sin_lat*cos_lon, -sin_lat*sin_lon, cos_lat
    north = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz

    # Row 3 (Up): cos_lat*cos_lon, cos_lat*sin_lon, sin_lat
    up = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz

    return east, north, up


def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees).

    Args:
        lat1, lon1: First point coordinates (degrees).
        lat2, lon2: Second point coordinates (degrees).

    Returns:
        float or np.array: Distance in meters.
    """
    # Radius of earth in meters
    R = 6371000.0

    # Convert decimal degrees to radians
    phi1, lambda1 = np.radians(lat1), np.radians(lon1)
    phi2, lambda2 = np.radians(lat2), np.radians(lon2)

    dphi = phi2 - phi1
    dlambda = lambda2 - lambda1

    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2

    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance = R * c
    return distance


def euclidean_distance(x1, y1, z1, x2, y2, z2):
    """
    Calculate Euclidean distance between two 3D points.

    Args:
        x1, y1, z1: First point coordinates.
        x2, y2, z2: Second point coordinates.

    Returns:
        float or np.array: Distance.
    """
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)


def calculate_los_vector(sat_x, sat_y, sat_z, user_x, user_y, user_z):
    """
    Calculate the Line-of-Sight (LOS) unit vector from user to satellite.

    Args:
        sat_x, sat_y, sat_z: Satellite ECEF coordinates.
        user_x, user_y, user_z: User ECEF coordinates.

    Returns:
        tuple: (u_x, u_y, u_z) Unit vector components.
    """
    dx = sat_x - user_x
    dy = sat_y - user_y
    dz = sat_z - user_z

    dist = np.sqrt(dx**2 + dy**2 + dz**2)

    # Avoid division by zero
    dist = np.where(dist == 0, 1e-9, dist)

    return dx / dist, dy / dist, dz / dist


def project_velocity(vx, vy, vz, ux, uy, uz):
    """
    Project a velocity vector onto a unit direction vector (dot product).

    Args:
        vx, vy, vz: Velocity vector components.
        ux, uy, uz: Unit direction vector components.

    Returns:
        float or np.array: Projected velocity (scalar).
    """
    return vx * ux + vy * uy + vz * uz
