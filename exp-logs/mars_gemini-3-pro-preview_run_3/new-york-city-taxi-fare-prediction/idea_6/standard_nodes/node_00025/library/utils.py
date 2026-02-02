import numpy as np


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculates the Haversine distance between two points on the earth.

    Args:
        lat1: Latitude of the first point(s).
        lon1: Longitude of the first point(s).
        lat2: Latitude of the second point(s).
        lon2: Longitude of the second point(s).

    Returns:
        Distance in kilometers.
    """
    # Radius of earth in kilometers
    R = 6371.0

    # Convert degrees to radians
    phi1 = np.radians(lat1)
    lambda1 = np.radians(lon1)
    phi2 = np.radians(lat2)
    lambda2 = np.radians(lon2)

    dphi = phi2 - phi1
    dlambda = lambda2 - lambda1

    # Haversine formula
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2

    # Ensure value is within domain of arcsin [0, 1] due to floating point errors
    a = np.clip(a, 0, 1)

    c = 2 * np.arcsin(np.sqrt(a))

    return R * c


def rotate_coordinates(lat, lon, angle_rad):
    """
    Rotates latitude and longitude coordinates by a specified angle.
    This is useful for aligning coordinates with the NYC street grid.

    Args:
        lat: Latitude(s) (y-coordinate).
        lon: Longitude(s) (x-coordinate).
        angle_rad: Rotation angle in radians.

    Returns:
        Tuple of (rotated_lat, rotated_lon).
    """
    # Treat longitude as x and latitude as y
    x = lon
    y = lat

    # Rotation matrix
    # x_new = x * cos(theta) - y * sin(theta)
    # y_new = x * sin(theta) + y * cos(theta)

    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)

    x_new = x * cos_theta - y * sin_theta
    y_new = x * sin_theta + y * cos_theta

    # Return (lat_new, lon_new) corresponding to (y_new, x_new)
    return y_new, x_new
