import numpy as np

# WGS84 Ellipsoid Constants
WGS84_A = 6378137.0  # Semi-major axis in meters
WGS84_F = 1 / 298.257223563  # Flattening
WGS84_B = WGS84_A * (1 - WGS84_F)  # Semi-minor axis
WGS84_E2 = 2 * WGS84_F - WGS84_F**2  # Square of first eccentricity
WGS84_EP2 = (WGS84_A**2 - WGS84_B**2) / (WGS84_B**2)  # Square of second eccentricity


def ecef_to_lla(x, y, z):
    """
    Convert Earth-Centered, Earth-Fixed (ECEF) coordinates to
    Latitude, Longitude, Altitude (LLA).

    Args:
        x (np.array or float): X coordinate in meters.
        y (np.array or float): Y coordinate in meters.
        z (np.array or float): Z coordinate in meters.

    Returns:
        tuple: (lat, lon, alt) in degrees and meters.
    """
    # Ensure inputs are numpy arrays for vectorization
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    p = np.sqrt(x**2 + y**2)
    theta = np.arctan2(WGS84_A * z, WGS84_B * p)

    lon = np.arctan2(y, x)

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    lat = np.arctan2(
        z + WGS84_EP2 * WGS84_B * sin_theta**3, p - WGS84_E2 * WGS84_A * cos_theta**3
    )

    # Calculate altitude
    sin_lat = np.sin(lat)
    N = WGS84_A / np.sqrt(1 - WGS84_E2 * sin_lat**2)
    alt = p / np.cos(lat) - N

    # Convert radians to degrees
    lat = np.degrees(lat)
    lon = np.degrees(lon)

    return lat, lon, alt


def get_local_scale_factors(lat_deg):
    """
    Calculate the meters per degree of Latitude and Longitude
    at a specific latitude.

    Args:
        lat_deg (float or np.array): Latitude in degrees.

    Returns:
        tuple: (meters_per_deg_lat, meters_per_deg_lon)
    """
    lat_rad = np.radians(lat_deg)
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)

    # Radius of curvature in the meridian (M)
    # M = a(1-e^2) / (1-e^2 sin^2 phi)^(3/2)
    term = 1 - WGS84_E2 * sin_lat**2
    M = (WGS84_A * (1 - WGS84_E2)) / (term**1.5)

    # Radius of curvature in the prime vertical (N)
    # N = a / sqrt(1-e^2 sin^2 phi)
    N = WGS84_A / np.sqrt(term)

    # Meters per degree
    # dLat = M * dPhi
    # dLon = N * cos(phi) * dLambda
    meters_per_deg_lat = M * (np.pi / 180.0)
    meters_per_deg_lon = N * cos_lat * (np.pi / 180.0)

    return meters_per_deg_lat, meters_per_deg_lon


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees).

    Args:
        lat1, lon1: First point coordinates (degrees).
        lat2, lon2: Second point coordinates (degrees).

    Returns:
        float or np.array: Distance in meters.
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2

    # Check for numerical stability (a should be <= 1.0)
    a = np.clip(a, 0.0, 1.0)

    c = 2.0 * np.arcsin(np.sqrt(a))

    # Radius of earth in meters (mean radius)
    R = 6371000.0
    return R * c
