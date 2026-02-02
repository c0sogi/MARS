import numpy as np
from library.config import WGS84_A, WGS84_B


def ecef_to_lla(x, y, z):
    """
    Convert Earth-Centered, Earth-Fixed (ECEF) coordinates to
    Geodetic coordinates (Latitude, Longitude, Altitude).

    This function uses Ferrari's closed-form solution.

    Args:
        x (float or np.array): ECEF X coordinate in meters.
        y (float or np.array): ECEF Y coordinate in meters.
        z (float or np.array): ECEF Z coordinate in meters.

    Returns:
        tuple: (lat, lon, alt)
            lat (float or np.array): Latitude in degrees.
            lon (float or np.array): Longitude in degrees.
            alt (float or np.array): Altitude in meters.
    """
    # Ensure inputs are numpy arrays for consistent vectorization
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    a = WGS84_A
    b = WGS84_B

    # Square of first eccentricity
    e2 = 1 - (b**2 / a**2)
    # Square of second eccentricity
    ep2 = (a**2 / b**2) - 1

    # Distance from Z-axis
    p = np.sqrt(x**2 + y**2)

    # Auxiliary angle theta
    # Use arctan2 to safely handle p=0 (though unlikely for valid GPS)
    th = np.arctan2(a * z, b * p)

    # Longitude
    lon = np.arctan2(y, x)

    # Latitude
    # numerator: z + ep2 * b * sin^3(theta)
    # denominator: p - e2 * a * cos^3(theta)
    lat = np.arctan2(z + ep2 * b * np.sin(th) ** 3, p - e2 * a * np.cos(th) ** 3)

    # Radius of curvature in the prime vertical
    N = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)

    # Altitude
    # To avoid numerical instability near poles (where cos(lat) -> 0),
    # we switch formulas based on latitude.
    # Threshold is typically 45 degrees (pi/4 radians).

    # Formula 1 (stable for non-polar regions): p / cos(lat) - N
    alt_equator = p / np.cos(lat) - N

    # Formula 2 (stable for polar regions): z / sin(lat) - N * (1 - e2)
    alt_pole = z / np.sin(lat) - N * (1 - e2)

    # Select appropriate altitude calculation
    # Using 45 degrees as the crossover point
    mask_pole = np.abs(lat) > (np.pi / 4)
    alt = np.where(mask_pole, alt_pole, alt_equator)

    # Convert radians to degrees
    lat = np.degrees(lat)
    lon = np.degrees(lon)

    return lat, lon, alt


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Compute the Haversine distance between two points on the Earth's surface.

    Args:
        lat1 (float or np.array): Latitude of first point in degrees.
        lon1 (float or np.array): Longitude of first point in degrees.
        lat2 (float or np.array): Latitude of second point in degrees.
        lon2 (float or np.array): Longitude of second point in degrees.

    Returns:
        float or np.array: Distance in meters.
    """
    R = 6371000.0  # Earth radius in meters

    # Convert degrees to radians
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    # Haversine formula
    a = (
        np.sin(delta_phi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    )

    # Clip a to [0, 1] to avoid numerical errors in sqrt (though unlikely with valid coords)
    a = np.clip(a, 0.0, 1.0)

    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    d = R * c
    return d


def compute_percentile_error(distances):
    """
    Compute the 50th and 95th percentile errors from a collection of distances.

    Args:
        distances (np.array): Array of distance errors in meters.

    Returns:
        tuple: (p50, p95)
            p50 (float): 50th percentile error.
            p95 (float): 95th percentile error.
    """
    if len(distances) == 0:
        return 0.0, 0.0

    p50 = np.percentile(distances, 50)
    p95 = np.percentile(distances, 95)

    return p50, p95
