import numpy as np
from library.config import Config


def ecef_to_lla(x, y, z):
    """
    Convert Earth-Centered, Earth-Fixed (ECEF) coordinates to
    Latitude, Longitude, and Altitude (LLA) using WGS84 constants.

    Args:
        x (np.array or float): X coordinate in meters.
        y (np.array or float): Y coordinate in meters.
        z (np.array or float): Z coordinate in meters.

    Returns:
        tuple: (latitude_deg, longitude_deg, altitude_m)
    """
    # WGS84 ellipsoid constants
    a = 6378137.0
    e = 8.1819190842622e-2  # Eccentricity

    asq = a**2
    esq = e**2

    b = np.sqrt(asq * (1 - esq))
    b_sq = b**2
    ep = np.sqrt((asq - b_sq) / b_sq)

    p = np.sqrt(x**2 + y**2)
    th = np.arctan2(a * z, b * p)

    lon = np.arctan2(y, x)
    lat = np.arctan2((z + ep**2 * b * np.sin(th) ** 3), (p - esq * a * np.cos(th) ** 3))

    # Calculate Altitude
    # N is the radius of curvature in the prime vertical
    sin_lat = np.sin(lat)
    N = a / np.sqrt(1 - esq * sin_lat**2)
    alt = p / np.cos(lat) - N

    # Convert radians to degrees
    lat_deg = np.degrees(lat)
    lon_deg = np.degrees(lon)

    return lat_deg, lon_deg, alt


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the Haversine distance between two points on the Earth.

    Args:
        lat1 (np.array or float): Latitude of the first point in degrees.
        lon1 (np.array or float): Longitude of the first point in degrees.
        lat2 (np.array or float): Latitude of the second point in degrees.
        lon2 (np.array or float): Longitude of the second point in degrees.

    Returns:
        np.array or float: Distance in meters.
    """
    R = 6371000.0  # Radius of Earth in meters

    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2

    # Clip a to [0, 1] to avoid numerical errors (sqrt of negative)
    a = np.clip(a, 0, 1)

    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c


def degrees_to_meters_diff(d_lat_deg, d_lon_deg, ref_lat_deg):
    """
    Convert differences in degrees to differences in meters using a local
    flat-earth approximation (scaling).

    Args:
        d_lat_deg (np.array or float): Difference in latitude in degrees.
        d_lon_deg (np.array or float): Difference in longitude in degrees.
        ref_lat_deg (np.array or float): Reference latitude in degrees for longitude scaling.

    Returns:
        tuple: (d_lat_meters, d_lon_meters)
    """
    lat_scale = Config.LAT_SCALE

    d_lat_m = d_lat_deg * lat_scale
    d_lon_m = d_lon_deg * lat_scale * np.cos(np.radians(ref_lat_deg))

    return d_lat_m, d_lon_m


def meters_to_degrees_diff(d_lat_m, d_lon_m, ref_lat_deg):
    """
    Convert differences in meters to differences in degrees using a local
    flat-earth approximation (inverse scaling).

    Args:
        d_lat_m (np.array or float): Difference in latitude in meters.
        d_lon_m (np.array or float): Difference in longitude in meters.
        ref_lat_deg (np.array or float): Reference latitude in degrees for longitude scaling.

    Returns:
        tuple: (d_lat_deg, d_lon_deg)
    """
    lat_scale = Config.LAT_SCALE

    d_lat_deg = d_lat_m / lat_scale

    # Avoid division by zero at poles, though unlikely in this dataset
    cos_lat = np.cos(np.radians(ref_lat_deg))
    # Clip cos_lat to avoid explosion if exactly 0 (unlikely)
    cos_lat = np.where(np.abs(cos_lat) < 1e-6, 1e-6, cos_lat)

    d_lon_deg = d_lon_m / (lat_scale * cos_lat)

    return d_lat_deg, d_lon_deg
