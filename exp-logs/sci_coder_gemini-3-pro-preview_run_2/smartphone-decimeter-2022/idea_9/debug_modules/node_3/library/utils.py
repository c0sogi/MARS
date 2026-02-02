import numpy as np
from library.config import DEG_TO_M_LAT, DEG_TO_M_LON


def ecef_to_lla(x, y, z):
    """
    Convert Earth-Centered, Earth-Fixed (ECEF) coordinates to
    Latitude, Longitude, Altitude (LLA) using WGS84 ellipsoid constants.

    Args:
        x, y, z: ECEF coordinates in meters (scalars or numpy arrays).

    Returns:
        lat, lon, alt: Latitude (degrees), Longitude (degrees), Altitude (meters).
    """
    # WGS84 ellipsoid constants
    a = 6378137.0
    e = 8.1819190842622e-2  # Eccentricity

    b = np.sqrt(a**2 * (1 - e**2))
    ep = np.sqrt((a**2 - b**2) / b**2)

    p = np.sqrt(x**2 + y**2)
    th = np.arctan2(a * z, b * p)

    lon = np.arctan2(y, x)
    lat = np.arctan2(
        (z + ep**2 * b * np.sin(th) ** 3), (p - e**2 * a * np.cos(th) ** 3)
    )

    # Calculate altitude (N)
    sin_lat = np.sin(lat)
    N = a / np.sqrt(1 - e**2 * sin_lat**2)
    alt = p / np.cos(lat) - N

    # Convert radians to degrees
    lat = np.degrees(lat)
    lon = np.degrees(lon)

    return lat, lon, alt


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points on the Earth surface.

    Args:
        lat1, lon1: Coordinates of the first point (degrees).
        lat2, lon2: Coordinates of the second point (degrees).

    Returns:
        Distance in meters.
    """
    R = 6371000  # Radius of Earth in meters
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c


def degrees_to_meters(delta_lat_deg, delta_lon_deg, ref_lat_deg):
    """
    Convert differences in degrees to meters using a local flat Earth approximation.

    Args:
        delta_lat_deg: Difference in latitude (degrees).
        delta_lon_deg: Difference in longitude (degrees).
        ref_lat_deg: Reference latitude for longitude scaling (degrees).

    Returns:
        delta_lat_m, delta_lon_m: Differences in meters.
    """
    delta_lat_m = delta_lat_deg * DEG_TO_M_LAT
    # Scale longitude difference by cosine of latitude
    scale_lon = np.cos(np.radians(ref_lat_deg))
    delta_lon_m = delta_lon_deg * DEG_TO_M_LON * scale_lon

    return delta_lat_m, delta_lon_m


def meters_to_degrees(delta_lat_m, delta_lon_m, ref_lat_deg):
    """
    Convert differences in meters to degrees using a local flat Earth approximation.

    Args:
        delta_lat_m: Difference in latitude (meters).
        delta_lon_m: Difference in longitude (meters).
        ref_lat_deg: Reference latitude for longitude scaling (degrees).

    Returns:
        delta_lat_deg, delta_lon_deg: Differences in degrees.
    """
    delta_lat_deg = delta_lat_m / DEG_TO_M_LAT

    # Avoid division by zero at poles, though unlikely in this dataset
    scale_lon = np.cos(np.radians(ref_lat_deg))
    # Clip scale to avoid extreme values if near poles
    scale_lon = np.maximum(scale_lon, 1e-6)

    delta_lon_deg = delta_lon_m / (DEG_TO_M_LON * scale_lon)

    return delta_lat_deg, delta_lon_deg
