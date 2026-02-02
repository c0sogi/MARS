import numpy as np
from library.config import Config


def wgs84_to_cartesian(lat, lon, lat_ref, lon_ref):
    """
    Convert WGS84 coordinates (Latitude, Longitude) to Cartesian offsets (North, East)
    relative to a reference point. This uses a flat-earth approximation suitable for
    local localization tasks.

    Args:
        lat (float or np.ndarray): Target Latitude in degrees.
        lon (float or np.ndarray): Target Longitude in degrees.
        lat_ref (float or np.ndarray): Reference Latitude in degrees.
        lon_ref (float or np.ndarray): Reference Longitude in degrees.

    Returns:
        tuple: (north, east) where:
            north (float or np.ndarray): Offset in meters towards North.
            east (float or np.ndarray): Offset in meters towards East.
    """
    # Calculate differences in degrees
    delta_lat = lat - lat_ref
    delta_lon = lon - lon_ref

    # Convert to meters using constants defined in Config
    # These constants are approximations for the competition region
    north = delta_lat * Config.LAT_TO_M
    east = delta_lon * Config.LON_TO_M

    return north, east


def cartesian_to_wgs84(north, east, lat_ref, lon_ref):
    """
    Convert Cartesian offsets (North, East) back to WGS84 coordinates (Latitude, Longitude)
    relative to a reference point. This is the inverse of wgs84_to_cartesian.

    Args:
        north (float or np.ndarray): Offset in meters towards North.
        east (float or np.ndarray): Offset in meters towards East.
        lat_ref (float or np.ndarray): Reference Latitude in degrees.
        lon_ref (float or np.ndarray): Reference Longitude in degrees.

    Returns:
        tuple: (latitude, longitude) in degrees.
    """
    # Convert meters back to degrees
    delta_lat = north / Config.LAT_TO_M
    delta_lon = east / Config.LON_TO_M

    # Add offsets to the reference coordinates
    lat = lat_ref + delta_lat
    lon = lon_ref + delta_lon

    return lat, lon


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the Haversine distance between two points on the Earth's surface.
    This is the standard metric for calculating geodesic distance between WGS84 coordinates.

    Args:
        lat1 (float or np.ndarray): Latitude of the first point(s) in degrees.
        lon1 (float or np.ndarray): Longitude of the first point(s) in degrees.
        lat2 (float or np.ndarray): Latitude of the second point(s) in degrees.
        lon2 (float or np.ndarray): Longitude of the second point(s) in degrees.

    Returns:
        float or np.ndarray: Distance in meters.
    """
    R = 6371000.0  # Radius of Earth in meters

    # Convert degrees to radians
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    # Haversine formula
    a = (
        np.sin(dphi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    )

    # Numerical stability for arctan2
    a = np.clip(a, 0, 1)

    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))

    distance = R * c
    return distance
