import numpy as np

# WGS84 Ellipsoid Constants
WGS84_A = 6378137.0  # Semi-major axis
WGS84_F = 1 / 298.257223563  # Flattening
WGS84_B = WGS84_A * (1 - WGS84_F)  # Semi-minor axis
WGS84_E2 = 2 * WGS84_F - WGS84_F**2  # Square of eccentricity

# Meters per degree latitude (approximate)
METERS_PER_DEGREE_LAT = 111319.9


def ecef_to_lla(x, y, z):
    """
    Convert Earth-Centered, Earth-Fixed (ECEF) coordinates to
    Latitude, Longitude, Altitude (LLA).

    Args:
        x (np.array): X coordinate in meters.
        y (np.array): Y coordinate in meters.
        z (np.array): Z coordinate in meters.

    Returns:
        tuple: (latitude_degrees, longitude_degrees, altitude_meters)
    """
    # Ensure inputs are numpy arrays
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    # Calculations
    p = np.sqrt(x**2 + y**2)
    theta = np.arctan2(z * WGS84_A, p * WGS84_B)

    lon = np.arctan2(y, x)

    num_lat = z + (WGS84_E2 * WGS84_A / (1 - WGS84_F)) * (np.sin(theta) ** 3)
    den_lat = p - (WGS84_E2 * WGS84_A) * (np.cos(theta) ** 3)
    lat = np.arctan2(num_lat, den_lat)

    N = WGS84_A / np.sqrt(1 - WGS84_E2 * (np.sin(lat) ** 2))
    alt = (p / np.cos(lat)) - N

    # Convert radians to degrees
    lat_deg = np.degrees(lat)
    lon_deg = np.degrees(lon)

    return lat_deg, lon_deg, alt


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees).

    Args:
        lat1, lon1: First point coordinates in degrees.
        lat2, lon2: Second point coordinates in degrees.

    Returns:
        np.array: Distance in meters.
    """
    # Radius of earth in meters
    R = 6371000.0

    # Convert degrees to radians
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = (
        np.sin(dphi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    )

    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c


def degrees_to_meters(lat_diff, lon_diff, ref_lat):
    """
    Convert differences in latitude and longitude degrees to meters
    using a local flat-earth approximation.

    Args:
        lat_diff (np.array): Difference in latitude (degrees).
        lon_diff (np.array): Difference in longitude (degrees).
        ref_lat (np.array): Reference latitude (degrees) for longitude scaling.

    Returns:
        tuple: (north_meters, east_meters)
    """
    # Latitude conversion is roughly constant
    north_m = lat_diff * METERS_PER_DEGREE_LAT

    # Longitude conversion depends on latitude
    # m_per_deg_lon = m_per_deg_lat * cos(lat)
    scale = np.cos(np.radians(ref_lat))
    east_m = lon_diff * METERS_PER_DEGREE_LAT * scale

    return north_m, east_m


def meters_to_degrees(north_m, east_m, ref_lat):
    """
    Convert differences in meters to latitude and longitude degrees
    using a local flat-earth approximation.

    Args:
        north_m (np.array): Difference in North direction (meters).
        east_m (np.array): Difference in East direction (meters).
        ref_lat (np.array): Reference latitude (degrees) for longitude scaling.

    Returns:
        tuple: (lat_diff_degrees, lon_diff_degrees)
    """
    # Latitude conversion
    lat_diff = north_m / METERS_PER_DEGREE_LAT

    # Longitude conversion
    scale = np.cos(np.radians(ref_lat))
    # Avoid division by zero at poles (though unlikely in dataset)
    scale = np.where(np.abs(scale) < 1e-9, 1e-9, scale)

    lon_diff = east_m / (METERS_PER_DEGREE_LAT * scale)

    return lat_diff, lon_diff
