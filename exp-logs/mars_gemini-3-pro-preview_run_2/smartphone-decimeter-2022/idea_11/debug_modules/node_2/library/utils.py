import numpy as np


def ecef_to_lla(x, y, z):
    """
    Convert Earth-Centered, Earth-Fixed (ECEF) coordinates to
    Latitude, Longitude, Altitude (LLA) using WGS84 ellipsoid.

    Args:
        x, y, z: ECEF coordinates in meters (float or np.array)

    Returns:
        lat, lon, alt: Latitude (deg), Longitude (deg), Altitude (m)
    """
    # WGS84 ellipsoid constants
    a = 6378137.0
    e = 8.1819190842622e-2  # eccentricity

    b = np.sqrt(a**2 * (1 - e**2))
    ep = np.sqrt((a**2 - b**2) / b**2)

    p = np.sqrt(x**2 + y**2)
    th = np.arctan2(a * z, b * p)

    lon = np.arctan2(y, x)
    lat = np.arctan2(
        (z + ep**2 * b * np.sin(th) ** 3), (p - e**2 * a * np.cos(th) ** 3)
    )

    # Calculate altitude
    # N is the radius of curvature in the prime vertical
    sin_lat = np.sin(lat)
    N = a / np.sqrt(1 - e**2 * sin_lat**2)
    # Handle scalar vs array input for altitude calculation
    if np.isscalar(lat):
        cos_lat = np.cos(lat)
        # Avoid division by zero near poles
        if abs(cos_lat) < 1e-6:
            alt = np.abs(z) - b  # Approximate at poles
        else:
            alt = p / cos_lat - N
    else:
        cos_lat = np.cos(lat)
        alt = np.zeros_like(lat)
        mask = np.abs(cos_lat) < 1e-6
        alt[mask] = np.abs(z[mask]) - b
        alt[~mask] = p[~mask] / cos_lat[~mask] - N[~mask]

    # Convert to degrees
    lat = np.degrees(lat)
    lon = np.degrees(lon)

    return lat, lon, alt


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the Haversine distance between two points on the Earth.

    Args:
        lat1, lon1: First point coordinates in degrees
        lat2, lon2: Second point coordinates in degrees

    Returns:
        distance: Distance in meters
    """
    R = 6371000.0  # Radius of Earth in meters

    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c


def latlon_to_meters_diff(lat_base, lon_base, lat_target, lon_target):
    """
    Convert latitude/longitude difference to meters (North/East)
    using local flat earth approximation.

    Args:
        lat_base, lon_base: Base coordinates (deg)
        lat_target, lon_target: Target coordinates (deg)

    Returns:
        d_east, d_north: Difference in meters
    """
    DEG_TO_M = 111320.0

    d_lat = lat_target - lat_base
    d_lon = lon_target - lon_base

    d_north = d_lat * DEG_TO_M
    scale_lon = np.cos(np.radians(lat_base))
    d_east = d_lon * DEG_TO_M * scale_lon

    return d_east, d_north


def meters_diff_to_latlon(lat_base, lon_base, d_east, d_north):
    """
    Convert meters difference (North/East) to latitude/longitude
    using local flat earth approximation.

    Args:
        lat_base, lon_base: Base coordinates (deg)
        d_east, d_north: Difference in meters

    Returns:
        lat_target, lon_target: Target coordinates (deg)
    """
    DEG_TO_M = 111320.0

    d_lat = d_north / DEG_TO_M
    lat_target = lat_base + d_lat

    scale_lon = np.cos(np.radians(lat_base))
    # Avoid division by zero at poles
    scale_lon = np.where(np.abs(scale_lon) < 1e-9, 1e-9, scale_lon)

    d_lon = d_east / (DEG_TO_M * scale_lon)
    lon_target = lon_base + d_lon

    return lat_target, lon_target
