import numpy as np
from library.config import set_seed

# ==========================================
# WGS84 Ellipsoid Constants
# ==========================================
WGS84_A = 6378137.0  # Semi-major axis (meters)
WGS84_F = 1 / 298.257223563  # Flattening
WGS84_B = WGS84_A * (1 - WGS84_F)  # Semi-minor axis (meters)
WGS84_E2 = 1 - (WGS84_B**2 / WGS84_A**2)  # First eccentricity squared


def geodetic_to_enu(lat, lon, lat0, lon0):
    """
    Convert geodetic coordinates (lat, lon) to local ENU coordinates (east, north)
    relative to a reference point (lat0, lon0) using WGS84 ellipsoid approximations.

    This function is used to transform the target Latitude/Longitude into
    local Cartesian offsets (Meters) which are easier for the model to regress.

    Args:
        lat: Target latitude in degrees (float or np.array).
        lon: Target longitude in degrees (float or np.array).
        lat0: Reference latitude in degrees (float or np.array).
        lon0: Reference longitude in degrees (float or np.array).

    Returns:
        d_east: East offset in meters.
        d_north: North offset in meters.
    """
    # Convert degrees to radians
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    lat0_rad = np.deg2rad(lat0)
    lon0_rad = np.deg2rad(lon0)

    # Calculate differences
    d_lat = lat_rad - lat0_rad
    d_lon = lon_rad - lon0_rad

    # Calculate radii of curvature at the reference latitude
    sin_lat0 = np.sin(lat0_rad)

    # Prime Vertical Radius of Curvature (N)
    # Radius of curvature in the prime vertical (East-West direction)
    N = WGS84_A / np.sqrt(1 - WGS84_E2 * sin_lat0**2)

    # Meridian Radius of Curvature (M)
    # Radius of curvature in the meridian (North-South direction)
    M = (WGS84_A * (1 - WGS84_E2)) / np.power(1 - WGS84_E2 * sin_lat0**2, 1.5)

    # Calculate ENU components
    # We approximate the local surface as flat for the small deltas involved in GNSS error
    d_north = d_lat * M
    d_east = d_lon * N * np.cos(lat0_rad)

    return d_east, d_north


def enu_to_geodetic(d_east, d_north, lat0, lon0):
    """
    Convert local ENU coordinates (east, north) back to geodetic coordinates (lat, lon)
    relative to a reference point (lat0, lon0).

    This function is used to transform the model's predicted offsets (Meters)
    back into global Latitude/Longitude degrees for submission.

    Args:
        d_east: East offset in meters (float or np.array).
        d_north: North offset in meters (float or np.array).
        lat0: Reference latitude in degrees (float or np.array).
        lon0: Reference longitude in degrees (float or np.array).

    Returns:
        lat: Target latitude in degrees.
        lon: Target longitude in degrees.
    """
    # Convert reference to radians
    lat0_rad = np.deg2rad(lat0)

    # Calculate radii of curvature at the reference latitude
    sin_lat0 = np.sin(lat0_rad)
    N = WGS84_A / np.sqrt(1 - WGS84_E2 * sin_lat0**2)
    M = (WGS84_A * (1 - WGS84_E2)) / np.power(1 - WGS84_E2 * sin_lat0**2, 1.5)

    # Calculate delta radians from meters
    d_lat_rad = d_north / M
    d_lon_rad = d_east / (N * np.cos(lat0_rad))

    # Convert back to degrees and add to reference
    lat = lat0 + np.rad2deg(d_lat_rad)
    lon = lon0 + np.rad2deg(d_lon_rad)

    return lat, lon


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees) using the Haversine formula.

    Args:
        lat1, lon1: First point coordinates (float or np.array).
        lat2, lon2: Second point coordinates (float or np.array).

    Returns:
        Distance in meters.
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.deg2rad, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    # Mean Earth radius in meters
    r = 6371000
    return c * r
