import numpy as np
import pandas as pd
from library.config import Config

# -------------------------------------------------------------------------
# Distance Metrics
# -------------------------------------------------------------------------


def haversine_array(lat1, lon1, lat2, lon2):
    """
    Calculate the Great Circle distance between two points on the earth (specified in decimal degrees).
    Vectorized version using numpy.

    Args:
        lat1, lon1: Start point coordinates (float or array-like).
        lat2, lon2: End point coordinates (float or array-like).

    Returns:
        Distance in kilometers (float or numpy array).
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2

    # Clip 'a' to prevent numerical errors (sqrt of negative or > 1)
    a = np.clip(a, 0.0, 1.0)

    c = 2 * np.arcsin(np.sqrt(a))

    return Config.EARTH_RADIUS_KM * c


def manhattan_array(lat1, lon1, lat2, lon2):
    """
    Calculate the Manhattan distance (L1 norm) between two points on the earth.
    Approximated as the sum of the Haversine distance of the latitude difference
    and the Haversine distance of the longitude difference.

    Args:
        lat1, lon1: Start point coordinates.
        lat2, lon2: End point coordinates.

    Returns:
        Manhattan distance in kilometers.
    """
    # Distance along latitude (keeping longitude constant)
    dist_lat = haversine_array(lat1, lon1, lat2, lon1)

    # Distance along longitude (keeping latitude constant at the destination)
    # Note: We use lat2 for the longitude distance calculation to account for
    # the convergence of meridians at the specific latitude.
    dist_lon = haversine_array(lat2, lon1, lat2, lon2)

    return dist_lat + dist_lon


# -------------------------------------------------------------------------
# Geometric Transformations
# -------------------------------------------------------------------------


def rotate_coordinates(lat, lon, angle_deg):
    """
    Rotate coordinates by a given angle around the origin (0,0).
    Useful for aligning the coordinate system with the Manhattan street grid.

    Args:
        lat: Latitude (y-coordinate).
        lon: Longitude (x-coordinate).
        angle_deg: Rotation angle in degrees.

    Returns:
        Tuple (lat_rotated, lon_rotated)
    """
    angle_rad = np.radians(angle_deg)
    cos_val = np.cos(angle_rad)
    sin_val = np.sin(angle_rad)

    # Standard 2D rotation matrix:
    # x' = x cos(theta) - y sin(theta)
    # y' = x sin(theta) + y cos(theta)
    # Here x = lon, y = lat

    lon_rot = lon * cos_val - lat * sin_val
    lat_rot = lon * sin_val + lat * cos_val

    return lat_rot, lon_rot


def clamp_coordinates(df, lat_col, lon_col):
    """
    Clamp latitude and longitude values to the bounding box defined in Config.
    Modifies the DataFrame in-place to save memory.

    Args:
        df: Pandas DataFrame containing the coordinates.
        lat_col: Name of the latitude column.
        lon_col: Name of the longitude column.
    """
    # We use .loc to ensure we modify the original dataframe
    df.loc[:, lat_col] = df[lat_col].clip(
        lower=Config.BB_MIN_LAT, upper=Config.BB_MAX_LAT
    )
    df.loc[:, lon_col] = df[lon_col].clip(
        lower=Config.BB_MIN_LON, upper=Config.BB_MAX_LON
    )
    return df


# -------------------------------------------------------------------------
# Geohashing (Pure Python Implementation)
# -------------------------------------------------------------------------


def _encode_single_geohash(latitude, longitude, precision):
    """
    Encode a single lat/lon pair into a Geohash string of specified precision.
    Internal helper function.
    """
    __base32 = "0123456789bcdefghjkmnpqrstuvwxyz"
    lat_interval = (-90.0, 90.0)
    lon_interval = (-180.0, 180.0)
    geohash = []
    bits = [16, 8, 4, 2, 1]
    bit = 0
    ch = 0
    even = True

    while len(geohash) < precision:
        if even:
            mid = (lon_interval[0] + lon_interval[1]) / 2
            if longitude > mid:
                ch |= bits[bit]
                lon_interval = (mid, lon_interval[1])
            else:
                lon_interval = (lon_interval[0], mid)
        else:
            mid = (lat_interval[0] + lat_interval[1]) / 2
            if latitude > mid:
                ch |= bits[bit]
                lat_interval = (mid, lat_interval[1])
            else:
                lat_interval = (lat_interval[0], mid)

        even = not even
        if bit < 4:
            bit += 1
        else:
            geohash.append(__base32[ch])
            bit = 0
            ch = 0

    return "".join(geohash)


def encode_geohash(latitudes, longitudes, precision):
    """
    Generate Geohash strings for arrays of latitudes and longitudes.

    Args:
        latitudes: Iterable (list, Series, array) of latitude values.
        longitudes: Iterable (list, Series, array) of longitude values.
        precision: Integer length of the geohash string.

    Returns:
        List of geohash strings.
    """
    # Ensure inputs are iterable and of same length
    # Using list comprehension for performance (faster than np.vectorize for string ops)
    return [
        _encode_single_geohash(lat, lon, precision)
        for lat, lon in zip(latitudes, longitudes)
    ]
