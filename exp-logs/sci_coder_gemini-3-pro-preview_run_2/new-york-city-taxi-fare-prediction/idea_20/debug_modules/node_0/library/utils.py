import numpy as np
import pandas as pd
from library.config import NYC_BBOX

# Base32 encoding character set for Geohash
__base32 = "0123456789bcdefghjkmnpqrstuvwxyz"
__decodemap = {k: v for v, k in enumerate(__base32)}


def clamp_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clamps the pickup and dropoff coordinates to the NYC bounding box defined in config.
    This prevents the model from seeing outliers or 0,0 coordinates that ruin
    distance calculations.

    Args:
        df: Input DataFrame containing pickup/dropoff latitude and longitude columns.

    Returns:
        DataFrame with clamped coordinates.
    """
    # Create a copy to avoid SettingWithCopy warnings on the original dataframe
    df = df.copy()

    # Unpack Bounding Box
    lon_min, lat_min, lon_max, lat_max = NYC_BBOX

    # Clamp Pickup
    if "pickup_longitude" in df.columns:
        df["pickup_longitude"] = df["pickup_longitude"].clip(lon_min, lon_max)
    if "pickup_latitude" in df.columns:
        df["pickup_latitude"] = df["pickup_latitude"].clip(lat_min, lat_max)

    # Clamp Dropoff
    if "dropoff_longitude" in df.columns:
        df["dropoff_longitude"] = df["dropoff_longitude"].clip(lon_min, lon_max)
    if "dropoff_latitude" in df.columns:
        df["dropoff_latitude"] = df["dropoff_latitude"].clip(lat_min, lat_max)

    return df


def calculate_haversine(lat1, lon1, lat2, lon2):
    """
    Vectorized calculation of the Haversine distance (Great Circle distance)
    between two points.

    Args:
        lat1, lon1: Start coordinates (float or numpy array) in degrees.
        lat2, lon2: End coordinates (float or numpy array) in degrees.

    Returns:
        Distance in kilometers.
    """
    R = 6371.0  # Earth radius in kilometers

    # Convert degrees to radians
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = (
        np.sin(dphi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    )

    # Handle potential floating point errors where a > 1
    a = np.clip(a, 0.0, 1.0)

    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c


def calculate_manhattan(lat1, lon1, lat2, lon2):
    """
    Vectorized calculation of the Manhattan distance (L1 norm) in Kilometers.
    Approximates the distance by converting degrees to km locally.

    Args:
        lat1, lon1: Start coordinates (float or numpy array) in degrees.
        lat2, lon2: End coordinates (float or numpy array) in degrees.

    Returns:
        Distance in kilometers.
    """
    # Average latitude for longitude scaling
    avg_lat_rad = np.radians((lat1 + lat2) / 2.0)

    # Conversion factors
    # 1 deg lat ~= 111.32 km
    # 1 deg lon ~= 111.32 * cos(lat) km
    lat_diff_km = np.abs(lat1 - lat2) * 111.32
    lon_diff_km = np.abs(lon1 - lon2) * 111.32 * np.cos(avg_lat_rad)

    return lat_diff_km + lon_diff_km


def _encode_single_geohash(lat, lon, precision=5):
    """
    Encodes a single latitude/longitude pair into a Geohash string.

    Args:
        lat: Latitude float.
        lon: Longitude float.
        precision: Length of the resulting geohash string.

    Returns:
        Geohash string.
    """
    lat_interval, lon_interval = (-90.0, 90.0), (-180.0, 180.0)
    geohash = []
    bits = [16, 8, 4, 2, 1]
    bit = 0
    ch = 0
    even = True

    while len(geohash) < precision:
        if even:
            mid = (lon_interval[0] + lon_interval[1]) / 2
            if lon > mid:
                ch |= bits[bit]
                lon_interval = (mid, lon_interval[1])
            else:
                lon_interval = (lon_interval[0], mid)
        else:
            mid = (lat_interval[0] + lat_interval[1]) / 2
            if lat > mid:
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


def encode_geohash(lat, lon, precision=5):
    """
    Vectorized wrapper to encode arrays of coordinates into Geohash strings.

    Args:
        lat: Numpy array or Series of latitudes.
        lon: Numpy array or Series of longitudes.
        precision: Integer precision (length of hash).

    Returns:
        Numpy array of geohash strings.
    """
    # Use numpy vectorize for convenience.
    # While a pure vectorized bitwise implementation is faster,
    # this is robust and sufficient for the subsampled training set.
    vectorized_func = np.vectorize(_encode_single_geohash)
    return vectorized_func(lat, lon, precision)
