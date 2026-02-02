import numpy as np
import pandas as pd
from library.config import Config


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculates the Haversine distance between two points or arrays of points.

    Args:
        lat1, lon1: Latitude and Longitude of starting point(s).
        lat2, lon2: Latitude and Longitude of destination point(s).

    Returns:
        Distance in kilometers (float or np.array).
    """
    R = 6371.0  # Earth radius in kilometers

    # Convert to radians
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = (
        np.sin(dphi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    )

    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c


def clamp_coordinates(df):
    """
    Clamps the pickup and dropoff coordinates to the NYC bounding box defined in Config.

    Args:
        df (pd.DataFrame): DataFrame containing coordinate columns.

    Returns:
        pd.DataFrame: DataFrame with clamped coordinates.
    """
    df = df.copy()

    # Define bounds
    min_lon, max_lon = Config.MIN_LON, Config.MAX_LON
    min_lat, max_lat = Config.MIN_LAT, Config.MAX_LAT

    # Clamp Pickup
    if "pickup_longitude" in df.columns:
        df["pickup_longitude"] = df["pickup_longitude"].clip(min_lon, max_lon)
    if "pickup_latitude" in df.columns:
        df["pickup_latitude"] = df["pickup_latitude"].clip(min_lat, max_lat)

    # Clamp Dropoff
    if "dropoff_longitude" in df.columns:
        df["dropoff_longitude"] = df["dropoff_longitude"].clip(min_lon, max_lon)
    if "dropoff_latitude" in df.columns:
        df["dropoff_latitude"] = df["dropoff_latitude"].clip(min_lat, max_lat)

    return df


def encode_geohash(lat, lon, precision=7):
    """
    Vectorized Geohash encoding.

    Args:
        lat (np.array or pd.Series): Latitudes.
        lon (np.array or pd.Series): Longitudes.
        precision (int): Length of the geohash string.

    Returns:
        np.array: Array of geohash strings.
    """
    # Ensure inputs are numpy arrays
    lat = np.asarray(lat)
    lon = np.asarray(lon)

    # Base32 character map
    base32 = np.array(list("0123456789bcdefghjkmnpqrstuvwxyz"))

    # Normalize coordinates to [0, 1]
    # Clip to slightly less than max to avoid index overflow at exactly 90/180
    # Geohash bounds are [-90, 90] and [-180, 180]
    lat = np.clip(lat, -90.0, 90.0 - 1e-9)
    lon = np.clip(lon, -180.0, 180.0 - 1e-9)

    lat_norm = (lat + 90.0) / 180.0
    lon_norm = (lon + 180.0) / 360.0

    # Convert to high-precision integers (32-bit is sufficient for precision <= 7)
    # Precision 7 requires ~35 bits total (17 lat, 18 lon).
    # We map 0..1 to 0..2^32.
    scale = 1 << 32
    lat_int = (lat_norm * scale).astype(np.uint64)
    lon_int = (lon_norm * scale).astype(np.uint64)

    # Bit interleaving
    # We need to extract bits from the MSB downwards.
    # Lon: bit 31, Lat: bit 31, Lon: bit 30, Lat: bit 30...

    result_chars = []

    lon_bit_idx = 31
    lat_bit_idx = 31
    is_even = True  # Start with Longitude

    for _ in range(precision):
        chunk_val = np.zeros(lat.shape, dtype=np.uint8)

        # Construct 5-bit chunk
        for bit_pos in range(4, -1, -1):
            if is_even:
                # Extract bit from Lon
                bit = (lon_int >> lon_bit_idx) & 1
                lon_bit_idx -= 1
            else:
                # Extract bit from Lat
                bit = (lat_int >> lat_bit_idx) & 1
                lat_bit_idx -= 1

            chunk_val |= bit << bit_pos
            is_even = not is_even

        # Map to character
        result_chars.append(base32[chunk_val])

    # Combine characters
    if precision == 0:
        return np.full(lat.shape, "")

    result = result_chars[0]
    for i in range(1, precision):
        result = np.char.add(result, result_chars[i])

    return result
