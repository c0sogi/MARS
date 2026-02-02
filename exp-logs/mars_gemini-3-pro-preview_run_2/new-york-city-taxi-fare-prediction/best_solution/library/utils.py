import numpy as np
import pandas as pd
from library.config import NYC_BB


def clamp_coordinates(df):
    """
    Clamps the pickup and dropoff coordinates to the NYC bounding box defined in config.
    Modifies the DataFrame in-place to save memory, but returns it for chaining.

    Args:
        df (pd.DataFrame): Input dataframe containing coordinate columns.

    Returns:
        pd.DataFrame: The modified dataframe with clamped coordinates.
    """
    # Define columns and their corresponding limits
    limits = {
        "pickup_longitude": (NYC_BB["min_lon"], NYC_BB["max_lon"]),
        "pickup_latitude": (NYC_BB["min_lat"], NYC_BB["max_lat"]),
        "dropoff_longitude": (NYC_BB["min_lon"], NYC_BB["max_lon"]),
        "dropoff_latitude": (NYC_BB["min_lat"], NYC_BB["max_lat"]),
    }

    for col, (min_val, max_val) in limits.items():
        if col in df.columns:
            # efficient in-place clipping
            df[col] = df[col].clip(lower=min_val, upper=max_val)

    return df


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculates the Haversine distance between two sets of points in kilometers.
    Vectorized implementation using numpy.

    Args:
        lat1, lon1: First point coordinates (scalar or array-like)
        lat2, lon2: Second point coordinates (scalar or array-like)

    Returns:
        np.ndarray: Distance in kilometers.
    """
    R = 6371.0  # Earth radius in kilometers

    # Convert to radians
    phi1 = np.radians(np.asarray(lat1))
    phi2 = np.radians(np.asarray(lat2))
    delta_phi = np.radians(np.asarray(lat2) - np.asarray(lat1))
    delta_lambda = np.radians(np.asarray(lon2) - np.asarray(lon1))

    # Haversine formula
    a = (
        np.sin(delta_phi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    )

    # Numerical stability clip
    a = np.clip(a, 0.0, 1.0)

    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))

    return R * c


def manhattan_distance(lat1, lon1, lat2, lon2):
    """
    Calculates the Manhattan distance in kilometers.
    Approximates the L1 distance on the surface of the Earth.

    Args:
        lat1, lon1: First point coordinates (scalar or array-like)
        lat2, lon2: Second point coordinates (scalar or array-like)

    Returns:
        np.ndarray: Distance in kilometers.
    """
    lat1 = np.asarray(lat1)
    lon1 = np.asarray(lon1)
    lat2 = np.asarray(lat2)
    lon2 = np.asarray(lon2)

    # Average latitude for scaling longitude distance
    mean_lat_rad = np.radians((lat1 + lat2) / 2.0)

    # Conversion factors
    # 1 degree latitude is approx 111 km
    # 1 degree longitude is approx 111 * cos(lat) km
    km_per_deg_lat = 111.0

    lat_dist = np.abs(lat1 - lat2) * km_per_deg_lat
    lon_dist = np.abs(lon1 - lon2) * km_per_deg_lat * np.cos(mean_lat_rad)

    return lat_dist + lon_dist


def vectorized_geohash(lats, lons, precision):
    """
    Vectorized Geohash encoding using NumPy.
    Converts latitude and longitude arrays to geohash strings of a given precision.

    Args:
        lats (np.array or pd.Series): Latitudes
        lons (np.array or pd.Series): Longitudes
        precision (int): Number of characters in the geohash

    Returns:
        np.ndarray: Array of geohash strings.
    """
    # Ensure inputs are numpy arrays
    lats = np.asarray(lats, dtype=np.float64)
    lons = np.asarray(lons, dtype=np.float64)

    # Base32 map
    base32 = np.array(list("0123456789bcdefghjkmnpqrstuvwxyz"))

    # Initialize bounds for all points
    # We only track the min bounds and use a scalar width that decreases
    lat_min = np.full_like(lats, -90.0)
    lon_min = np.full_like(lons, -180.0)

    # Initial widths
    lat_width = 180.0
    lon_width = 360.0

    # List to store characters for each position
    hash_chars = []

    # Loop over each character position
    for i in range(precision):
        char_idx = np.zeros_like(lats, dtype=np.int32)

        # Each character represents 5 bits
        for j in range(5):
            # Determine if we are processing a Longitude (even) or Latitude (odd) bit
            # Global bit index: i * 5 + j
            is_even_bit = ((i * 5 + j) % 2) == 0

            if is_even_bit:
                mid = lon_min + lon_width / 2.0
                # If lon is in the upper half, bit is 1
                mask = lons >= mid
                # Update lower bound for points in upper half
                lon_min[mask] += lon_width / 2.0
                # Update width for the next bit (applies to all points)
                lon_width /= 2.0
            else:
                mid = lat_min + lat_width / 2.0
                # If lat is in the upper half, bit is 1
                mask = lats >= mid
                # Update lower bound for points in upper half
                lat_min[mask] += lat_width / 2.0
                # Update width for the next bit
                lat_width /= 2.0

            # Shift existing bits and add the new bit
            char_idx = (char_idx << 1) | mask.astype(np.int32)

        # Map indices to Base32 characters
        hash_chars.append(base32[char_idx])

    # Join characters to form final strings
    # Start with the first character array
    result = hash_chars[0]
    # Append subsequent characters
    for k in range(1, precision):
        result = np.char.add(result, hash_chars[k])

    return result
