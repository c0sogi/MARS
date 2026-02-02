import numpy as np
import pandas as pd
from library.config import NYC_BBOX, EARTH_RADIUS_KM


def clamp_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clips the pickup and dropoff coordinates of the DataFrame to the
    NYC bounding box defined in library.config.NYC_BBOX.

    Modifies the DataFrame in-place but also returns it for chaining.

    Args:
        df (pd.DataFrame): DataFrame containing coordinate columns:
                           'pickup_longitude', 'pickup_latitude',
                           'dropoff_longitude', 'dropoff_latitude'.

    Returns:
        pd.DataFrame: The modified DataFrame with clamped coordinates.
    """
    min_lon, min_lat, max_lon, max_lat = NYC_BBOX

    # Clip Pickup Coordinates
    df["pickup_longitude"] = df["pickup_longitude"].clip(min_lon, max_lon)
    df["pickup_latitude"] = df["pickup_latitude"].clip(min_lat, max_lat)

    # Clip Dropoff Coordinates
    df["dropoff_longitude"] = df["dropoff_longitude"].clip(min_lon, max_lon)
    df["dropoff_latitude"] = df["dropoff_latitude"].clip(min_lat, max_lat)

    return df


def compute_haversine(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """
    Computes the Haversine distance (great-circle distance) between two points
    in kilometers using vectorized numpy operations.

    Args:
        lat1, lon1: Arrays of start coordinates (latitude, longitude).
        lat2, lon2: Arrays of end coordinates (latitude, longitude).

    Returns:
        np.ndarray: Array of distances in kilometers.
    """
    # Convert decimal degrees to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    # Haversine formula
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))

    return c * EARTH_RADIUS_KM


def compute_manhattan(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """
    Computes the Manhattan distance (L1 norm) between two points in degrees.
    This serves as a proxy for city-block travel distance.

    Args:
        lat1, lon1: Arrays of start coordinates.
        lat2, lon2: Arrays of end coordinates.

    Returns:
        np.ndarray: Array of Manhattan distances in degrees.
    """
    return np.abs(lat1 - lat2) + np.abs(lon1 - lon2)


def vectorized_geohash(
    lats: np.ndarray, lons: np.ndarray, precision: int
) -> np.ndarray:
    """
    Computes a unique integer spatial index (hash) for coordinates based on
    standard Geohash grid resolutions.

    This function is a vectorized alternative to string-based Geohashing.
    It maps coordinates to a grid cell index, which is functionally equivalent
    to a Geohash for grouping and aggregation purposes, but significantly
    faster and memory-efficient for large datasets.

    Grid Resolution (approximate):
    - Precision 5: ~2.4km x 2.4km
    - Precision 6: ~1.2km x 0.6km
    - Precision 7: ~152m x 152m

    Args:
        lats (np.ndarray): Array of latitudes.
        lons (np.ndarray): Array of longitudes.
        precision (int): Geohash precision level (number of characters).
                         Determines the grid granularity.

    Returns:
        np.ndarray: Array of unique integer identifiers (int64) for the grid cells.
    """
    # 1. Determine Bit Counts based on Precision
    # Standard Geohash: 5 bits per character.
    # Bits are interleaved (Lon, Lat, Lon, Lat...).
    # Lon gets the extra bit if total bits is odd.
    total_bits = precision * 5
    n_lon_bits = total_bits // 2 + (total_bits % 2)
    n_lat_bits = total_bits // 2

    # 2. Normalize Coordinates to [0, 1]
    # Latitude range: [-90, 90]
    # Longitude range: [-180, 180]
    # We clip input to ensure stability against float errors near boundaries
    lat_norm = (np.clip(lats, -90.0, 90.0) + 90.0) / 180.0
    lon_norm = (np.clip(lons, -180.0, 180.0) + 180.0) / 360.0

    # 3. Quantize to Grid Indices
    # Calculate the number of bins for each dimension
    n_lat_bins = 1 << n_lat_bits
    n_lon_bins = 1 << n_lon_bits

    # Compute indices (0 to n_bins - 1)
    # Use astype(int64) to prevent overflow during combination
    lat_idx = (lat_norm * n_lat_bins).astype(np.int64)
    lon_idx = (lon_norm * n_lon_bins).astype(np.int64)

    # Handle edge case where value is exactly max (e.g., lat=90.0)
    # This maps it to the last bin instead of n_bins
    lat_idx = np.clip(lat_idx, 0, n_lat_bins - 1)
    lon_idx = np.clip(lon_idx, 0, n_lon_bins - 1)

    # 4. Combine into Unique Integer ID
    # We pack the two indices into a single integer.
    # ID = lat_idx * (2^lon_bits) + lon_idx
    # This creates a unique key for every grid cell.
    spatial_id = (lat_idx << n_lon_bits) | lon_idx

    return spatial_id
