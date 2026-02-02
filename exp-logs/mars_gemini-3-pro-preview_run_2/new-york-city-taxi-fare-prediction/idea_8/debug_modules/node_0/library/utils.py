import numpy as np
import pandas as pd
from library.config import BBOX


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculates the Haversine distance between two points on the earth.

    The Haversine formula determines the great-circle distance between two points
    on a sphere given their longitudes and latitudes.

    Args:
        lat1 (float or np.array): Latitude of the first point(s).
        lon1 (float or np.array): Longitude of the first point(s).
        lat2 (float or np.array): Latitude of the second point(s).
        lon2 (float or np.array): Longitude of the second point(s).

    Returns:
        float or np.array: The distance between the two points in kilometers.
    """
    R = 6371.0  # Earth radius in kilometers

    # Convert degrees to radians
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    # Haversine formula
    a = (
        np.sin(delta_phi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    )

    # Calculate the central angle
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    # Calculate distance
    d = R * c
    return d


def manhattan_distance(lat1, lon1, lat2, lon2):
    """
    Calculates the Manhattan distance (L1 norm) between two points using their coordinates.

    This acts as a proxy for street-grid distance in cities like NYC.

    Args:
        lat1 (float or np.array): Latitude of the first point(s).
        lon1 (float or np.array): Longitude of the first point(s).
        lat2 (float or np.array): Latitude of the second point(s).
        lon2 (float or np.array): Longitude of the second point(s).

    Returns:
        float or np.array: The L1 distance (sum of absolute differences in degrees).
    """
    return np.abs(lat1 - lat2) + np.abs(lon1 - lon2)


def filter_within_bbox(df, bbox=None):
    """
    Filters the DataFrame to keep only rows where pickup and dropoff locations
    are within the specified bounding box.

    This function sanitizes the data by removing outliers and invalid GPS coordinates
    that fall outside the area of interest (e.g., NYC and surrounding airports).

    Args:
        df (pd.DataFrame): Input dataframe containing 'pickup_longitude', 'pickup_latitude',
                           'dropoff_longitude', and 'dropoff_latitude' columns.
        bbox (dict, optional): Dictionary with keys 'min_long', 'max_long',
                               'min_lat', 'max_lat'. Defaults to library.config.BBOX.

    Returns:
        pd.DataFrame: A new DataFrame containing only the rows within the bounding box.
    """
    if bbox is None:
        bbox = BBOX

    mask = (
        (df["pickup_longitude"] >= bbox["min_long"])
        & (df["pickup_longitude"] <= bbox["max_long"])
        & (df["pickup_latitude"] >= bbox["min_lat"])
        & (df["pickup_latitude"] <= bbox["max_lat"])
        & (df["dropoff_longitude"] >= bbox["min_long"])
        & (df["dropoff_longitude"] <= bbox["max_long"])
        & (df["dropoff_latitude"] >= bbox["min_lat"])
        & (df["dropoff_latitude"] <= bbox["max_lat"])
    )

    return df[mask].copy()
