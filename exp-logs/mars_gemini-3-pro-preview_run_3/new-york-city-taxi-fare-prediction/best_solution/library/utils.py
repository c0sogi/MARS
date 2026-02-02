import numpy as np
import pandas as pd
from library.config import R_EARTH


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculates the great circle distance between two points
    on the earth (specified in decimal degrees).

    Args:
        lat1, lon1: Latitude and Longitude of starting point(s).
        lat2, lon2: Latitude and Longitude of ending point(s).

    Returns:
        Distance in kilometers.
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    d = R_EARTH * c
    return d


def rotate_coordinates(lat, lon, angle_rad):
    """
    Rotates coordinates by a specific angle (in radians).
    Longitude is treated as X, Latitude as Y.

    Args:
        lat: Latitude (Y)
        lon: Longitude (X)
        angle_rad: Rotation angle in radians

    Returns:
        Tuple of (rotated_latitude, rotated_longitude)
    """
    # x' = x cos(theta) - y sin(theta)
    # y' = x sin(theta) + y cos(theta)
    lon_rot = lon * np.cos(angle_rad) - lat * np.sin(angle_rad)
    lat_rot = lon * np.sin(angle_rad) + lat * np.cos(angle_rad)
    return lat_rot, lon_rot


def manhattan_distance(lat1, lon1, lat2, lon2):
    """
    Calculates the Manhattan (L1) distance between two points.

    Args:
        lat1, lon1: First point coordinates.
        lat2, lon2: Second point coordinates.

    Returns:
        L1 distance (|lat1-lat2| + |lon1-lon2|).
    """
    return np.abs(lat1 - lat2) + np.abs(lon1 - lon2)


def reduce_mem_usage(df):
    """
    Iterate through all the columns of a dataframe and modify the data type
    to reduce memory usage.

    Args:
        df: Pandas DataFrame.

    Returns:
        Pandas DataFrame with optimized memory usage.
    """
    for col in df.columns:
        col_type = df[col].dtype

        if (
            col_type != object
            and col_type.name != "category"
            and "datetime" not in col_type.name
        ):
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    return df
