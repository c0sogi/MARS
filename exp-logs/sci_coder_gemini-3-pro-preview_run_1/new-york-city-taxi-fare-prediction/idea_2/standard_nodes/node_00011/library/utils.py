import numpy as np
import pandas as pd


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculates the great circle distance between two points
    on the earth (specified in decimal degrees).
    Returns distance in kilometers.

    Args:
        lat1, lon1: Start point latitude and longitude (float or array-like)
        lat2, lon2: End point latitude and longitude (float or array-like)

    Returns:
        Distance in kilometers (float or array-like)
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2

    # Clip to handle potential floating point errors slightly > 1
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    r = 6371.0  # Radius of earth in kilometers
    return c * r


def manhattan_distance(lat1, lon1, lat2, lon2):
    """
    Calculates the Manhattan distance between two points in kilometers.
    This approximates the distance traveled along a grid aligned with lines of latitude and longitude.

    Args:
        lat1, lon1: Start point latitude and longitude (float or array-like)
        lat2, lon2: End point latitude and longitude (float or array-like)

    Returns:
        Distance in kilometers (float or array-like)
    """
    R = 6371.0

    # Convert to radians
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)

    # Distance in Latitude direction (simple arc length)
    dlat = np.abs(lat2_rad - lat1_rad) * R

    # Distance in Longitude direction (arc length scaled by cos of avg latitude)
    avg_lat = (lat1_rad + lat2_rad) / 2.0
    dlon = np.abs(lon2_rad - lon1_rad) * R * np.cos(avg_lat)

    return dlat + dlon


def reduce_memory_usage(df):
    """
    Iterates through all the columns of a dataframe and modifies the data type
    to reduce memory usage.

    Args:
        df: Input pandas DataFrame

    Returns:
        DataFrame with reduced memory usage
    """
    # start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtype

        # Skip object, category, and datetime columns
        if (
            col_type != object
            and str(col_type) != "category"
            and not pd.api.types.is_datetime64_any_dtype(df[col])
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
                # Float types
                # Use float32 if values fit within range and precision requirements are met
                # For this dataset, float32 is generally sufficient for coords and fares
                if (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    # end_mem = df.memory_usage().sum() / 1024**2
    return df
