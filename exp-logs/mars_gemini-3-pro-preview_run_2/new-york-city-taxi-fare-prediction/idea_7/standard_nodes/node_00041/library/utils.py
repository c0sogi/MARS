import numpy as np
import pandas as pd
from library.config import NYC_BOUNDING_BOX, haversine_distance


def clamp_coordinates(df, bounding_box=NYC_BOUNDING_BOX):
    """
    Restricts the pickup and dropoff coordinates to the specified bounding box.

    Args:
        df (pd.DataFrame): Input dataframe containing coordinate columns.
        bounding_box (dict): Dictionary with min/max lat/lon keys.

    Returns:
        pd.DataFrame: Dataframe with clamped coordinates.
    """
    # Operates on a copy to prevent side effects
    df = df.copy()

    df["pickup_longitude"] = df["pickup_longitude"].clip(
        bounding_box["min_lon"], bounding_box["max_lon"]
    )
    df["pickup_latitude"] = df["pickup_latitude"].clip(
        bounding_box["min_lat"], bounding_box["max_lat"]
    )
    df["dropoff_longitude"] = df["dropoff_longitude"].clip(
        bounding_box["min_lon"], bounding_box["max_lon"]
    )
    df["dropoff_latitude"] = df["dropoff_latitude"].clip(
        bounding_box["min_lat"], bounding_box["max_lat"]
    )

    return df


def haversine_array(lat1, lon1, lat2, lon2):
    """
    Calculates Haversine distance between two sets of coordinates.
    Wraps the implementation from config.py to ensure consistency.

    Args:
        lat1, lon1, lat2, lon2: Arrays or Series of coordinates.

    Returns:
        np.array: Haversine distance in km.
    """
    return haversine_distance(lat1, lon1, lat2, lon2)


def manhattan_distance(lat1, lon1, lat2, lon2):
    """
    Calculates Manhattan distance (L1 norm) between two sets of coordinates.

    Args:
        lat1, lon1, lat2, lon2: Arrays or Series of coordinates.

    Returns:
        np.array: Manhattan distance in degrees (sum of absolute differences).
    """
    return np.abs(lat1 - lat2) + np.abs(lon1 - lon2)


def rotate_coordinates(df, angle=45):
    """
    Adds rotated coordinate features to the DataFrame.
    Rotates the coordinate system by the specified angle (in degrees).

    Args:
        df (pd.DataFrame): Input dataframe.
        angle (float): Angle of rotation in degrees.

    Returns:
        pd.DataFrame: Dataframe with added '_rot' coordinate columns.
    """
    df = df.copy()

    # Convert angle to radians
    theta = np.radians(angle)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # Rotation matrix application
    # x' = x cos(theta) - y sin(theta)
    # y' = x sin(theta) + y cos(theta)
    # We map Longitude -> x, Latitude -> y

    df["pickup_lon_rot"] = (
        df["pickup_longitude"] * cos_theta - df["pickup_latitude"] * sin_theta
    )
    df["pickup_lat_rot"] = (
        df["pickup_longitude"] * sin_theta + df["pickup_latitude"] * cos_theta
    )

    df["dropoff_lon_rot"] = (
        df["dropoff_longitude"] * cos_theta - df["dropoff_latitude"] * sin_theta
    )
    df["dropoff_lat_rot"] = (
        df["dropoff_longitude"] * sin_theta + df["dropoff_latitude"] * cos_theta
    )

    return df
