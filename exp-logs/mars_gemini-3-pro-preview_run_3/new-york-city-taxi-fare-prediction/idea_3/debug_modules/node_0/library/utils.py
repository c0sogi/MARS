import numpy as np
import pandas as pd
from library.config import LANDMARKS, ROTATION_ANGLE


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculates the great circle distance between two points
    on the earth (specified in decimal degrees).

    Args:
        lat1, lon1: Latitude and Longitude of starting point (float or array-like).
        lat2, lon2: Latitude and Longitude of ending point (float or array-like).

    Returns:
        Distance in kilometers (float or array-like).
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2

    # Use arctan2 for better numerical stability than arcsin
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    # Radius of earth in kilometers
    R = 6371.0
    return R * c


def rotate_coordinates(df, angle=ROTATION_ANGLE):
    """
    Rotates the coordinates by a specific angle to align with the NYC street grid.
    Adds rotated columns to the DataFrame.

    Args:
        df (pd.DataFrame): Input dataframe containing 'pickup_longitude', 'pickup_latitude',
                           'dropoff_longitude', and 'dropoff_latitude'.
        angle (float): Rotation angle in degrees. Defaults to config.ROTATION_ANGLE.

    Returns:
        pd.DataFrame: DataFrame with added columns:
                      'pickup_x_rot', 'pickup_y_rot', 'dropoff_x_rot', 'dropoff_y_rot'.
    """
    # Convert angle to radians
    theta = np.radians(angle)

    # Precompute rotation matrix components
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # Apply rotation to pickup coordinates
    # Treating Longitude as X and Latitude as Y
    df["pickup_x_rot"] = (
        df["pickup_longitude"] * cos_theta - df["pickup_latitude"] * sin_theta
    )
    df["pickup_y_rot"] = (
        df["pickup_longitude"] * sin_theta + df["pickup_latitude"] * cos_theta
    )

    # Apply rotation to dropoff coordinates
    df["dropoff_x_rot"] = (
        df["dropoff_longitude"] * cos_theta - df["dropoff_latitude"] * sin_theta
    )
    df["dropoff_y_rot"] = (
        df["dropoff_longitude"] * sin_theta + df["dropoff_latitude"] * cos_theta
    )

    return df


def add_landmark_features(df, landmarks=LANDMARKS):
    """
    Calculates distances from pickup and dropoff locations to specific landmarks.

    Args:
        df (pd.DataFrame): Input dataframe with lat/lon columns.
        landmarks (dict): Dictionary of landmarks {name: (lat, lon)}.
                          Defaults to config.LANDMARKS.

    Returns:
        pd.DataFrame: DataFrame with added distance features.
    """
    for name, (lat, lon) in landmarks.items():
        # Calculate distance from pickup to landmark
        df[f"dist_pickup_{name}"] = haversine_distance(
            df["pickup_latitude"], df["pickup_longitude"], lat, lon
        )

        # Calculate distance from dropoff to landmark
        df[f"dist_dropoff_{name}"] = haversine_distance(
            df["dropoff_latitude"], df["dropoff_longitude"], lat, lon
        )

    return df
