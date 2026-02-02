import numpy as np
import pandas as pd


def ecef_to_geodetic(x, y, z):
    """
    Convert Earth-Centered, Earth-Fixed (ECEF) coordinates to Geodetic coordinates
    (Latitude, Longitude, Altitude) using the WGS84 ellipsoid.

    This function uses a vectorized iterative method to compute latitude and altitude.

    Args:
        x (np.array): ECEF X coordinate in meters.
        y (np.array): ECEF Y coordinate in meters.
        z (np.array): ECEF Z coordinate in meters.

    Returns:
        tuple: (lat, lon, alt)
            lat (np.array): Latitude in degrees.
            lon (np.array): Longitude in degrees.
            alt (np.array): Altitude above the WGS84 ellipsoid in meters.
    """
    # WGS84 ellipsoid constants
    a = 6378137.0  # Semi-major axis
    f = 1 / 298.257223563  # Flattening
    b = a * (1 - f)  # Semi-minor axis
    e2 = 2 * f - f**2  # Square of first eccentricity

    # Distance from Z-axis
    r = np.sqrt(x**2 + y**2)

    # Initial guess for latitude
    lat = np.arctan2(z, r)

    # Iteratively update latitude and altitude
    # 5 iterations are generally sufficient for high precision
    h = 0
    for _ in range(5):
        sin_lat = np.sin(lat)
        N = a / np.sqrt(1 - e2 * sin_lat**2)
        h = r / np.cos(lat) - N
        lat = np.arctan2(z, r * (1 - e2 * (N / (N + h))))

    lon = np.arctan2(y, x)

    return np.degrees(lat), np.degrees(lon), h


def haversine_loss(pred_lat, pred_lon, true_lat, true_lon):
    """
    Calculate the Great Circle distance (Haversine distance) between two points
    on the earth (specified in decimal degrees).

    Args:
        pred_lat (np.array): Predicted latitude in degrees.
        pred_lon (np.array): Predicted longitude in degrees.
        true_lat (np.array): Ground truth latitude in degrees.
        true_lon (np.array): Ground truth longitude in degrees.

    Returns:
        np.array: Distance in meters.
    """
    R = 6371000.0  # Earth radius in meters

    # Convert degrees to radians
    phi1 = np.radians(pred_lat)
    phi2 = np.radians(true_lat)
    dphi = np.radians(true_lat - pred_lat)
    dlambda = np.radians(true_lon - pred_lon)

    # Haversine formula
    a = (
        np.sin(dphi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    )

    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance = R * c
    return distance


def calc_score(df_pred, df_gt):
    """
    Calculate the competition metric: the mean of the 50th and 95th percentile
    distance errors, averaged across all phones (tripIds).

    Args:
        df_pred (pd.DataFrame): DataFrame containing predictions with columns
                                ['tripId', 'UnixTimeMillis', 'LatitudeDegrees', 'LongitudeDegrees'].
        df_gt (pd.DataFrame): DataFrame containing ground truth with columns
                              ['tripId', 'UnixTimeMillis', 'LatitudeDegrees', 'LongitudeDegrees'].

    Returns:
        float: The calculated score (lower is better).
    """
    # Ensure inputs are sorted or aligned, but merge is safer
    # Rename columns to avoid suffixes collision if names are identical
    pred_subset = df_pred[
        ["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]
    ].copy()
    gt_subset = df_gt[
        ["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]
    ].copy()

    pred_subset.rename(
        columns={"LatitudeDegrees": "lat_pred", "LongitudeDegrees": "lon_pred"},
        inplace=True,
    )
    gt_subset.rename(
        columns={"LatitudeDegrees": "lat_gt", "LongitudeDegrees": "lon_gt"},
        inplace=True,
    )

    # Merge predictions and ground truth
    merged = pd.merge(
        pred_subset, gt_subset, on=["tripId", "UnixTimeMillis"], how="inner"
    )

    if len(merged) == 0:
        print(
            "Warning: No overlapping timestamps found between prediction and ground truth."
        )
        return np.nan

    # Calculate distance errors
    merged["error_dist"] = haversine_loss(
        merged["lat_pred"].values,
        merged["lon_pred"].values,
        merged["lat_gt"].values,
        merged["lon_gt"].values,
    )

    # Calculate metric per tripId
    trip_scores = []
    for trip_id, group in merged.groupby("tripId"):
        errors = group["error_dist"].values
        p50 = np.percentile(errors, 50)
        p95 = np.percentile(errors, 95)
        trip_score = (p50 + p95) / 2.0
        trip_scores.append(trip_score)

    # Final score is the mean over all trips
    final_score = np.mean(trip_scores)
    return final_score
