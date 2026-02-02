import numpy as np
import pandas as pd


def calculate_euclidean_distance(x1, y1, x2, y2):
    """
    Calculates the Euclidean distance between two sets of coordinates.

    Args:
        x1, y1: Coordinates of the first entity (scalar or pd.Series).
        x2, y2: Coordinates of the second entity (scalar or pd.Series).

    Returns:
        np.array or pd.Series: The Euclidean distance.
    """
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def calculate_closure_rate(distance_series, time_delta=0.1):
    """
    Calculates the closure rate (derivative of distance w.r.t time).
    Positive closure rate implies moving away, negative implies closing in.

    Args:
        distance_series (pd.Series): Series of distances (must be sorted by time).
        time_delta (float): Time step in seconds (default 0.1 for 10Hz data).

    Returns:
        pd.Series: Closure rate. First value is filled with 0.
    """
    # Calculate discrete difference
    diff = distance_series.diff()

    # Calculate rate
    rate = diff / time_delta

    # Fill the initial NaN resulting from diff() with 0
    return rate.fillna(0.0)


def project_vector_to_body_frame(df, x_col, y_col, orientation_col, result_prefix):
    """
    Projects a 2D vector (x, y) into the body frame (Surge, Sway) based on orientation.

    Assumes orientation is in degrees, where 0 degrees aligns with the Y-axis (North)
    and increases clockwise (standard NFL tracking data convention).

    Surge: Component in the direction of orientation (Forward/Backward).
    Sway: Component perpendicular to orientation (Lateral Left/Right).

    Args:
        df (pd.DataFrame): Input dataframe containing the vector and orientation columns.
        x_col (str): Column name for the X component of the vector (e.g., 'speed_x', 'acc_x').
        y_col (str): Column name for the Y component of the vector.
        orientation_col (str): Column name for the orientation in degrees.
        result_prefix (str): Prefix for the output columns (e.g., 'v' -> 'v_surge', 'v_sway').

    Returns:
        pd.DataFrame: A DataFrame containing two columns:
                      - {result_prefix}_surge
                      - {result_prefix}_sway
    """
    # Convert orientation degrees to radians
    # Fill NaNs in orientation with 0 to prevent propagation errors, though tracking data should be clean
    theta = np.radians(df[orientation_col].fillna(0))

    # Precompute sine and cosine
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    # Extract vector components, filling NaNs with 0
    vec_x = df[x_col].fillna(0)
    vec_y = df[y_col].fillna(0)

    # Project Vector
    # Assuming 0 deg = Y-axis (North), 90 deg = X-axis (East)
    # Forward Unit Vector (Surge Axis): (sin(theta), cos(theta))
    # Rightward Unit Vector (Sway Axis): (cos(theta), -sin(theta))

    # Surge = Dot Product(Vector, Forward_Unit)
    surge = vec_x * sin_theta + vec_y * cos_theta

    # Sway = Dot Product(Vector, Rightward_Unit)
    sway = vec_x * cos_theta - vec_y * sin_theta

    # Construct result DataFrame
    result = pd.DataFrame(
        {f"{result_prefix}_surge": surge, f"{result_prefix}_sway": sway}, index=df.index
    )

    return result


def create_temporal_lags(df, group_cols, feature_cols, lags):
    """
    Generates flattened temporal features using lag windows.
    Creates both past (t - lag) and future (t + lag) features for each specified lag.

    Args:
        df (pd.DataFrame): Input dataframe.
        group_cols (list): Columns to group by (e.g., ['game_play', 'nfl_player_id']) to ensure
                           shifts don't cross play/player boundaries.
        feature_cols (list): List of column names to generate lags for.
        lags (list): List of integer time steps to shift by (e.g., [1, 2, 4]).

    Returns:
        pd.DataFrame: DataFrame containing only the new lag columns, aligned with the input index.
    """
    lag_features = {}

    # Create a groupby object once
    grouped = df.groupby(group_cols)

    for col in feature_cols:
        for lag in lags:
            # Past Lag (t - lag)
            # Example: value at t-1 available at t
            lag_features[f"{col}_lag_{lag}"] = grouped[col].shift(lag)

            # Future Lag (t + lag)
            # Example: value at t+1 available at t (lookahead)
            # In offline processing, future context is valid.
            lag_features[f"{col}_lag_minus_{lag}"] = grouped[col].shift(-lag)

    # Concatenate all lag features into a single DataFrame
    # Using a dictionary for creation is generally faster than repeated concat/assignment
    return pd.DataFrame(lag_features, index=df.index)
