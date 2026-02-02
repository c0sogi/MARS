import numpy as np
import pandas as pd
from typing import Union, Tuple


def calculate_euclidean_distance(
    x1: Union[pd.Series, np.ndarray, float],
    y1: Union[pd.Series, np.ndarray, float],
    x2: Union[pd.Series, np.ndarray, float],
    y2: Union[pd.Series, np.ndarray, float],
) -> Union[pd.Series, np.ndarray, float]:
    """
    Calculates the Euclidean distance between two sets of coordinates.

    Args:
        x1, y1: Coordinates of the first entity.
        x2, y2: Coordinates of the second entity.

    Returns:
        Euclidean distance.
    """
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def calculate_closure_rate(
    distance_series: pd.Series, time_delta: float = 0.1
) -> pd.Series:
    """
    Calculates the closure rate (rate of decrease in distance) over time.
    Positive closure rate indicates entities are getting closer.

    Formula: - (d_t - d_{t-1}) / dt

    Args:
        distance_series: A pandas Series of distances, assumed to be sorted by time
                         and grouped by the entity pair.
        time_delta: The time step between observations (default 0.1s for 10Hz data).

    Returns:
        A pandas Series representing the closure rate. First value will be NaN (or 0 if filled).
    """
    # Calculate difference: dist_t - dist_{t-1}
    # If distance decreases, diff is negative.
    # Closure rate = -diff / dt
    # If distance decreases, closure rate is positive.
    diff = distance_series.diff()
    return -diff / time_delta


def project_ego_velocity(
    speed: Union[pd.Series, np.ndarray],
    direction: Union[pd.Series, np.ndarray],
    orientation: Union[pd.Series, np.ndarray],
) -> Tuple[Union[pd.Series, np.ndarray], Union[pd.Series, np.ndarray]]:
    """
    Decomposes the velocity vector into Ego-Centric components based on the player's orientation.

    Args:
        speed: Scalar speed of the player.
        direction: Angle of motion in degrees (0-360).
        orientation: Angle the player is facing in degrees (0-360).

    Returns:
        Tuple (v_surge, v_sway):
            v_surge: Velocity component in the direction of orientation (Forward/Backward).
            v_sway: Velocity component orthogonal to orientation (Left/Right).
    """
    # Convert degrees to radians
    # Note: NFL tracking data angles are clockwise from Y-axis (usually).
    # The relative angle (direction - orientation) is invariant to the coordinate system rotation
    # as long as both follow the same convention.

    theta_rad = np.radians(direction - orientation)

    # Surge is the projection onto the orientation vector (Cos)
    v_surge = speed * np.cos(theta_rad)

    # Sway is the projection onto the orthogonal vector (Sin)
    v_sway = speed * np.sin(theta_rad)

    return v_surge, v_sway


def calculate_derivatives(series: pd.Series, time_delta: float = 0.1) -> pd.Series:
    """
    Calculates the first time derivative of a series.
    Used to compute Acceleration from Velocity, or Jerk from Acceleration.

    Args:
        series: Input data series (e.g., speed, v_surge, acceleration).
        time_delta: Time step.

    Returns:
        Derivative series.
    """
    return series.diff() / time_delta


def calculate_iou_metrics(
    box1: Union[np.ndarray, pd.Series], box2: Union[np.ndarray, pd.Series]
) -> Union[np.ndarray, float]:
    """
    Calculates Intersection over Union (IoU) for two bounding boxes or arrays of boxes.

    Box Format: [left, width, top, height]

    Args:
        box1: First box or array of boxes (N, 4).
        box2: Second box or array of boxes (N, 4).

    Returns:
        IoU value(s). Returns 0 if no overlap.
    """
    # Handle input types (ensure they are accessible via indices)
    # If pandas series, we might need to unpack if they contain lists,
    # but typically this is called with columns like 'left1', 'width1', etc.
    # Here we assume the inputs are either 1D arrays (single box) or 2D arrays (N boxes).
    # For flexibility with the pipeline, let's assume inputs are unpacked coordinates if possible,
    # but the signature asks for box objects.
    # Let's support separate coordinate inputs or array inputs.
    # Given the complexity of pandas apply, vectorized coordinate inputs are usually better.
    # However, to strictly match the signature "box1, box2", we assume they are arrays/lists/series of shape (4,) or (N, 4).

    # Convert to numpy for consistent handling
    b1 = np.array(box1)
    b2 = np.array(box2)

    # Check if single box or batch
    if b1.ndim == 1:
        b1 = b1.reshape(1, -1)
        b2 = b2.reshape(1, -1)
        single_output = True
    else:
        single_output = False

    # Unpack coordinates: [left, width, top, height]
    # x_min, width, y_min, height
    b1_x1 = b1[:, 0]
    b1_w = b1[:, 1]
    b1_y1 = b1[:, 2]
    b1_h = b1[:, 3]
    b1_x2 = b1_x1 + b1_w
    b1_y2 = b1_y1 + b1_h

    b2_x1 = b2[:, 0]
    b2_w = b2[:, 1]
    b2_y1 = b2[:, 2]
    b2_h = b2[:, 3]
    b2_x2 = b2_x1 + b2_w
    b2_y2 = b2_y1 + b2_h

    # Intersection coordinates
    xi1 = np.maximum(b1_x1, b2_x1)
    yi1 = np.maximum(b1_y1, b2_y1)
    xi2 = np.minimum(b1_x2, b2_x2)
    yi2 = np.minimum(b1_y2, b2_y2)

    # Intersection Area
    inter_width = np.maximum(0, xi2 - xi1)
    inter_height = np.maximum(0, yi2 - yi1)
    inter_area = inter_width * inter_height

    # Union Area
    b1_area = b1_w * b1_h
    b2_area = b2_w * b2_h
    union_area = b1_area + b2_area - inter_area

    # Avoid division by zero
    iou = np.divide(
        inter_area, union_area, out=np.zeros_like(inter_area), where=union_area > 0
    )

    if single_output:
        return iou[0]
    return iou
