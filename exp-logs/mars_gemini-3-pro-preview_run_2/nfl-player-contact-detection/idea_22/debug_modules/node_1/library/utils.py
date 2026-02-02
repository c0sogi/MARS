import numpy as np


def calculate_shortest_arc(angle_a, angle_b):
    """
    Computes the shortest angular difference between two angles in degrees.
    Handles the 0/360 discontinuity to ensure physical continuity in features.

    Formula: min(|a-b|, 360-|a-b|)

    Args:
        angle_a (float or np.ndarray): First angle(s) in degrees.
        angle_b (float or np.ndarray): Second angle(s) in degrees.

    Returns:
        float or np.ndarray: The shortest angular difference (always positive).
    """
    # Ensure inputs are treated as arrays for consistent behavior
    a = np.array(angle_a)
    b = np.array(angle_b)

    # Calculate absolute difference modulo 360
    diff = np.abs(a - b) % 360

    # Calculate shortest path (min of direct diff and wrap-around diff)
    shortest_arc = np.minimum(diff, 360 - diff)

    return shortest_arc


def normalize_coordinates(x, y, field_length=120.0, field_width=53.3):
    """
    Normalizes field coordinates to a [0, 1] range based on standard NFL field dimensions.

    Args:
        x (float or np.ndarray): x-coordinate (long axis).
        y (float or np.ndarray): y-coordinate (short axis).
        field_length (float): Length of field in yards. Default 120.
        field_width (float): Width of field in yards. Default 53.3.

    Returns:
        tuple: (normalized_x, normalized_y)
    """
    norm_x = x / field_length
    norm_y = y / field_width
    return norm_x, norm_y


def calculate_euclidean_distance(x1, y1, x2, y2):
    """
    Calculates Euclidean distance between two points (or arrays of points).

    Args:
        x1, y1: Coordinates of first point.
        x2, y2: Coordinates of second point.

    Returns:
        float or np.ndarray: Euclidean distance.
    """
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def calculate_log_distance(distance):
    """
    Calculates log(1 + distance) for numerical stability and better distribution features.

    Args:
        distance (float or np.ndarray): The Euclidean distance.

    Returns:
        float or np.ndarray: Log-transformed distance.
    """
    return np.log1p(distance)


def calculate_closing_speed(vx1, vy1, vx2, vy2, x1, y1, x2, y2):
    """
    Calculates the closing speed between two entities.
    Closing speed is the rate at which the distance between two objects is decreasing.

    Args:
        vx1, vy1: Velocity components of entity 1.
        vx2, vy2: Velocity components of entity 2.
        x1, y1: Position of entity 1.
        x2, y2: Position of entity 2.

    Returns:
        np.ndarray: Closing speed (positive means closing in, negative means moving apart).
    """
    # Relative velocity
    rel_vx = vx1 - vx2
    rel_vy = vy1 - vy2

    # Relative position vector
    rel_px = x2 - x1
    rel_py = y2 - y1

    # Distance
    dist = np.sqrt(rel_px**2 + rel_py**2)

    # Project relative velocity onto the unit vector pointing from 1 to 2
    # Unit vector u = (rel_px/dist, rel_py/dist)
    # Projection = rel_vx * u_x + rel_vy * u_y

    with np.errstate(divide="ignore", invalid="ignore"):
        u_x = rel_px / dist
        u_y = rel_py / dist

        # Handle cases where distance is 0 (fill NaNs with 0)
        u_x = np.nan_to_num(u_x, nan=0.0)
        u_y = np.nan_to_num(u_y, nan=0.0)

    closing_speed = rel_vx * u_x + rel_vy * u_y

    return closing_speed
