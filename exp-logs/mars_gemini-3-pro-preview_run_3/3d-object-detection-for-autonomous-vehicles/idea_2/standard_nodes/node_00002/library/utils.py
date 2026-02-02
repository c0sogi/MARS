import os
import numpy as np
import cv2
from library.config import Config


def load_lidar(path):
    """
    Loads binary lidar data from the given path.
    Supports 4-feature (x, y, z, intensity) and 5-feature formats.

    Args:
        path (str): Relative path to the .bin file (e.g., 'train_lidar/file.bin').

    Returns:
        np.ndarray: A (N, 4) numpy array containing x, y, z, intensity.
    """
    # Resolve full path
    full_path = os.path.join(Config.INPUT_DIR, path)

    if not os.path.exists(full_path):
        # Check if path was already absolute or relative to CWD
        if os.path.exists(path):
            full_path = path
        else:
            # Return empty array if file missing to prevent pipeline crash,
            # though usually this should raise an error.
            return np.zeros((0, 4), dtype=np.float32)

    try:
        # Load binary data
        raw_data = np.fromfile(full_path, dtype=np.float32)

        # Reshape based on feature stride
        # Common formats are (x, y, z, i) or (x, y, z, i, r)
        if len(raw_data) % 5 == 0:
            points = raw_data.reshape(-1, 5)
            # Keep only first 4 features (x, y, z, intensity)
            points = points[:, :4]
        elif len(raw_data) % 4 == 0:
            points = raw_data.reshape(-1, 4)
        else:
            # Fallback: Attempt to reshape to 4, truncating extra bytes if necessary
            # This handles potential minor corruptions or non-standard formats
            num_points = len(raw_data) // 4
            points = raw_data[: num_points * 4].reshape(-1, 4)

        return points

    except Exception as e:
        # In production, logging this error is crucial
        return np.zeros((0, 4), dtype=np.float32)


def box_to_corners(box):
    """
    Converts a box parameter vector to 8 corner coordinates in world frame.

    Args:
        box (list or np.array): [center_x, center_y, center_z, width, length, height, yaw]

    Returns:
        np.ndarray: (8, 3) array of corner coordinates.
    """
    x, y, z, w, l, h, yaw = box[:7]

    # Pre-compute rotation
    c = np.cos(yaw)
    s = np.sin(yaw)

    # Define corners in local coordinate system
    # x corresponds to width (left/right)
    # y corresponds to length (forward/back)
    # z corresponds to height (up/down)
    # Corners order: top-front-left, top-front-right, etc. (arbitrary as long as consistent)

    # Half dimensions
    dx = w / 2
    dy = l / 2
    dz = h / 2

    # Local corners (x, y, z)
    # We generate all 8 combinations of +/- dx, +/- dy, +/- dz
    x_corners = np.array([dx, dx, -dx, -dx, dx, dx, -dx, -dx])
    y_corners = np.array([dy, -dy, -dy, dy, dy, -dy, -dy, dy])
    z_corners = np.array([dz, dz, dz, dz, -dz, -dz, -dz, -dz])

    # Rotate around Z-axis
    # x' = x*cos(yaw) - y*sin(yaw)
    # y' = x*sin(yaw) + y*cos(yaw)
    x_rot = x_corners * c - y_corners * s
    y_rot = x_corners * s + y_corners * c

    # Translate to world center
    corners_x = x_rot + x
    corners_y = y_rot + y
    corners_z = z_corners + z

    # Stack into (8, 3)
    corners = np.stack([corners_x, corners_y, corners_z], axis=1)

    return corners


def calc_iou_3d(box1, box2):
    """
    Calculates the 3D Intersection over Union (IoU) between two boxes.
    Uses the formula: (Area_BEV_Intersection * Height_Overlap) / Volume_Union.

    Args:
        box1 (list or np.array): [x, y, z, w, l, h, yaw]
        box2 (list or np.array): [x, y, z, w, l, h, yaw]

    Returns:
        float: IoU score between 0.0 and 1.0.
    """
    # Unpack parameters
    x1, y1, z1, w1, l1, h1, yaw1 = box1[:7]
    x2, y2, z2, w2, l2, h2, yaw2 = box2[:7]

    # 1. Calculate Height Intersection
    z1_min, z1_max = z1 - h1 / 2, z1 + h1 / 2
    z2_min, z2_max = z2 - h2 / 2, z2 + h2 / 2

    inter_h = max(0.0, min(z1_max, z2_max) - max(z1_min, z2_min))

    # Optimization: If heights don't overlap, IoU is 0
    if inter_h == 0:
        return 0.0

    # 2. Calculate BEV (Bird's Eye View) Intersection Area
    # Create RotatedRect structures for OpenCV
    # cv2.RotatedRect accepts ((center_x, center_y), (width, height), angle_degrees)
    # Note: OpenCV angle is degrees clockwise? Standard math is counter-clockwise.
    # However, intersectConvexConvex works on the resulting polygon vertices,
    # so as long as we extract vertices correctly, the internal angle convention cancels out.
    # We use np.degrees(yaw) assuming standard math notation (CCW from X).

    rect1 = ((float(x1), float(y1)), (float(w1), float(l1)), float(np.degrees(yaw1)))
    rect2 = ((float(x2), float(y2)), (float(w2), float(l2)), float(np.degrees(yaw2)))

    # Get the 4 corners of the rectangles on the ground plane
    try:
        box1_pts = cv2.boxPoints(rect1)
        box2_pts = cv2.boxPoints(rect2)
    except Exception:
        return 0.0

    # Calculate intersection polygon
    try:
        # intersectConvexConvex returns (area, intersection_polygon_points)
        # Note: The return signature can vary by OpenCV version.
        # We check the result type to be safe.
        res = cv2.intersectConvexConvex(box1_pts, box2_pts)

        # Parse result
        if isinstance(res, tuple) or isinstance(res, list):
            # Format: (area, points)
            inter_area = res[0]
            # Verify area validity (sometimes it returns area but points are invalid)
            if res[1] is None:
                inter_area = 0.0
        else:
            # Some versions might return just area? Unlikely for this function.
            # Fallback to 0 if format is unexpected
            inter_area = 0.0

    except Exception:
        inter_area = 0.0

    if inter_area <= 0:
        return 0.0

    # 3. Calculate Volumes and IoU
    inter_vol = inter_area * inter_h

    vol1 = w1 * l1 * h1
    vol2 = w2 * l2 * h2

    union_vol = vol1 + vol2 - inter_vol

    if union_vol <= 1e-6:
        return 0.0

    return inter_vol / union_vol
