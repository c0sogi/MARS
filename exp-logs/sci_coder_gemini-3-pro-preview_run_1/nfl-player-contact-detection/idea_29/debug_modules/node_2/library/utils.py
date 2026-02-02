import os
import logging
import hashlib
import numpy as np
import random
import sys
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for python, numpy, and other relevant libraries
    to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def setup_logging(log_file=None, level=logging.INFO):
    """
    Configures the logging module to output to both console and a file.
    """
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        # Ensure directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers,
        force=True,  # Reset any existing config
    )


def get_file_hash(filepath, block_size=65536):
    """
    Computes the MD5 hash of a file to detect changes.
    Used for parameter-aware caching.
    """
    if not os.path.exists(filepath):
        return None

    file_hash = hashlib.md5()
    with open(filepath, "rb") as f:
        fb = f.read(block_size)
        while len(fb) > 0:
            file_hash.update(fb)
            fb = f.read(block_size)

    return file_hash.hexdigest()


def calculate_quadratic_min_distance(
    x1, y1, sx1, sy1, ax1, ay1, x2, y2, sx2, sy2, ax2, ay2, time_window=1.5
):
    """
    Vectorized calculation of the minimum distance between two objects
    within a future time window using a quadratic approximation of the
    distance function: d(t) = d0 + v_rel*t + 0.5*a_rel*t^2.

    This implements the "Relaxed Quadratic Gating" logic.

    Args:
        x1, y1: Position arrays for entity 1
        sx1, sy1: Speed/Velocity components for entity 1 (yards/s)
        ax1, ay1: Acceleration components for entity 1 (yards/s^2)
        x2, y2: Position arrays for entity 2
        sx2, sy2: Speed/Velocity components for entity 2
        ax2, ay2: Acceleration components for entity 2
        time_window: Maximum time into the future to check (seconds)

    Returns:
        min_dist: Array of minimum predicted distances.
    """
    # 1. Relative Vectors (Entity 1 relative to Entity 2)
    rx = x1 - x2
    ry = y1 - y2
    vx = sx1 - sx2
    vy = sy1 - sy2
    ax_rel = ax1 - ax2
    ay_rel = ay1 - ay2

    # 2. Current Distance (d0)
    d2 = rx**2 + ry**2
    d0 = np.sqrt(d2)

    # Handle division by zero for d0 (if objects overlap perfectly)
    epsilon = 1e-6
    d0_safe = np.where(d0 < epsilon, epsilon, d0)

    # 3. First Derivative of Distance (d_dot) at t=0
    # d_dot = (r . v) / d
    r_dot_v = rx * vx + ry * vy
    d_dot = r_dot_v / d0_safe

    # 4. Second Derivative of Distance (d_ddot) at t=0
    # d_ddot = (v^2 + r.a - d_dot^2) / d
    v2 = vx**2 + vy**2
    r_dot_a = rx * ax_rel + ry * ay_rel
    d_ddot = (v2 + r_dot_a - d_dot**2) / d0_safe

    # 5. Find critical point t* where derivative of quadratic approx is 0
    # Approximation: D(t) ~ d0 + d_dot * t + 0.5 * d_ddot * t^2
    # D'(t) = d_dot + d_ddot * t = 0  => t* = -d_dot / d_ddot

    t_star = np.zeros_like(d0)
    valid_accel = np.abs(d_ddot) > epsilon

    # Where acceleration is significant
    # If d_ddot != 0, vertex is at -d_dot/d_ddot
    t_star[valid_accel] = -d_dot[valid_accel] / d_ddot[valid_accel]

    # Where acceleration is negligible (linear motion)
    # If closing (d_dot < 0), min is at window end. If opening, min is at 0.
    linear_closing = (~valid_accel) & (d_dot < 0)
    t_star[linear_closing] = time_window

    # 6. Clip t* to window [0, time_window]
    t_star = np.clip(t_star, 0, time_window)

    # 7. Evaluate quadratic at t*
    # D_min = d0 + d_dot * t* + 0.5 * d_ddot * (t*)^2
    min_dist = d0 + d_dot * t_star + 0.5 * d_ddot * (t_star**2)

    # Safety: Distance cannot be negative
    min_dist = np.maximum(min_dist, 0.0)

    # 8. Handle Concavity (d_ddot < 0)
    # If concave down, the vertex is a maximum. Minimum is at boundaries.
    concave = d_ddot < -epsilon
    if np.any(concave):
        dist_at_0 = d0
        dist_at_window = d0 + d_dot * time_window + 0.5 * d_ddot * (time_window**2)
        dist_at_window = np.maximum(dist_at_window, 0.0)

        min_dist_concave = np.minimum(dist_at_0, dist_at_window)
        min_dist = np.where(concave, min_dist_concave, min_dist)

    return min_dist


def project_vector(u_x, u_y, v_x, v_y):
    """
    Projects vector u onto basis vector v.
    Returns the scalar component (signed magnitude).

    Args:
        u_x, u_y: Components of vector to project
        v_x, v_y: Components of basis vector

    Returns:
        scalar_proj: The scalar projection (u . v_hat)
    """
    # Normalize v
    v_norm = np.sqrt(v_x**2 + v_y**2)
    epsilon = 1e-6
    v_norm = np.where(v_norm < epsilon, epsilon, v_norm)

    v_hat_x = v_x / v_norm
    v_hat_y = v_y / v_norm

    # Dot product
    scalar_proj = u_x * v_hat_x + u_y * v_hat_y

    return scalar_proj
