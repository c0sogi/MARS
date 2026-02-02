import numpy as np
import torch
from library.config import Config


def spherical_to_cartesian(azimuth, zenith):
    """
    Convert spherical coordinates to Cartesian unit vector.

    Args:
        azimuth: float or array-like, angle in radians (0 to 2pi).
        zenith: float or array-like, angle in radians (0 to pi).

    Returns:
        x, y, z: Cartesian coordinates (unit vector).
    """
    sin_zenith = np.sin(zenith)
    x = np.cos(azimuth) * sin_zenith
    y = np.sin(azimuth) * sin_zenith
    z = np.cos(zenith)
    return x, y, z


def cartesian_to_spherical(x, y, z):
    """
    Convert Cartesian coordinates to spherical coordinates.

    Args:
        x, y, z: Cartesian coordinates.

    Returns:
        azimuth, zenith: Angles in radians.
    """
    # Compute radius (norm)
    r = np.sqrt(x**2 + y**2 + z**2)
    # Avoid division by zero
    r = np.where(r == 0, 1e-6, r)

    # Zenith: arccos(z / r). Clip to [-1, 1] to avoid numerical errors.
    z_clamped = np.clip(z / r, -1.0, 1.0)
    zenith = np.arccos(z_clamped)

    # Azimuth: arctan2(y, x). Result is in [-pi, pi].
    azimuth = np.arctan2(y, x)
    # Convert range to [0, 2pi]
    azimuth = np.where(azimuth < 0, azimuth + 2 * np.pi, azimuth)

    return azimuth, zenith


def get_rotation_matrix_from_vectors(u, v):
    """
    Compute the rotation matrix that rotates unit vector u to unit vector v.

    Args:
        u: Source vector (3,).
        v: Target vector (3,).

    Returns:
        R: (3, 3) Rotation matrix.
    """
    # Normalize vectors to ensure they are unit vectors
    u = u / (np.linalg.norm(u) + 1e-8)
    v = v / (np.linalg.norm(v) + 1e-8)

    cross = np.cross(u, v)
    dot = np.dot(u, v)
    norm_cross = np.linalg.norm(cross)

    # Case 1: Vectors are parallel
    if norm_cross < 1e-6:
        if dot > 0:
            # Identical direction
            return np.eye(3)
        else:
            # Opposite direction: 180 degree rotation
            # Find an arbitrary orthogonal vector axis
            if np.abs(u[0]) < 0.9:
                ortho = np.cross(u, np.array([1, 0, 0]))
            else:
                ortho = np.cross(u, np.array([0, 1, 0]))
            ortho = ortho / np.linalg.norm(ortho)

            # Construct rotation matrix for 180 deg around 'ortho'
            # R = 2 * (n outer n) - I
            return 2 * np.outer(ortho, ortho) - np.eye(3)

    # Case 2: General case using Rodrigues' rotation formula
    # R = I + [k]x + [k]x^2 * (1 / (1 + dot))
    vx = np.array(
        [[0, -cross[2], cross[1]], [cross[2], 0, -cross[0]], [-cross[1], cross[0], 0]]
    )

    R = np.eye(3) + vx + (vx @ vx) * (1.0 / (1.0 + dot))
    return R


def compute_canonical_rotation(pos, time, charge):
    """
    Compute the rotation matrix to align the event's principal axis with the Z-axis.
    Also returns the weighted center of the event for translation.

    Args:
        pos: (N, 3) numpy array of pulse positions.
        time: (N,) numpy array of pulse times.
        charge: (N,) numpy array of pulse charges.

    Returns:
        R: (3, 3) Rotation matrix.
        center: (3,) Weighted center position.
    """
    # 1. Compute Weighted Center
    total_charge = np.sum(charge)
    if total_charge <= 1e-6:
        # Fallback for empty or zero-charge events
        center = np.mean(pos, axis=0)
        weights = np.ones_like(charge) / len(charge)
    else:
        weights = charge / total_charge
        center = np.sum(pos * weights[:, np.newaxis], axis=0)

    # 2. Center the positions
    pos_centered = pos - center

    # 3. Compute Weighted Covariance Matrix
    # We weight the centered positions by sqrt(weights) so that X.T @ X gives the weighted covariance
    weighted_pos = pos_centered * np.sqrt(weights[:, np.newaxis])
    covariance = weighted_pos.T @ weighted_pos

    # 4. SVD / Eigendecomposition to find Principal Axis
    try:
        # eigh returns eigenvalues in ascending order
        evals, evecs = np.linalg.eigh(covariance)
        # The principal axis corresponds to the largest eigenvalue (last column)
        principal_axis = evecs[:, -1]
    except np.linalg.LinAlgError:
        # Fallback if SVD fails
        return np.eye(3), center

    # 5. Orient Axis with Time Flow
    # Project positions onto the principal axis
    projections = pos_centered @ principal_axis

    # Check covariance between projection and time
    if len(time) > 1:
        time_centered = time - np.mean(time)
        proj_centered = projections - np.mean(projections)
        # We only need the sign of the covariance
        cov = np.sum(time_centered * proj_centered)

        # If covariance is negative, the axis points opposite to time flow
        if cov < 0:
            principal_axis = -principal_axis

    # 6. Compute Rotation to align Principal Axis with Global Z (0, 0, 1)
    target_axis = np.array([0.0, 0.0, 1.0])
    R = get_rotation_matrix_from_vectors(principal_axis, target_axis)

    return R, center


def apply_rotation(points, rotation_matrix, center=None, inverse=False):
    """
    Apply rotation (and optional centering) to points or vectors.

    Args:
        points: (N, 3) or (B, N, 3) array of points.
        rotation_matrix: (3, 3) or (B, 3, 3) rotation matrix.
        center: (3,) or (B, 3) array, optional center to subtract (only in forward mode).
        inverse: bool, if True applies the inverse rotation (R^T).

    Returns:
        transformed_points: Array of same shape as input.
    """
    is_batched = points.ndim == 3

    if not inverse:
        # Forward Mode: Translate -> Rotate
        # p' = R * (p - center)
        # Since points are row vectors: p' = (p - center) @ R.T

        if center is not None:
            if is_batched:
                # Broadcast center: (B, 3) -> (B, 1, 3)
                points = points - center[:, np.newaxis, :]
            else:
                points = points - center

        if is_batched:
            # Batched Matmul: (B, N, 3) @ (B, 3, 3)^T
            # transpose(0, 2, 1) swaps the last two dimensions
            return points @ rotation_matrix.transpose(0, 2, 1)
        else:
            return points @ rotation_matrix.T

    else:
        # Inverse Mode: Rotate Back
        # p = R^T * p'
        # Since points are row vectors: p = p' @ R

        if is_batched:
            return points @ rotation_matrix
        else:
            return points @ rotation_matrix
