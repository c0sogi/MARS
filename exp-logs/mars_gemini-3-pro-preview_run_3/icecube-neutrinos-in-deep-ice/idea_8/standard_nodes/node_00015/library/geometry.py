import numpy as np
import pandas as pd
from library.config import Config


def load_sensor_geometry(path=None):
    """
    Loads sensor geometry and returns a dict mapping sensor_id to [x, y, z].

    Args:
        path (str, optional): Path to the sensor geometry CSV file.
                              Defaults to Config.SENSOR_GEO_PATH.

    Returns:
        dict: Mapping from sensor_id (int) to coordinates (np.array of shape (3,)).
    """
    if path is None:
        path = Config.SENSOR_GEO_PATH

    # Use pandas to read the CSV for efficiency
    df = pd.read_csv(path)

    # Identify sensor_id column or use index if not present
    if "sensor_id" in df.columns:
        sensor_ids = df["sensor_id"].values
    else:
        sensor_ids = df.index.values

    # Extract coordinates (x, y, z)
    coords = df[["x", "y", "z"]].values.astype(np.float32)

    # Create dictionary mapping
    return dict(zip(sensor_ids, coords))


def compute_canonical_rotation(xyz, charge, time):
    """
    Computes the rotation matrix and center of gravity to align the event's principal axis.

    The alignment logic follows the SVD approach: the principal axis (direction of
    largest variance) is aligned with the first dimension (X-axis) of the transformed
    coordinate system. The direction along this axis is corrected to ensure positive
    correlation with time (particle flow).

    Args:
        xyz (np.ndarray): Pulse coordinates of shape (N, 3).
        charge (np.ndarray): Pulse charges of shape (N,).
        time (np.ndarray): Pulse times of shape (N,).

    Returns:
        tuple: (rotation_matrix, cog)
            rotation_matrix (np.ndarray): Orthogonal matrix of shape (3, 3).
            cog (np.ndarray): Center of gravity vector of shape (3,).
    """
    # 1. Compute Center of Gravity (COG)
    # Add epsilon to avoid division by zero
    total_charge = charge.sum() + 1e-6
    weights = charge / total_charge
    cog = np.sum(xyz * weights[:, None], axis=0)

    # Center the coordinates around the COG
    xyz_centered = xyz - cog

    # 2. Compute Weighted Covariance Matrix
    # We weight the centered coordinates by sqrt(weights) so that
    # (X_w)^T @ X_w equals the weighted covariance matrix
    weighted_xyz = xyz_centered * np.sqrt(weights[:, None])
    cov = weighted_xyz.T @ weighted_xyz

    # 3. Perform Singular Value Decomposition (SVD)
    try:
        # U contains the eigenvectors (principal axes) as columns
        U, S, Vh = np.linalg.svd(cov)
    except np.linalg.LinAlgError:
        # Fallback: Return Identity rotation and computed COG if SVD fails
        return np.eye(3, dtype=np.float32), cog

    # 4. Orientation Correction
    # The SVD determines the axis line but not the direction (sign).
    # We align the principal axis (first column of U) with the flow of time.

    # Project centered positions onto the principal axis (U[:, 0])
    axis = U[:, 0]
    projections = xyz_centered @ axis

    # Check correlation between position along axis and time
    # Ensure sufficient variance to avoid numerical instability
    if np.std(projections) > 1e-6 and np.std(time) > 1e-6:
        corr = np.corrcoef(projections, time)[0, 1]

        # If correlation is negative, the axis points opposite to time flow
        if corr < 0:
            # Flip the principal axis (and the corresponding column in U)
            U[:, 0] = -U[:, 0]

    return U, cog


def apply_rotation(xyz, rotation_matrix, cog):
    """
    Applies the canonical rotation transformation to a set of coordinates.

    Args:
        xyz (np.ndarray): Input coordinates of shape (N, 3).
        rotation_matrix (np.ndarray): Rotation matrix of shape (3, 3) (usually U from SVD).
        cog (np.ndarray): Center of gravity of shape (3,).

    Returns:
        np.ndarray: Transformed coordinates of shape (N, 3).
    """
    # Center the data
    xyz_centered = xyz - cog

    # Apply rotation
    # Projects the centered coordinates onto the basis vectors defined by rotation_matrix.
    # If rotation_matrix is U from SVD, this aligns the principal axis with the X-axis.
    xyz_transformed = xyz_centered @ rotation_matrix

    return xyz_transformed
