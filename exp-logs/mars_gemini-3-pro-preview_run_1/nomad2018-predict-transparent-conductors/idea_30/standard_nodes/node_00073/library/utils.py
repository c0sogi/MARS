import numpy as np


def calculate_cell_volume(lattice_matrix):
    """
    Calculates the volume of the unit cell given the lattice matrix.

    Args:
        lattice_matrix (np.ndarray): 3x3 matrix where rows are lattice vectors.

    Returns:
        float: Volume of the unit cell.
    """
    return np.abs(np.linalg.det(lattice_matrix))


def compute_pbc_distances(coords, lattice_matrix):
    """
    Computes pairwise distances between atoms respecting periodic boundary conditions
    using the Minimum Image Convention (MIC).

    Args:
        coords (np.ndarray): (N, 3) array of atomic Cartesian coordinates.
        lattice_matrix (np.ndarray): (3, 3) matrix where rows are lattice vectors.

    Returns:
        np.ndarray: (N, N) matrix of pairwise distances.
    """
    # Compute difference vectors (N, N, 3)
    # diff[i, j] = coords[i] - coords[j]
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]

    # Convert to fractional coordinates to apply MIC
    # r_cart = r_frac @ lattice_matrix  => r_frac = r_cart @ lattice_matrix^-1
    try:
        inv_lattice = np.linalg.inv(lattice_matrix)
    except np.linalg.LinAlgError:
        # Fallback: return simple Euclidean distances if lattice is singular
        return np.linalg.norm(diff, axis=-1)

    diff_frac = diff @ inv_lattice

    # Apply Minimum Image Convention: map fractional coordinates to [-0.5, 0.5]
    diff_frac -= np.round(diff_frac)

    # Convert back to Cartesian coordinates
    diff_cart = diff_frac @ lattice_matrix

    # Compute Euclidean norms
    dist_matrix = np.linalg.norm(diff_cart, axis=-1)

    return dist_matrix


def log_transform_targets(targets):
    """
    Applies log(1 + x) transformation to targets to normalize distribution
    and align with RMSLE metric optimization.

    Args:
        targets (np.ndarray): Array of target values.

    Returns:
        np.ndarray: Transformed targets.
    """
    return np.log1p(targets)


def inverse_log_transform_targets(transformed_targets):
    """
    Applies exp(x) - 1 transformation to reverse log transform
    and recover original scale.

    Args:
        transformed_targets (np.ndarray): Array of log-transformed target values.

    Returns:
        np.ndarray: Original scale targets.
    """
    return np.expm1(transformed_targets)


def calculate_rmsle(y_true, y_pred):
    """
    Calculates the Column-wise Root Mean Squared Logarithmic Error.

    Args:
        y_true (np.ndarray): Ground truth values (N, D) or (N,).
        y_pred (np.ndarray): Predicted values (N, D) or (N,).

    Returns:
        float: Mean RMSLE across columns.
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Clip predictions to be non-negative to avoid log domain errors
    # (Energy values should physically be non-negative or handled appropriately)
    y_pred = np.maximum(y_pred, 0)
    y_true = np.maximum(y_true, 0)

    # Calculate squared log error
    squared_log_error = (np.log1p(y_pred) - np.log1p(y_true)) ** 2

    # Mean squared log error per column (if 2D) or overall (if 1D)
    if y_true.ndim > 1:
        msle_per_column = np.mean(squared_log_error, axis=0)
    else:
        msle_per_column = np.mean(squared_log_error)

    # Root mean squared log error per column
    rmsle_per_column = np.sqrt(msle_per_column)

    # Average across columns (if multiple targets)
    return np.mean(rmsle_per_column)
