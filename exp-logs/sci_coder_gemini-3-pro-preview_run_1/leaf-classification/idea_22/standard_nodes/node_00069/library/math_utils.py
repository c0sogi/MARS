import numpy as np


def compute_geometric_median(X, eps=1e-6, max_iter=100):
    """
    Computes the geometric median of a point cloud using Weiszfeld's algorithm.

    The geometric median is the point that minimizes the sum of Euclidean distances
    to all points in the dataset. It is a robust estimator of location, less
    sensitive to outliers than the arithmetic mean.

    Args:
        X (np.ndarray): Input data array of shape (n_samples, n_features).
        eps (float): Convergence threshold. Iteration stops when the update
                     shift is less than this value. Default is 1e-6.
        max_iter (int): Maximum number of iterations. Default is 100.

    Returns:
        np.ndarray: The geometric median vector of shape (n_features,).
    """
    # Ensure double precision for numerical stability and metric requirements
    X = np.asarray(X, dtype=np.float64)

    # Initialize with the arithmetic mean (center of mass)
    y = np.mean(X, axis=0)

    for _ in range(max_iter):
        # Calculate Euclidean distances from the current estimate 'y' to all points in X
        # shape: (n_samples,)
        distances = np.linalg.norm(X - y, axis=1)

        # Handle singularity: If the estimate falls exactly on a data point, distance is 0.
        # We clip the distance to a very small positive number to avoid division by zero.
        # In high-dimensional continuous space, exact overlap is rare, but this ensures robustness.
        distances = np.maximum(distances, 1e-20)

        # Calculate Weiszfeld weights: w_i = 1 / ||x_i - y||
        weights = 1.0 / distances

        # Normalize weights to sum to 1
        total_weight = np.sum(weights)
        if total_weight == 0:
            # This theoretically shouldn't happen with the clipping above
            break
        weights = weights / total_weight

        # Update the estimate: y_{t+1} = sum(w_i * x_i)
        # This is the weighted average of the points
        y_new = np.dot(weights, X)

        # Check for convergence
        shift = np.linalg.norm(y_new - y)
        if shift < eps:
            y = y_new
            break

        y = y_new

    return y
