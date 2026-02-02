import numpy as np
import torch


def compute_pbc_distance_matrix(coords, lattice_matrix):
    """
    Computes the pairwise distance matrix for a set of coordinates within a unit cell,
    respecting periodic boundary conditions (Minimum Image Convention).

    Args:
        coords: (N, 3) numpy array of atomic coordinates.
        lattice_matrix: (3, 3) numpy array where rows are lattice vectors [v1, v2, v3].

    Returns:
        dist_matrix: (N, N) numpy array of PBC-corrected distances.
    """
    # Calculate difference vectors: diff[i, j] = coords[i] - coords[j]
    # Shape: (N, N, 3)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]

    # Invert lattice matrix to convert to fractional coordinates
    # lattice_matrix rows are vectors. r = f1*v1 + f2*v2 + f3*v3 = f @ L
    # f = r @ inv(L)
    try:
        inv_lattice = np.linalg.inv(lattice_matrix)
    except np.linalg.LinAlgError:
        # Fallback for singular matrix (should not happen in valid geometry)
        # Return Euclidean distances without PBC if lattice is invalid
        return np.linalg.norm(diff, axis=-1)

    # Convert differences to fractional coordinates
    # diff is (N, N, 3). We want to multiply the last dimension by inv_lattice.
    # diff @ inv_lattice results in fractional differences.
    frac_diff = diff @ inv_lattice

    # Apply Minimum Image Convention in fractional space
    # Round to nearest integer to find the nearest image
    # The fractional coordinates should be in [-0.5, 0.5] for the nearest image
    frac_diff = frac_diff - np.round(frac_diff)

    # Convert back to Cartesian coordinates
    # cart_diff = frac_diff @ lattice_matrix
    cart_diff = frac_diff @ lattice_matrix

    # Compute Euclidean norms along the last axis (x, y, z)
    dist_matrix = np.linalg.norm(cart_diff, axis=-1)

    return dist_matrix


class RobustScaler:
    """
    Standardizes features by removing the mean and scaling to unit variance.
    (Z-score normalization).
    Named RobustScaler as per requirements, though implements StandardScaler logic.
    """

    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        """
        Compute the mean and std to be used for later scaling.
        X: numpy array of shape (n_samples, n_features) or (n_samples,)
        """
        # Ensure X is at least 2D for consistent axis behavior, or handle 1D
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)

        # Handle constant features (std=0) to avoid division by zero
        # If std is 0, we set it to 1 so that the feature becomes (x - mean) / 1 = 0
        self.scale_[self.scale_ == 0.0] = 1.0
        return self

    def transform(self, X):
        """
        Perform standardization by centering and scaling.
        """
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler has not been fitted yet.")

        orig_shape = X.shape
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        X_scaled = (X - self.mean_) / self.scale_

        return X_scaled.reshape(orig_shape)

    def fit_transform(self, X):
        """
        Fit to data, then transform it.
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X_scaled):
        """
        Scale back the data to the original representation.
        """
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler has not been fitted yet.")

        orig_shape = X_scaled.shape
        if X_scaled.ndim == 1:
            X_scaled = X_scaled.reshape(-1, 1)

        X_orig = X_scaled * self.scale_ + self.mean_

        return X_orig.reshape(orig_shape)


def log_transform(y):
    """
    Applies log(1 + y) transformation.
    Useful for targets that are non-negative and span several orders of magnitude.
    """
    return np.log1p(y)


def inverse_log_transform(y_log):
    """
    Applies exp(y) - 1 transformation.
    Inverse of log_transform.
    """
    return np.expm1(y_log)


def compute_rmsle(y_true, y_pred):
    """
    Computes the Root Mean Squared Logarithmic Error (RMSLE).

    Metric definition:
    RMSLE = sqrt( mean( (log(1+pred) - log(1+true))^2 ) )

    This function handles both numpy arrays and torch tensors.
    It computes the metric over all elements in the input.
    """
    # Convert torch tensors to numpy if necessary
    if hasattr(y_true, "detach"):
        y_true = y_true.detach().cpu().numpy()
    if hasattr(y_pred, "detach"):
        y_pred = y_pred.detach().cpu().numpy()

    # Clip predictions to be non-negative as log is undefined for negative values
    # This is a safety measure for regression models that might output negative values
    y_pred = np.maximum(y_pred, 0)
    y_true = np.maximum(y_true, 0)

    # Compute log(1 + x)
    log_pred = np.log1p(y_pred)
    log_true = np.log1p(y_true)

    # Compute squared errors
    squared_log_errors = (log_pred - log_true) ** 2

    # Compute mean squared error
    mean_squared_log_error = np.mean(squared_log_errors)

    # Compute root
    rmsle = np.sqrt(mean_squared_log_error)

    return rmsle
