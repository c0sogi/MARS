import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class NBTransformer(BaseEstimator, TransformerMixin):
    """
    A Transformer that scales features by the log-ratio of their class-conditional probabilities
    (Naive Bayes weights). This transformation is the core component of NB-SVM and NB-LR
    models, which are strong baselines for short-text classification tasks.
    """

    def __init__(self, alpha=1.0):
        """
        Initialize the NBTransformer.

        Args:
            alpha (float): Additive (Laplace/Lidstone) smoothing parameter.
                           Must be > 0 to prevent division by zero. Default is 1.0.
        """
        self.alpha = alpha
        self._r = None

    def fit(self, X, y):
        """
        Compute the Naive Bayes log-count ratios from the training data.

        Args:
            X (sparse matrix or array-like): Training features of shape (n_samples, n_features).
            y (array-like): Target labels of shape (n_samples,). Must be binary (0 and 1).

        Returns:
            self: Returns the instance itself.
        """
        # Validate inputs
        # We accept sparse matrices (CSR/CSC) to handle high-dimensional text data efficiently.
        X, y = check_X_y(X, y, accept_sparse=["csr", "csc", "coo"], dtype=np.float64)

        # Ensure binary classification
        classes = np.unique(y)
        if len(classes) != 2:
            # Fallback for single-class batches or multi-class errors
            raise ValueError(
                f"NBTransformer expects exactly 2 classes, found {len(classes)}: {classes}"
            )

        # Identify classes: assuming 0 is negative (Neutral) and 1 is positive (Insult)
        # We sort to ensure consistent mapping: smaller label -> neg, larger label -> pos
        classes.sort()
        neg_class, pos_class = classes[0], classes[1]

        # Create boolean masks for each class
        mask_pos = y == pos_class
        mask_neg = y == neg_class

        # Ensure X is CSR for efficient row slicing
        if not sparse.issparse(X) or X.format != "csr":
            X_csr = sparse.csr_matrix(X)
        else:
            X_csr = X

        # Calculate sum of features for each class
        # sum(axis=0) on a sparse matrix returns a matrix/array of shape (1, n_features)
        # We flatten it to a 1D array for element-wise operations
        p_sum = np.array(X_csr[mask_pos].sum(axis=0)).flatten()
        q_sum = np.array(X_csr[mask_neg].sum(axis=0)).flatten()

        # Apply smoothing
        p = p_sum + self.alpha
        q = q_sum + self.alpha

        # Normalize to get probabilities (Likelihoods)
        # p_norm[i] = P(feature_i | Positive)
        # q_norm[i] = P(feature_i | Negative)
        p_norm = p / p.sum()
        q_norm = q / q.sum()

        # Compute the log-ratio (weight vector)
        # r[i] = log( P(feature_i | Pos) / P(feature_i | Neg) )
        self._r = np.log(p_norm / q_norm)

        return self

    def transform(self, X):
        """
        Scale the features by the learned log-count ratios.

        Args:
            X (sparse matrix or array-like): Features to transform.

        Returns:
            sparse matrix: The transformed features (element-wise multiplied by weights).
        """
        # Check if fit has been called
        check_is_fitted(self, "_r")

        # Validate input
        X = check_array(X, accept_sparse=["csr", "csc", "coo"], dtype=np.float64)

        # Perform element-wise multiplication
        # The sparse matrix multiply method broadcasts the 1D array _r across all rows of X
        # equivalent to: X[i, j] * _r[j]
        X_transformed = X.multiply(self._r)

        return X_transformed
