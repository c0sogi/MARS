import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class Float64Wrapper(BaseEstimator, TransformerMixin):
    """
    A transformer that strictly enforces float64 precision on the input data.
    This ensures that subsequent pipeline steps operate with double precision,
    minimizing numerical noise which is critical for the stability of
    LDA and other linear algebra-heavy operations in this task.
    """

    def fit(self, X, y=None):
        """
        Stateless transformer, fit does nothing.
        """
        return self

    def transform(self, X):
        """
        Casts X to float64.
        """
        # copy=False allows zero-copy if data is already float64 and contiguous
        return check_array(X, dtype=np.float64, copy=False)


class GroupedLDAReducer(BaseEstimator, TransformerMixin):
    """
    A supervised transformer that implements the 'Factorized-Bottleneck' topology.

    It splits the input feature matrix into semantic groups based on provided indices,
    fits an independent Linear Discriminant Analysis (LDA) reduction model for each group,
    and concatenates the projected discriminative subspaces.

    This allows for domain-specific dimensionality reduction before interaction expansion.
    """

    def __init__(
        self, feature_indices, n_components=5, solver="eigen", shrinkage="auto"
    ):
        """
        Args:
            feature_indices (dict): Dictionary mapping group names to lists of column indices.
                                    Example: {'shape': [0, 1, ...], 'texture': [64, 65, ...]}
            n_components (int): Number of discriminative components to retain per group.
            solver (str): Solver to use for LDA ('svd', 'lsqr', 'eigen').
            shrinkage (str or float): Shrinkage parameter for LDA regularization.
        """
        self.feature_indices = feature_indices
        self.n_components = n_components
        self.solver = solver
        self.shrinkage = shrinkage

    def fit(self, X, y):
        """
        Fits an independent LDA model for each feature group defined in feature_indices.

        Args:
            X (array-like): Training data of shape (n_samples, n_features).
            y (array-like): Target values.
        """
        # Ensure input is float64 for precision
        X, y = check_X_y(X, y, dtype=np.float64)

        self.models_ = {}
        self.group_names_ = sorted(
            list(self.feature_indices.keys())
        )  # Sort for deterministic order

        unique_classes = np.unique(y)
        n_classes = len(unique_classes)

        for group_name in self.group_names_:
            indices = self.feature_indices[group_name]

            # Extract feature subset for this group
            # We use safe indexing
            X_subset = X[:, indices]
            n_features_group = X_subset.shape[1]

            # Determine valid number of components
            # LDA requires n_components <= min(n_features, n_classes - 1)
            max_possible_components = min(n_features_group, n_classes - 1)
            actual_n_components = min(self.n_components, max_possible_components)

            # Initialize and fit LDA
            # If actual_n_components is 0 (e.g. 1 class or 0 features), this handles edge cases gracefully usually,
            # but for this task n_classes=99, so we are safe.
            if actual_n_components < 1:
                actual_n_components = 1  # Fallback, though unlikely given dataset

            lda = LinearDiscriminantAnalysis(
                n_components=actual_n_components,
                solver=self.solver,
                shrinkage=self.shrinkage,
            )

            lda.fit(X_subset, y)
            self.models_[group_name] = lda

        return self

    def transform(self, X):
        """
        Transforms the input data by projecting each group onto its discriminative subspace
        and concatenating the results.

        Args:
            X (array-like): Input data of shape (n_samples, n_features).

        Returns:
            np.ndarray: The transformed data with shape (n_samples, sum(actual_n_components)).
        """
        check_is_fitted(self, ["models_"])
        X = check_array(X, dtype=np.float64)

        transformed_parts = []

        # Iterate in the same sorted order as fit
        for group_name in self.group_names_:
            indices = self.feature_indices[group_name]
            model = self.models_[group_name]

            X_subset = X[:, indices]

            # Project
            X_trans = model.transform(X_subset)
            transformed_parts.append(X_trans)

        # Concatenate along the feature axis (axis 1)
        return np.hstack(transformed_parts)
