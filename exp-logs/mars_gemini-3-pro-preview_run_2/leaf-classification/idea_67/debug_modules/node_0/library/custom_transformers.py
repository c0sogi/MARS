import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class Float64Transformer(BaseEstimator, TransformerMixin):
    """
    Transformer that casts the input data to float64 precision.
    This ensures numerical stability for downstream operations and metric calculation.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Convert to numpy array with float64 dtype
        return np.array(X, dtype=np.float64)


class StratifiedLDAReducer(BaseEstimator, TransformerMixin):
    """
    Applies Linear Discriminant Analysis (LDA) independently to specified subsets of features.

    This transformer addresses the issue where high-variance feature groups dominate
    the global covariance matrix, potentially obscuring the structure of lower-variance groups.
    By fitting LDA separately to each group, we ensure optimal manifold alignment for each
    biological component (Margin, Shape, Texture).
    """

    def __init__(
        self,
        feature_slices,
        n_components=None,
        shrinkage=None,
        priors=None,
        solver="svd",
        tol=1e-4,
        store_covariance=False,
    ):
        """
        Args:
            feature_slices (dict): Dictionary mapping group names to slice objects or column indices.
                                   Example: {'margin': slice(0, 64), 'shape': slice(64, 128)}
            n_components (int, optional): Number of components for LDA.
            shrinkage (str or float, optional): Shrinkage parameter for LDA ('auto' or float).
            priors (array-like, optional): Class priors.
            solver (str, optional): Solver to use ('svd', 'lsqr', 'eigen').
            tol (float, optional): Tolerance for SVD solver.
            store_covariance (bool, optional): Whether to store covariance matrices.
        """
        self.feature_slices = feature_slices
        self.n_components = n_components
        self.shrinkage = shrinkage
        self.priors = priors
        self.solver = solver
        self.tol = tol
        self.store_covariance = store_covariance
        self.estimators_ = {}

    def fit(self, X, y):
        """
        Fits a separate LDA estimator for each feature slice defined in self.feature_slices.
        """
        self.estimators_ = {}
        for name, sl in self.feature_slices.items():
            # Extract feature subset
            if hasattr(X, "iloc"):
                X_sub = X.iloc[:, sl].values
            else:
                X_sub = X[:, sl]

            # Create and fit LDA estimator
            lda = LinearDiscriminantAnalysis(
                n_components=self.n_components,
                shrinkage=self.shrinkage,
                priors=self.priors,
                solver=self.solver,
                tol=self.tol,
                store_covariance=self.store_covariance,
            )
            lda.fit(X_sub, y)
            self.estimators_[name] = lda
        return self

    def transform(self, X):
        """
        Transforms the input data by applying the fitted LDA estimators to their respective slices
        and concatenating the results.
        """
        outputs = []
        for name, sl in self.feature_slices.items():
            if name not in self.estimators_:
                raise ValueError(
                    f"Estimator for group '{name}' not found. Ensure fit is called."
                )

            if hasattr(X, "iloc"):
                X_sub = X.iloc[:, sl].values
            else:
                X_sub = X[:, sl]

            # Transform and append
            out = self.estimators_[name].transform(X_sub)
            outputs.append(out)

        # Concatenate all transformed subsets horizontally
        return np.hstack(outputs)


class GroupedInteractionTransformer(BaseEstimator, TransformerMixin):
    """
    Computes pairwise interactions (cross-products) between specified groups of features.

    This is used to model dependencies between different biological domains (e.g., Margin x Texture)
    explicitly. It calculates the row-wise outer product between the feature vectors of the
    specified pairs.
    """

    def __init__(self, feature_slices, interaction_pairs):
        """
        Args:
            feature_slices (dict): Dictionary mapping group names to slice objects.
            interaction_pairs (list of tuples): List of pairs of group names to compute interactions for.
                                                Example: [('margin', 'texture'), ('shape', 'texture')]
        """
        self.feature_slices = feature_slices
        self.interaction_pairs = interaction_pairs

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        outputs = []
        for name1, name2 in self.interaction_pairs:
            sl1 = self.feature_slices[name1]
            sl2 = self.feature_slices[name2]

            if hasattr(X, "iloc"):
                X1 = X.iloc[:, sl1].values
                X2 = X.iloc[:, sl2].values
            else:
                X1 = X[:, sl1]
                X2 = X[:, sl2]

            # Compute row-wise outer product and flatten: (N, D1) x (N, D2) -> (N, D1*D2)
            # Reshape for broadcasting: (N, D1, 1) * (N, 1, D2) = (N, D1, D2)
            # This captures the interaction between every feature in group 1 and every feature in group 2
            interaction = (X1[:, :, None] * X2[:, None, :]).reshape(X1.shape[0], -1)
            outputs.append(interaction)

        if not outputs:
            return np.empty((X.shape[0], 0))

        return np.hstack(outputs)
