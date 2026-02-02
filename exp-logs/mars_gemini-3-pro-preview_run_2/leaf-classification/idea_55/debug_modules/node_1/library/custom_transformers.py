import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array

from library.config import FEATURE_SLICES, FLOAT_PRECISION, LDA_SOLVER


class StratifiedDiscriminantProjector(BaseEstimator, TransformerMixin):
    """
    A transformer that splits the input features into Margin, Shape, and Texture subsets,
    projects each subset into a discriminative subspace using LDA, and concatenates
    the results.

    This implements the 'Stratified Projection' step of Topology C in the SDPGE strategy.
    """

    def __init__(self, shrinkage=None, n_components=None, solver=LDA_SOLVER):
        """
        Args:
            shrinkage (str or float, optional): Shrinkage parameter for LDA.
                                                'auto', float between 0 and 1, or None.
            n_components (int, optional): Number of components to keep per feature group.
                                          If None, keeps min(n_classes - 1, n_features).
            solver (str): Solver to use for LDA. Defaults to config.LDA_SOLVER ('lsqr').
        """
        self.shrinkage = shrinkage
        self.n_components = n_components
        self.solver = solver

        # Dictionary to hold the fitted LDA models for each group
        self.estimators_ = {}
        # List of group names to ensure consistent order
        self.group_names_ = ["margin", "shape", "texture"]

    def fit(self, X, y):
        """
        Fits separate LDA models for Margin, Shape, and Texture feature subsets.

        Args:
            X (array-like): Input data of shape (n_samples, 192).
            y (array-like): Target labels.

        Returns:
            self
        """
        # Ensure X and y are valid
        X, y = check_X_y(X, y, dtype=FLOAT_PRECISION)

        # Clear any existing estimators
        self.estimators_ = {}

        # Iterate through the defined feature slices
        for group_name in self.group_names_:
            if group_name not in FEATURE_SLICES:
                raise ValueError(
                    f"Expected feature slice '{group_name}' not found in configuration."
                )

            slice_obj = FEATURE_SLICES[group_name]

            # Extract the subset of features
            X_subset = X[:, slice_obj]

            # Initialize LDA for this group
            # Note: solver='svd' does not support shrinkage, so we respect the passed solver
            # unless shrinkage is specified, which requires lsqr or eigen.
            # The config defaults to 'lsqr', so this is generally safe.
            lda = LinearDiscriminantAnalysis(
                solver=self.solver,
                shrinkage=self.shrinkage,
                n_components=self.n_components,
            )

            # Fit the LDA
            lda.fit(X_subset, y)

            # Store the fitted estimator
            self.estimators_[group_name] = lda

        return self

    def transform(self, X):
        """
        Projects X into the stratified discriminative subspaces.

        Args:
            X (array-like): Input data of shape (n_samples, 192).

        Returns:
            np.ndarray: Concatenated projected features.
        """
        # Check if fit has been called
        check_is_fitted(self, "estimators_")

        # Validate input
        X = check_array(X, dtype=FLOAT_PRECISION)

        projected_parts = []

        for group_name in self.group_names_:
            slice_obj = FEATURE_SLICES[group_name]
            estimator = self.estimators_[group_name]

            # Extract subset
            X_subset = X[:, slice_obj]

            # Project
            X_projected = estimator.transform(X_subset)

            # Ensure strictly float64
            X_projected = X_projected.astype(FLOAT_PRECISION)

            projected_parts.append(X_projected)

        # Concatenate all projected parts horizontally
        X_transformed = np.hstack(projected_parts)

        return X_transformed

    def get_feature_names_out(self, input_features=None):
        """
        Returns the names of the output features.
        Format: {group}_lda{component_index}
        """
        check_is_fitted(self, "estimators_")

        feature_names = []
        for group_name in self.group_names_:
            estimator = self.estimators_[group_name]
            # Determine number of output components for this estimator
            # LDA stores this in classes_ - 1 or n_components
            # We can inspect the shape of the means_ or scalings_ to infer,
            # but usually we just check the transformed shape on a dummy or rely on n_components logic.
            # A safer way without transforming data is checking `estimator.scalings_`.
            # scalings_ shape is (n_features, n_components)

            n_out = estimator.scalings_.shape[1]

            names = [f"{group_name}_lda{i}" for i in range(n_out)]
            feature_names.extend(names)

        return np.array(feature_names, dtype=object)
