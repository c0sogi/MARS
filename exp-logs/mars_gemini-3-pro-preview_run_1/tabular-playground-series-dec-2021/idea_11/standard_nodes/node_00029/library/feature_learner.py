import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from library.config import Config


class SupervisedProjector:
    """
    A class to handle supervised feature learning using Linear Discriminant Analysis (LDA).
    It projects high-dimensional features into a lower-dimensional space that maximizes
    class separability.
    """

    def __init__(self, n_components=None):
        """
        Initialize the SupervisedProjector.

        Args:
            n_components (int, optional): Number of components for dimensionality reduction.
                                          Defaults to Config.LDA_COMPONENTS.
        """
        self.n_components = (
            n_components if n_components is not None else Config.LDA_COMPONENTS
        )
        self.model = None

    def fit(self, X, y):
        """
        Fits the LDA model to the training data.

        Args:
            X (array-like or pd.DataFrame): Training features.
            y (array-like): Target labels.

        Returns:
            self: Returns the instance itself.
        """
        # Ensure inputs are numpy-compatible
        if isinstance(X, pd.DataFrame):
            X = X.values
        y = np.array(y).ravel()

        # Determine the maximum allowable components
        # LDA constraint: n_components <= min(n_classes - 1, n_features)
        n_classes = len(np.unique(y))
        n_features = X.shape[1]
        max_components = min(n_classes - 1, n_features)

        # Adjust n_components if necessary
        actual_components = min(self.n_components, max_components)

        # Initialize and fit the model
        self.model = LinearDiscriminantAnalysis(n_components=actual_components)
        self.model.fit(X, y)

        return self

    def transform(self, X):
        """
        Projects the input data using the learned LDA transformation.

        Args:
            X (array-like or pd.DataFrame): Features to transform.

        Returns:
            np.ndarray: The projected features.
        """
        if self.model is None:
            raise RuntimeError(
                "The SupervisedProjector must be fitted before calling transform."
            )

        if isinstance(X, pd.DataFrame):
            X = X.values

        return self.model.transform(X)

    def fit_transform(self, X, y):
        """
        Fits the model to the data and returns the transformed version.

        Args:
            X (array-like or pd.DataFrame): Training features.
            y (array-like): Target labels.

        Returns:
            np.ndarray: The projected features.
        """
        self.fit(X, y)
        return self.transform(X)
