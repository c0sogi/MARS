import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from library import config


class LeafLDA:
    """
    A wrapper for Linear Discriminant Analysis (LDA) specifically configured for
    high-dimensional, low-sample datasets using Ledoit-Wolf shrinkage.

    Implements a Transductive Self-Training loop to leverage unlabeled test data
    for robust covariance estimation.
    """

    def __init__(self, solver=config.LDA_SOLVER, shrinkage=config.LDA_SHRINKAGE):
        """
        Initialize the LeafLDA classifier.

        Args:
            solver (str): Solver to use ('lsqr' or 'eigen'). Defaults to config.LDA_SOLVER.
            shrinkage (str or float): Shrinkage parameter ('auto' for Ledoit-Wolf).
                                      Defaults to config.LDA_SHRINKAGE.
        """
        self.solver = solver
        self.shrinkage = shrinkage
        self.model = LinearDiscriminantAnalysis(
            solver=self.solver, shrinkage=self.shrinkage
        )
        self.classes_ = None

    def fit(self, X, y):
        """
        Fit the Linear Discriminant Analysis model according to the given training data.

        Args:
            X (pd.DataFrame or np.ndarray): Training vector, where n_samples is the number of samples
                                            and n_features is the number of features.
            y (pd.Series or np.ndarray): Target values (class labels).

        Returns:
            self: Returns the instance itself.
        """
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        return self

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Args:
            X (pd.DataFrame or np.ndarray): The input samples.

        Returns:
            np.ndarray: Vector of predicted class labels.
        """
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Estimate probability.

        Args:
            X (pd.DataFrame or np.ndarray): The input samples.

        Returns:
            np.ndarray: Returns the probability of the sample for each class in the model.
        """
        return self.model.predict_proba(X)
