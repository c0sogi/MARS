import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from library import utils, config


class Float64Transformer(BaseEstimator, TransformerMixin):
    """
    A transformer that strictly casts input data to double precision (float64).
    This ensures numerical stability and minimizes noise, particularly for
    manifold learning and density estimation tasks.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        """
        Stateless transformer, fit does nothing.
        """
        return self

    def transform(self, X):
        """
        Casts the input data to float64.
        """
        return utils.enforce_float64(X)


class LDADimensionalityReducer(BaseEstimator, TransformerMixin):
    """
    A wrapper around sklearn's LinearDiscriminantAnalysis designed to be used
    purely as a supervised dimensionality reduction transformer within a pipeline.

    It projects the input data onto the most discriminative linear subspaces
    identified by LDA. This is crucial for the 'Hierarchical Interaction Experts'
    to create dense, discriminative embeddings before polynomial expansion.
    """

    def __init__(
        self,
        n_components=None,
        solver="svd",
        shrinkage=None,
        priors=None,
        store_covariance=False,
        tol=1e-4,
    ):
        """
        Args:
            n_components (int, optional): Number of components for dimensionality reduction.
            solver (str): Solver to use ('svd', 'lsqr', or 'eigen').
            shrinkage (str or float, optional): Shrinkage parameter for covariance estimation.
            priors (array-like, optional): Class priors.
            store_covariance (bool): Whether to compute class covariance matrices.
            tol (float): Tolerance for SVD.
        """
        self.n_components = n_components
        self.solver = solver
        self.shrinkage = shrinkage
        self.priors = priors
        self.store_covariance = store_covariance
        self.tol = tol
        self.lda_ = None

    def fit(self, X, y):
        """
        Fits the underlying LinearDiscriminantAnalysis model.

        Args:
            X: Training data (n_samples, n_features).
            y: Target labels (n_samples,).
        """
        # Ensure high precision for calculation
        X = utils.enforce_float64(X)

        self.lda_ = LinearDiscriminantAnalysis(
            n_components=self.n_components,
            solver=self.solver,
            shrinkage=self.shrinkage,
            priors=self.priors,
            store_covariance=self.store_covariance,
            tol=self.tol,
        )

        self.lda_.fit(X, y)
        return self

    def transform(self, X):
        """
        Projects the data onto the learned discriminative directions.

        Args:
            X: Input data (n_samples, n_features).

        Returns:
            Transformed data (n_samples, n_components) in float64.
        """
        if self.lda_ is None:
            raise RuntimeError("LDADimensionalityReducer must be fit before transform.")

        X = utils.enforce_float64(X)
        X_transformed = self.lda_.transform(X)

        return utils.enforce_float64(X_transformed)
