import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from library.config import Config


class GlobalLDA(BaseEstimator, ClassifierMixin):
    """
    Baseline Linear Discriminant Analysis model trained on all classes globally.
    Uses automatic shrinkage (Ledoit-Wolf) to handle high-dimensional feature spaces.
    """

    def __init__(
        self, solver=Config.GLOBAL_LDA_SOLVER, shrinkage=Config.GLOBAL_LDA_SHRINKAGE
    ):
        self.solver = solver
        self.shrinkage = shrinkage
        self.model = None

    def fit(self, X, y):
        self.model = LinearDiscriminantAnalysis(
            solver=self.solver, shrinkage=self.shrinkage
        )
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        if self.model is None:
            raise RuntimeError("GlobalLDA model not fitted")
        return self.model.predict_proba(X)
