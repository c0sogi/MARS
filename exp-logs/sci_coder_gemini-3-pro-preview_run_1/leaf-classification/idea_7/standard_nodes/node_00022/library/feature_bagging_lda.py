from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from library.config import (
    SEED,
    LDA_SOLVER,
    LDA_SHRINKAGE,
)
from library.utils import set_seed


class SingleLDA:
    """
    A wrapper for a single Linear Discriminant Analysis (LDA) estimator.
    Replaces the Feature Bagging ensemble to avoid variance reduction on stable classifiers.
    Cite solution_lesson_node_00011: Bagging degrades performance for stable classifiers.
    Cite solution_lesson_node_00008: Use standalone model when ensemble is redundant.
    """

    def __init__(
        self,
        random_state=SEED,
    ):
        self.random_state = random_state
        self.clf = None
        self.classes_ = None

    def fit(self, X, y):
        """
        Fits the single LDA model.

        Args:
            X (np.ndarray): Training features.
            y (np.ndarray): Training labels.
        """
        set_seed(self.random_state)

        # Using 'lsqr' solver and 'auto' shrinkage (Ledoit-Wolf)
        # Cite solution_lesson_node_00005: Use shrinkage for high-dimensional data.
        self.clf = LinearDiscriminantAnalysis(
            solver=LDA_SOLVER, shrinkage=LDA_SHRINKAGE
        )
        self.clf.fit(X, y)
        self.classes_ = self.clf.classes_
        return self

    def predict_proba(self, X):
        return self.clf.predict_proba(X)

    def predict(self, X):
        return self.clf.predict(X)
