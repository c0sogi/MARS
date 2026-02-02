import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from library.config import Config


class LDAManager:
    """
    Wrapper for Linear Discriminant Analysis (LDA) classifier.

    Implements the 'View-Expanded Manifold Stabilization' strategy component
    where a robust linear classifier is trained on the projected feature space.
    Includes Ledoit-Wolf shrinkage to handle high-dimensional covariance estimation.
    """

    def __init__(self):
        """
        Initialize the LDA model using parameters from the global Configuration.
        """
        self.solver = Config.LDA_SOLVER
        self.shrinkage = Config.LDA_SHRINKAGE

        self.model = LinearDiscriminantAnalysis(
            solver=self.solver, shrinkage=self.shrinkage
        )
        self.is_fitted = False

    def train(self, X, y):
        """
        Fit the LDA model to the provided training data.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
            y (np.ndarray): Target labels of shape (n_samples,).
        """
        # print(f"Training LDA (Solver: {self.solver}, Shrinkage: {self.shrinkage})...")
        self.model.fit(X, y)
        self.is_fitted = True
        # print(f"Model fitted. Classes identified: {len(self.model.classes_)}")

    def predict_proba(self, X):
        """
        Predict class probabilities for the given samples.

        Applies clipping to the range [1e-15, 1 - 1e-15] to avoid extremes
        in the multi-class log loss metric.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: Probability matrix of shape (n_samples, n_classes).
        """
        if not self.is_fitted:
            raise RuntimeError("LDAManager must be trained before prediction.")

        # Generate raw probabilities
        probs = self.model.predict_proba(X)

        # Clip probabilities to avoid log(0)
        # Metric requirement: max(min(p, 1-10^-15), 10^-15)
        epsilon = 1e-15
        probs = np.clip(probs, epsilon, 1.0 - epsilon)

        return probs

    @property
    def classes_(self):
        """
        Access the class labels identified during training.

        Returns:
            np.ndarray: Array of class labels.
        """
        if not self.is_fitted:
            return None
        return self.model.classes_
