import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import log_loss, accuracy_score


class LDAModel:
    """
    Implements the Linear Discriminant Analysis (LDA) classifier for the
    Transductive Gaussian-Fisher Discriminant strategy.

    Key Configuration:
    - Solver: 'eigen' (Eigenvalue Decomposition). This is chosen for its
      bit-wise deterministic behavior and high numerical precision, which is
      critical for minimizing log loss in the saturation regime.
    - Shrinkage: 'auto' (Ledoit-Wolf). This analytic shrinkage is essential
      for stabilizing the covariance matrix estimation given the small sample
      size (approx 10 samples per class) relative to the feature dimension (192).
    """

    def __init__(self, solver="eigen", shrinkage="auto", priors=None, tol=1e-4):
        """
        Initialize the LDA Model.

        Args:
            solver (str): Solver to use ('eigen' is required for this strategy).
            shrinkage (str or float): Regularization parameter ('auto' for Ledoit-Wolf).
            priors (array-like, optional): Class priors. If None, priors are empirical.
            tol (float): Tolerance for singular values.
        """
        self.solver = solver
        self.shrinkage = shrinkage
        self.priors = priors
        self.tol = tol
        self.model = None

    def fit(self, X, y):
        """
        Fit the Linear Discriminant Analysis model.

        Args:
            X (array-like): Training data of shape (n_samples, n_features).
            y (array-like): Target values of shape (n_samples,).

        Returns:
            self: Returns the instance itself.
        """
        self.model = LinearDiscriminantAnalysis(
            solver=self.solver,
            shrinkage=self.shrinkage,
            priors=self.priors,
            tol=self.tol,
        )
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        """
        Estimate probability.

        Args:
            X (array-like): Input data of shape (n_samples, n_features).

        Returns:
            array-like: Estimated probabilities of shape (n_samples, n_classes).
        """
        if self.model is None:
            raise RuntimeError("Model must be fitted before calling predict_proba")
        return self.model.predict_proba(X)

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Args:
            X (array-like): Input data of shape (n_samples, n_features).

        Returns:
            array-like: Predicted class labels.
        """
        if self.model is None:
            raise RuntimeError("Model must be fitted before calling predict")
        return self.model.predict(X)

    def evaluate(self, X, y):
        """
        Evaluate the model using Log Loss and Accuracy.
        Prints metrics with full precision as required.

        Args:
            X (array-like): Validation data.
            y (array-like): True labels.

        Returns:
            dict: Dictionary containing 'log_loss' and 'accuracy'.
        """
        if self.model is None:
            raise RuntimeError("Model must be fitted before calling evaluate")

        # Generate predictions
        probs = self.predict_proba(X)
        preds = self.predict(X)

        # Calculate metrics
        # Log loss automatically handles the mapping of probabilities to classes
        # provided the columns of probs correspond to self.classes_ (which sklearn ensures)
        loss = log_loss(y, probs)
        acc = accuracy_score(y, preds)

        # Print with full precision (no formatting strings)
        print(f"Validation Log Loss: {loss}")
        print(f"Validation Accuracy: {acc}")

        return {"log_loss": loss, "accuracy": acc}

    @property
    def classes_(self):
        """
        Returns the class labels known to the model.
        Useful for mapping probabilities to column names in submission.
        """
        if self.model is None:
            return None
        return self.model.classes_
