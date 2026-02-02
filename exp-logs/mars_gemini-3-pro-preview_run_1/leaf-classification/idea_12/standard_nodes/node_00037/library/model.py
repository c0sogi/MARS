import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import log_loss, accuracy_score
from library.config import Config


class RobustLDA:
    """
    Wrapper class for Linear Discriminant Analysis (LDA) with robust settings
    for high-dimensional, potentially small-sample data.

    This class implements the final classification stage of the Transductive PCA-LDA
    pipeline, utilizing the 'eigen' solver for stability and Ledoit-Wolf shrinkage
    for regularized covariance estimation.
    """

    def __init__(self):
        """
        Initialize the LDA model with hyperparameters from Config.

        Configuration:
            solver: Config.LDA_SOLVER (Expected: 'eigen')
            shrinkage: Config.LDA_SHRINKAGE (Expected: 'auto')

        Note:
            priors are set to None, which defaults to inferring class priors
            from the training data (Empirical Priors). This is appropriate
            given the stratified nature of the dataset.
        """
        self.solver = Config.LDA_SOLVER
        self.shrinkage = Config.LDA_SHRINKAGE

        self.model = LinearDiscriminantAnalysis(
            solver=self.solver, shrinkage=self.shrinkage
        )
        self.classes_ = None

    def fit(self, X, y):
        """
        Fit the LDA model to the training data.

        Args:
            X (np.ndarray): Training features (n_samples, n_features).
            y (np.ndarray): Training labels (n_samples,).

        Returns:
            self: The fitted instance.
        """
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        return self

    def predict_proba(self, X):
        """
        Predict class probabilities for the input samples.

        Args:
            X (np.ndarray): Input features (n_samples, n_features).

        Returns:
            np.ndarray: Class probabilities (n_samples, n_classes).
        """
        return self.model.predict_proba(X)

    def predict(self, X):
        """
        Predict class labels for the input samples.

        Args:
            X (np.ndarray): Input features (n_samples, n_features).

        Returns:
            np.ndarray: Predicted class labels.
        """
        return self.model.predict(X)

    def evaluate(self, X, y, dataset_name="Validation"):
        """
        Evaluate the model on a given dataset and print metrics with full precision.

        Args:
            X (np.ndarray): Features.
            y (np.ndarray): True labels.
            dataset_name (str): Name of the dataset for logging.

        Returns:
            dict: Dictionary containing 'log_loss' and 'accuracy'.
        """
        probs = self.predict_proba(X)
        preds = self.predict(X)

        # Calculate metrics
        # labels parameter ensures correct mapping even if y doesn't contain all classes
        # in the specific batch (though unlikely in validation set)
        loss = log_loss(y, probs, labels=self.classes_)
        acc = accuracy_score(y, preds)

        print(f"[{dataset_name}] Multi-class Log Loss: {loss}")
        print(f"[{dataset_name}] Accuracy: {acc}")

        return {"log_loss": loss, "accuracy": acc}
