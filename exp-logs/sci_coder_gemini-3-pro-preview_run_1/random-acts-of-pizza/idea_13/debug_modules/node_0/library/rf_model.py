import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from library.config import Config


class TopicAlignedRF:
    """
    Stream A: Topic-Aligned Random Forest.
    Wraps sklearn's RandomForestClassifier with specific configurations for this task,
    designed to ingest the hybrid feature set (Tabular + Topic Alignment + TF-IDF).
    """

    def __init__(self, params=None):
        """
        Initialize the Random Forest model.

        Args:
            params (dict, optional): Hyperparameters for the Random Forest.
                                     If None, uses Config.RF_PARAMS.
        """
        # Use defaults from Config if not provided
        self.params = params if params is not None else Config.RF_PARAMS.copy()

        # Ensure random state is explicitly set for reproducibility
        if "random_state" not in self.params:
            self.params["random_state"] = Config.RANDOM_STATE

        # Initialize the underlying sklearn model
        self.model = RandomForestClassifier(**self.params)
        self.is_fitted = False

    def train(self, X_train, y_train):
        """
        Trains the Random Forest model on the provided data.

        Args:
            X_train (array-like): Training feature matrix (dense or sparse).
            y_train (array-like): Training target vector.
        """
        print(
            f"Training Random Forest with {self.params['n_estimators']} estimators..."
        )

        # Fit the model
        self.model.fit(X_train, y_train)
        self.is_fitted = True

        # Calculate training metrics to monitor potential overfitting
        # We predict probabilities for the positive class (index 1)
        train_probs = self.model.predict_proba(X_train)[:, 1]
        train_auc = roc_auc_score(y_train, train_probs)

        # Print full precision as requested
        print(f"RF Training ROC AUC: {train_auc}")

    def predict_proba(self, X):
        """
        Generates probability predictions for the positive class (received pizza).

        Args:
            X (array-like): Feature matrix.

        Returns:
            np.ndarray: Array of probabilities for class 1.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call train() first.")

        # predict_proba returns shape (n_samples, 2), we want the second column
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, X_val, y_val):
        """
        Evaluates the model on a validation set.

        Args:
            X_val (array-like): Validation feature matrix.
            y_val (array-like): Validation target vector.

        Returns:
            float: ROC AUC score.
        """
        print("Evaluating Random Forest on validation set...")

        val_probs = self.predict_proba(X_val)
        val_auc = roc_auc_score(y_val, val_probs)

        # Print full precision as requested
        print(f"RF Validation ROC AUC: {val_auc}")

        return val_auc
