import os
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss

from library.config import (
    LR_SOLVER,
    LR_MAX_ITER,
    LR_CLASS_WEIGHT,
    LR_C,
    SEED,
    WORKING_DIR,
)


class MalignancyClassifier:
    """
    A wrapper around sklearn's LogisticRegression to classify skin lesions
    based on features extracted from a frozen backbone and tabular metadata.
    """

    def __init__(self):
        """
        Initializes the Logistic Regression model with configurations
        defined in library.config.
        """
        self.model = LogisticRegression(
            solver=LR_SOLVER,
            max_iter=LR_MAX_ITER,
            class_weight=LR_CLASS_WEIGHT,
            C=LR_C,
            random_state=SEED,
            n_jobs=-1,  # Use all available vCPUs
            verbose=0,  # Silent execution
        )

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Trains the model and prints evaluation metrics.

        Args:
            X_train (np.ndarray): Training feature matrix.
            y_train (np.ndarray): Training target vector.
            X_val (np.ndarray, optional): Validation feature matrix.
            y_val (np.ndarray, optional): Validation target vector.
        """
        print(f"Training Logistic Regression (Solver: {LR_SOLVER}, C: {LR_C})...")

        # Fit the model
        self.model.fit(X_train, y_train)

        # Calculate Training Metrics
        train_probs = self.model.predict_proba(X_train)[:, 1]
        train_auc = roc_auc_score(y_train, train_probs)
        train_loss = log_loss(y_train, train_probs)

        print("=== Training Metrics ===")
        print(f"Train ROC AUC: {train_auc}")
        print(f"Train Log Loss: {train_loss}")

        # Calculate Validation Metrics if data is provided
        if X_val is not None and y_val is not None:
            val_probs = self.model.predict_proba(X_val)[:, 1]
            val_auc = roc_auc_score(y_val, val_probs)
            val_loss = log_loss(y_val, val_probs)

            print("=== Validation Metrics ===")
            print(f"Validation ROC AUC: {val_auc}")
            print(f"Validation Log Loss: {val_loss}")

    def predict_proba(self, X):
        """
        Predicts the probability of malignancy (class 1).

        Args:
            X (np.ndarray): Feature matrix.

        Returns:
            np.ndarray: Probabilities for the positive class.
        """
        # predict_proba returns shape (n_samples, 2), we want column 1
        return self.model.predict_proba(X)[:, 1]

    def save(self, filename="logistic_regression_model.joblib"):
        """
        Saves the trained model to the working directory.

        Args:
            filename (str): Name of the file to save.
        """
        path = os.path.join(WORKING_DIR, filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        print(f"Model saved to {path}")

    def load(self, filename="logistic_regression_model.joblib"):
        """
        Loads a trained model from the working directory.

        Args:
            filename (str): Name of the file to load.

        Returns:
            self: The instance with the loaded model.
        """
        path = os.path.join(WORKING_DIR, filename)
        if os.path.exists(path):
            self.model = joblib.load(path)
            print(f"Model loaded from {path}")
        else:
            raise FileNotFoundError(f"Model file not found at {path}")
        return self
