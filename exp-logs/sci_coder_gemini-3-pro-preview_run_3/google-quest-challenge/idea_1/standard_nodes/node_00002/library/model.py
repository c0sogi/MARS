import os
import numpy as np
import joblib
from sklearn.linear_model import RidgeCV
from library.utils import compute_spearman_score
from library.config import SEED, WORKING_DIR


class LinearHead:
    """
    A wrapper around RidgeCV for multi-output regression with [0,1] clipping.
    """

    def __init__(self, alphas=None):
        """
        Initialize the LinearHead model.

        Args:
            alphas (np.ndarray or list, optional): Array of alpha values to try
                                                   for regularization.
                                                   Defaults to a logspace range.
        """
        if alphas is None:
            # Search over a broad range of regularization strengths
            self.alphas = np.logspace(-4, 4, 17)
        else:
            self.alphas = alphas

        # RidgeCV handles multi-output regression natively
        self.model = RidgeCV(alphas=self.alphas, scoring=None)
        self.is_fitted = False

    def fit(self, X_train, y_train, X_val=None, y_val=None, target_cols=None):
        """
        Trains the Ridge Regression model.

        Args:
            X_train (np.ndarray): Training features (N_train, n_features).
            y_train (np.ndarray): Training targets (N_train, n_targets).
            X_val (np.ndarray, optional): Validation features.
            y_val (np.ndarray, optional): Validation targets.
            target_cols (list, optional): List of target column names for metric calculation.
        """
        print(
            f"Training LinearHead (RidgeCV) on {X_train.shape[0]} samples with {X_train.shape[1]} features..."
        )

        # Fit the model
        # RidgeCV automatically selects the best alpha via efficient LOOCV (Generalized Cross-Validation)
        self.model.fit(X_train, y_train)
        self.is_fitted = True

        print(f"Model fitted. Best alpha (average): {np.mean(self.model.alpha_)}")

        # Evaluate if validation data is provided
        if X_val is not None and y_val is not None and target_cols is not None:
            print(f"Evaluating on {X_val.shape[0]} validation samples...")
            val_preds = self.predict(X_val)

            # Compute metric
            score = compute_spearman_score(y_val, val_preds, target_cols)
            print(f"Validation Spearman Correlation: {score}")

    def predict(self, X):
        """
        Generates predictions and clips them to [0, 1].

        Args:
            X (np.ndarray): Features (N, n_features).

        Returns:
            np.ndarray: Predicted probabilities (N, n_targets) in range [0, 1].
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

        # Generate raw predictions
        preds = self.model.predict(X)

        # Clip predictions to ensure valid probability range [0, 1]
        preds_clipped = np.clip(preds, 0.0, 1.0)

        return preds_clipped

    def save(self, filename="linear_head_model.joblib"):
        """
        Saves the trained model to the working directory.

        Args:
            filename (str): Name of the file.
        """
        if not self.is_fitted:
            print("Warning: Attempting to save an unfitted model.")

        path = os.path.join(WORKING_DIR, "idea_2", filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        print(f"Model saved to {path}")

    def load(self, filename="linear_head_model.joblib"):
        """
        Loads a trained model from the working directory.

        Args:
            filename (str): Name of the file.
        """
        path = os.path.join(WORKING_DIR, "idea_2", filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        self.model = joblib.load(path)
        self.is_fitted = True
        print(f"Model loaded from {path}")
