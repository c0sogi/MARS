import os
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from scipy.special import logit, expit
from library.config import Config
from library.utils import seed_everything, calculate_log_loss


class PlattScaler:
    """
    Implements Platt Scaling (Post-Hoc Calibration) using Logistic Regression.

    This class fits a scalar scaling model to map raw model outputs (logits)
    to calibrated probabilities, minimizing Log Loss.
    """

    def __init__(self):
        """
        Initialize the PlattScaler.
        Uses Logistic Regression with no penalty to act as a pure scaler.
        """
        seed_everything()
        # penalty=None is supported in recent sklearn versions.
        # We use lbfgs for stability and no regularization to minimize log loss directly on the training set (OOF).
        self.model = LogisticRegression(
            penalty=None, solver="lbfgs", random_state=Config.SEED, fit_intercept=True
        )
        self.is_fitted = False

    def fit(self, logits, labels):
        """
        Fit the calibration model on OOF logits and labels.

        Args:
            logits (np.ndarray): Raw output logits from the model/ensemble. Shape (N,) or (N, 1).
            labels (np.ndarray): Ground truth binary labels. Shape (N,).

        Returns:
            self
        """
        # Ensure numpy arrays
        logits = np.asarray(logits)
        labels = np.asarray(labels)

        # Reshape logits to (N, 1) for sklearn
        if logits.ndim == 1:
            logits = logits.reshape(-1, 1)

        print(f"Fitting PlattScaler on {len(logits)} samples...")

        # Calculate pre-calibration metric (converting logits to probs via sigmoid)
        pre_cal_probs = expit(logits)
        pre_loss = calculate_log_loss(labels, pre_cal_probs)
        print(f"Pre-calibration OOF Log Loss: {pre_loss:.15f}")

        # Fit the model
        self.model.fit(logits, labels)
        self.is_fitted = True

        # Calculate post-calibration metric
        post_cal_probs = self.model.predict_proba(logits)[:, 1]
        post_loss = calculate_log_loss(labels, post_cal_probs)
        print(f"Post-calibration OOF Log Loss: {post_loss:.15f}")

        # Print scaling parameters
        # Model: P(y=1|z) = sigmoid(coef * z + intercept)
        if hasattr(self.model, "coef_"):
            print(
                f"Scaling Parameters -> Coef (A): {self.model.coef_[0][0]:.6f}, Intercept (B): {self.model.intercept_[0]:.6f}"
            )

        return self

    def predict(self, logits):
        """
        Apply calibration to new logits.

        Args:
            logits (np.ndarray): Raw output logits. Shape (N,) or (N, 1).

        Returns:
            np.ndarray: Calibrated probabilities.
        """
        if not self.is_fitted:
            raise RuntimeError("PlattScaler is not fitted yet.")

        logits = np.asarray(logits)
        if logits.ndim == 1:
            logits = logits.reshape(-1, 1)

        # Return probability of class 1
        return self.model.predict_proba(logits)[:, 1]

    def predict_from_probs(self, probs, epsilon=1e-15):
        """
        Apply calibration to probabilities by first converting them to logits.
        Useful when the ensemble outputs probabilities.

        Args:
            probs (np.ndarray): Input probabilities.
            epsilon (float): Clipping value to avoid inf logits.

        Returns:
            np.ndarray: Calibrated probabilities.
        """
        probs = np.asarray(probs)
        # Clip probabilities to avoid logit(0) or logit(1) -> inf
        probs_clipped = np.clip(probs, epsilon, 1 - epsilon)

        # Convert to logits
        logits = logit(probs_clipped)

        return self.predict(logits)

    def save(self, path=None):
        """
        Save the fitted model to disk.

        Args:
            path (str, optional): Path to save the model. Defaults to Config.CALIBRATION_PATH.
        """
        if path is None:
            path = Config.CALIBRATION_PATH

        if not self.is_fitted:
            print("Warning: Saving an unfitted PlattScaler.")

        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)

        joblib.dump(self.model, path)
        print(f"PlattScaler saved to {path}")

    def load(self, path=None):
        """
        Load a fitted model from disk.

        Args:
            path (str, optional): Path to load the model from. Defaults to Config.CALIBRATION_PATH.

        Returns:
            self
        """
        if path is None:
            path = Config.CALIBRATION_PATH

        if not os.path.exists(path):
            raise FileNotFoundError(f"Calibration model not found at {path}")

        self.model = joblib.load(path)
        self.is_fitted = True
        print(f"PlattScaler loaded from {path}")

        # Print parameters for verification
        if hasattr(self.model, "coef_"):
            print(
                f"Loaded Parameters -> Coef (A): {self.model.coef_[0][0]:.6f}, Intercept (B): {self.model.intercept_[0]:.6f}"
            )

        return self
