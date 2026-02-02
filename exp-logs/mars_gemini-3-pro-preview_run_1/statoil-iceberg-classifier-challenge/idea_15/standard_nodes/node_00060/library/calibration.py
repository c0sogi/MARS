import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from library.config import Config


class PlattScaler:
    """
    Implements Platt Scaling using Logistic Regression to calibrate probabilities.
    Minimizes Log Loss between logits and targets.

    This class fits a Logistic Regression model on the raw logits (z) to learn
    parameters A and B such that P(y=1|z) = Sigmoid(A*z + B).

    It explicitly stores and applies coefficients to avoid pickling restrictions
    and ensure lightweight inference.
    """

    def __init__(self):
        self.A = None  # Coefficient (Scale)
        self.B = None  # Intercept (Shift)
        self.is_fitted = False

    def fit(self, logits, targets):
        """
        Fits the calibration model using sklearn's LogisticRegression.
        Extracts coefficients for manual application.

        Args:
            logits (np.ndarray): Raw model outputs (before sigmoid), shape (N,) or (N, 1).
            targets (np.ndarray): True binary labels, shape (N,).
        """
        # Ensure numpy arrays
        logits = np.array(logits)
        targets = np.array(targets)

        # Reshape logits to (N, 1) for sklearn input requirement
        if logits.ndim == 1:
            X = logits.reshape(-1, 1)
        else:
            X = logits

        # Initialize and fit Logistic Regression
        # C is set high (1000.0) to reduce regularization bias, as we want to
        # approximate the Maximum Likelihood Estimate of the calibration parameters.
        model = LogisticRegression(solver="lbfgs", C=1000.0)
        model.fit(X, targets)

        # Extract parameters
        # model.coef_ shape is (1, 1), model.intercept_ shape is (1,)
        self.A = float(model.coef_[0][0])
        self.B = float(model.intercept_[0])
        self.is_fitted = True

        print(f"Platt Scaler Fitted. A (Scale): {self.A:.6f}, B (Shift): {self.B:.6f}")

    def predict_proba(self, logits):
        """
        Applies calibration to logits using the learned coefficients.
        P(y=1|z) = 1 / (1 + exp(-(Az + B)))

        Args:
            logits (np.ndarray): Raw model outputs, shape (N,) or (N, 1).

        Returns:
            np.ndarray: Calibrated probabilities (P(y=1)), flattened to shape (N,).
        """
        if not self.is_fitted:
            raise RuntimeError("PlattScaler must be fitted before prediction.")

        logits = np.array(logits)

        # Apply linear transformation: z_calibrated = A * z + B
        # Broadcasting handles both (N,) and (N, 1) shapes correctly
        z_cal = logits * self.A + self.B

        # Numerical Stability: Clip values to prevent overflow in exp()
        # exp(-700) is ~0, exp(700) is huge.
        z_cal = np.clip(z_cal, -700, 700)

        # Apply Sigmoid: 1 / (1 + exp(-z))
        probs = 1.0 / (1.0 + np.exp(-z_cal))

        # Ensure flat output for consistency with submission requirements
        return probs.flatten()

    def save(self, filename="platt_scaler_params.npz"):
        """
        Saves the learned parameters to a .npz file in the checkpoint directory.
        """
        if not self.is_fitted:
            print("Warning: Attempting to save an unfitted PlattScaler.")
            return

        filepath = os.path.join(Config.CHECKPOINT_DIR, filename)
        np.savez(filepath, A=self.A, B=self.B)
        print(f"Platt Scaler parameters saved to {filepath}")

    def load(self, filename="platt_scaler_params.npz"):
        """
        Loads parameters from a .npz file in the checkpoint directory.
        """
        filepath = os.path.join(Config.CHECKPOINT_DIR, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No scaler parameters found at {filepath}")

        data = np.load(filepath)
        self.A = float(data["A"])
        self.B = float(data["B"])
        self.is_fitted = True
        print(
            f"Platt Scaler parameters loaded from {filepath}: A={self.A:.6f}, B={self.B:.6f}"
        )
