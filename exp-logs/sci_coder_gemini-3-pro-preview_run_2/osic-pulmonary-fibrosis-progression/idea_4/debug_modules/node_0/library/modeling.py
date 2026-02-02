import os
import numpy as np
import pandas as pd
import warnings
from sklearn.linear_model import QuantileRegressor, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from library.config import Config
from library.utils import laplace_log_likelihood_metric

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


class FVCPredictor:
    """
    A wrapper for Quantile Regression to predict the median FVC.
    Minimizes L1 loss (Least Absolute Deviations).
    """

    def __init__(self, quantile=0.5, alpha=1.0, max_iter=2500):
        self.quantile = quantile
        self.alpha = alpha
        self.max_iter = max_iter
        self.scaler = StandardScaler()
        self.model = None

    def fit(self, X, y):
        """
        Fits the Quantile Regressor.

        Args:
            X (np.array): Feature matrix (Static + Time + Interactions).
            y (np.array): Target FVC values.
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Initialize model
        # Try to use 'highs' solver for performance if available (sklearn >= 1.1)
        try:
            self.model = QuantileRegressor(
                quantile=self.quantile,
                alpha=self.alpha,
                solver="highs",
                fit_intercept=True,
            )
            self.model.fit(X_scaled, y)
        except Exception:
            # Fallback for older sklearn versions or missing dependencies
            self.model = QuantileRegressor(
                quantile=self.quantile,
                alpha=self.alpha,
                solver="interior-point",
                max_iter=self.max_iter,
                fit_intercept=True,
            )
            self.model.fit(X_scaled, y)

        return self

    def predict(self, X):
        """
        Predicts median FVC.

        Args:
            X (np.array): Feature matrix.

        Returns:
            np.array: Predicted values.
        """
        if self.model is None:
            raise RuntimeError("Model must be fitted before prediction.")

        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


class UncertaintyPredictor:
    """
    A wrapper for ElasticNet to predict the Mean Absolute Deviation (MAD) of residuals.
    Minimizes L2 loss on absolute errors.
    """

    def __init__(self, alpha=0.1, l1_ratio=0.5, seed=42):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.seed = seed
        self.scaler = StandardScaler()
        self.model = ElasticNet(
            alpha=self.alpha, l1_ratio=self.l1_ratio, random_state=self.seed
        )

    def fit(self, X, residuals):
        """
        Fits the ElasticNet on absolute residuals.

        Args:
            X (np.array): Static feature matrix.
            residuals (np.array): Absolute difference between true FVC and predicted FVC.
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        self.model.fit(X_scaled, residuals)
        return self

    def predict(self, X):
        """
        Predicts expected MAD.

        Args:
            X (np.array): Static feature matrix.

        Returns:
            np.array: Predicted MAD values.
        """
        if self.model is None:
            raise RuntimeError("Model must be fitted before prediction.")

        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


def train_and_evaluate(
    X_train_full, y_train, X_train_static, X_val_full, y_val, X_val_static
):
    """
    Orchestrates the two-stage training process and evaluates on validation set.

    Args:
        X_train_full (np.array): Training features for FVC predictor (includes time/interactions).
        y_train (np.array): Training targets.
        X_train_static (np.array): Training features for Uncertainty predictor (static only).
        X_val_full (np.array): Validation features for FVC predictor.
        y_val (np.array): Validation targets.
        X_val_static (np.array): Validation features for Uncertainty predictor.

    Returns:
        tuple: (fvc_model, unc_model)
    """
    print("--- Stage 1: Training FVC Predictor (Quantile Regression) ---")
    fvc_model = FVCPredictor(
        quantile=Config.QUANTILE, alpha=0.5
    )  # Reduced alpha for less bias
    fvc_model.fit(X_train_full, y_train)

    # Evaluate FVC Model
    y_pred_train = fvc_model.predict(X_train_full)
    y_pred_val = fvc_model.predict(X_val_full)

    mae_train = mean_absolute_error(y_train, y_pred_train)
    mae_val = mean_absolute_error(y_val, y_pred_val)

    print(f"FVC Train MAE: {mae_train}")
    print(f"FVC Val MAE:   {mae_val}")

    print("\n--- Stage 2: Training Uncertainty Predictor (ElasticNet) ---")
    # Calculate residuals from training set
    train_residuals = np.abs(y_train - y_pred_train)

    unc_model = UncertaintyPredictor(alpha=0.1, l1_ratio=0.5, seed=Config.SEED)
    unc_model.fit(X_train_static, train_residuals)

    # Predict MAD on Validation
    mad_val = unc_model.predict(X_val_static)

    # Convert MAD to Sigma (Scale parameter for Laplace)
    # For Laplace distribution, sigma = MAD * sqrt(2)
    sigma_val = mad_val * np.sqrt(2)

    # Calculate Final Metric
    metric_score = laplace_log_likelihood_metric(y_val, y_pred_val, sigma_val)
    print(f"Validation Laplace Log Likelihood: {metric_score}")

    return fvc_model, unc_model


def generate_submission(fvc_model, unc_model, X_test_full, X_test_static, test_ids):
    """
    Generates predictions for the test set and saves the submission file.

    Args:
        fvc_model (FVCPredictor): Trained FVC model.
        unc_model (UncertaintyPredictor): Trained Uncertainty model.
        X_test_full (np.array): Test features for FVC predictor.
        X_test_static (np.array): Test features for Uncertainty predictor.
        test_ids (np.array): Array of Patient_Week strings.
    """
    print("\n--- Generating Submission ---")

    # Predict FVC (Median)
    y_pred_test = fvc_model.predict(X_test_full)

    # Predict Uncertainty (MAD)
    mad_test = unc_model.predict(X_test_static)

    # Convert to Sigma
    sigma_test = mad_test * np.sqrt(2)

    # Clip Confidence as per metric definition
    # Note: Metric function clips at 70, but we should also clip here for the file
    sigma_test = np.maximum(sigma_test, 70)

    # Construct DataFrame
    submission = pd.DataFrame(
        {"Patient_Week": test_ids, "FVC": y_pred_test, "Confidence": sigma_test}
    )

    # Save
    save_path = Config.SUBMISSION_PATH
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    submission.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")
    print(f"Submission shape: {submission.shape}")
    print(submission.head())
