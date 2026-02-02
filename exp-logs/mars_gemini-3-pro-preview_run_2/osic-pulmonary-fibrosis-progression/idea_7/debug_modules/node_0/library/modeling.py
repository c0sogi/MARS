import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from library.config import SUBMISSION_PATH, TEST_META_PATH, SEED
from library.utils import laplace_log_likelihood, seed_everything

# Ensure reproducibility
seed_everything(SEED)


class MedianRegressor:
    """
    A wrapper for Quantile Regression targeting the Median (q=0.5).
    This minimizes the L1 loss (Mean Absolute Error).
    """

    def __init__(self):
        self.model = None
        self.result = None

    def fit(self, X, y):
        """
        Fits the Quantile Regression model.
        Args:
            X: Feature matrix (numpy array).
            y: Target vector (numpy array).
        """
        # Add intercept
        X_const = sm.add_constant(X, has_constant="add")

        # Initialize QuantReg model
        self.model = sm.QuantReg(y, X_const)

        # Fit for median (q=0.5)
        # max_iter is increased to ensure convergence on high-dim data
        self.result = self.model.fit(q=0.5, max_iter=2000)

    def predict(self, X):
        """
        Predicts the median FVC.
        """
        if self.result is None:
            raise ValueError("Model not fitted yet.")

        X_const = sm.add_constant(X, has_constant="add")
        return self.result.predict(X_const)


class UncertaintyGLM:
    """
    A wrapper for Generalized Linear Model (Gamma Family, Log Link).
    Models the expected absolute error (uncertainty).
    """

    def __init__(self):
        self.model = None
        self.result = None

    def fit(self, X, residuals):
        """
        Fits the GLM on absolute residuals.
        Args:
            X: Feature matrix.
            residuals: Absolute errors |y_true - y_pred|.
        """
        # Add intercept
        X_const = sm.add_constant(X, has_constant="add")

        # Gamma family with Log link is standard for modeling variance/dispersion
        family = sm.families.Gamma(link=sm.families.links.log())

        self.model = sm.GLM(residuals, X_const, family=family)
        self.result = self.model.fit()

    def predict(self, X):
        """
        Predicts the expected absolute error (Delta).
        """
        if self.result is None:
            raise ValueError("Model not fitted yet.")

        X_const = sm.add_constant(X, has_constant="add")
        return self.result.predict(X_const)


def train_models(data_dict):
    """
    Orchestrates the training of FVC and Uncertainty models.

    Args:
        data_dict: Dictionary containing preprocessed numpy arrays:
                   'X_fvc_train', 'y_train', 'X_unc_train',
                   'X_fvc_val', 'y_val', 'X_unc_val'

    Returns:
        fvc_model: Trained MedianRegressor.
        unc_model: Trained UncertaintyGLM.
    """
    print("[Modeling] Starting training pipeline...")

    X_fvc_train = data_dict["X_fvc_train"]
    y_train = data_dict["y_train"]
    X_unc_train = data_dict["X_unc_train"]

    X_fvc_val = data_dict["X_fvc_val"]
    y_val = data_dict["y_val"]
    X_unc_val = data_dict["X_unc_val"]

    # --- Step 1: Train FVC Model (Median Regression) ---
    print("[Modeling] Fitting Median Regressor (FVC)...")
    fvc_model = MedianRegressor()
    fvc_model.fit(X_fvc_train, y_train)

    # Evaluate FVC Model
    train_preds_fvc = fvc_model.predict(X_fvc_train)
    val_preds_fvc = fvc_model.predict(X_fvc_val)

    train_mae = np.mean(np.abs(y_train - train_preds_fvc))
    val_mae = np.mean(np.abs(y_val - val_preds_fvc))

    print(f"[Modeling] FVC MAE - Train: {train_mae}, Val: {val_mae}")

    # --- Step 2: Compute Residuals for Uncertainty Training ---
    # We model the absolute error.
    # Add epsilon to ensure strictly positive values for Gamma log-link.
    epsilon = 1e-3
    train_residuals = np.abs(y_train - train_preds_fvc) + epsilon

    # --- Step 3: Train Uncertainty Model (Gamma GLM) ---
    print("[Modeling] Fitting Gamma GLM (Uncertainty)...")
    unc_model = UncertaintyGLM()
    unc_model.fit(X_unc_train, train_residuals)

    # Predict Expected Absolute Error (Delta)
    # Note: The GLM predicts the mean of the residuals (Delta)
    val_delta = unc_model.predict(X_unc_val)

    # Convert Delta to Sigma for Laplace Metric
    # For Laplace distribution, sigma = Delta * sqrt(2)
    val_sigma = val_delta * np.sqrt(2)

    # --- Step 4: Final Evaluation ---
    score = laplace_log_likelihood(y_val, val_preds_fvc, val_sigma)
    print(f"[Modeling] Validation Laplace Log Likelihood: {score}")

    return fvc_model, unc_model


def generate_submission(fvc_model, unc_model, X_fvc_test, X_unc_test):
    """
    Generates predictions for the test set and saves the submission file.

    Args:
        fvc_model: Trained MedianRegressor.
        unc_model: Trained UncertaintyGLM.
        X_fvc_test: Test features for FVC model.
        X_unc_test: Test features for Uncertainty model.
    """
    print("[Modeling] Generating submission...")

    # 1. Generate Predictions
    pred_fvc = fvc_model.predict(X_fvc_test)
    pred_delta = unc_model.predict(X_unc_test)

    # Convert Delta to Confidence (Sigma)
    # sigma = Delta * sqrt(2)
    pred_sigma = pred_delta * np.sqrt(2)

    # 2. Load Test Metadata to get IDs
    # We need the Patient_Week column which corresponds to the rows in X_test
    if not os.path.exists(TEST_META_PATH):
        raise FileNotFoundError(f"Test metadata not found at {TEST_META_PATH}")

    df_test = pd.read_csv(TEST_META_PATH)

    # Safety check
    if len(df_test) != len(pred_fvc):
        raise ValueError(
            f"Shape mismatch: Metadata has {len(df_test)} rows, predictions have {len(pred_fvc)}."
        )

    # 3. Create Submission DataFrame
    submission = pd.DataFrame(
        {
            "Patient_Week": df_test["Patient_Week"],
            "FVC": pred_fvc,
            "Confidence": pred_sigma,
        }
    )

    # 4. Save
    # Ensure directory exists (handled by config, but good practice)
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"[Modeling] Submission saved to {SUBMISSION_PATH}")
    print(submission.head())
