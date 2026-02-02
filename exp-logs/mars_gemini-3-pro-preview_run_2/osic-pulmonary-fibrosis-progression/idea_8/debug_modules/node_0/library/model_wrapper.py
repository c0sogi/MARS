import numpy as np
import pandas as pd
import os
from sklearn.linear_model import QuantileRegressor, GammaRegressor

from library.config import (
    SEED,
    QUANTILE_ALPHA,
    MIN_CONFIDENCE,
    MAX_ERROR_METRIC,
    SUBMISSION_PATH,
    CACHE_DIR,
)
from library.feature_pipeline import run_feature_pipeline

# Ensure reproducibility
np.random.seed(SEED)


class StratifiedQuantileGLM:
    def __init__(self, quantile=0.5, max_iter=1000):
        """
        Initializes the dual-branch regression system.

        Args:
            quantile (float): The target quantile for FVC prediction (default 0.5 for Median).
            max_iter (int): Maximum iterations for the solvers.
        """
        self.quantile = quantile

        # Branch 1: FVC Predictor (Linear Quantile Regressor)
        # We use alpha=0 for unregularized linear quantile regression.
        # solver='highs' is efficient for linear programming problems in quantile regression.
        self.fvc_model = QuantileRegressor(quantile=quantile, alpha=0.0, solver="highs")

        # Branch 2: Uncertainty Predictor (Gamma GLM)
        # GammaRegressor models strictly positive continuous data.
        # Log link is standard for Gamma to ensure positive predictions.
        self.unc_model = GammaRegressor(alpha=0.0, max_iter=max_iter, solver="lbfgs")

    def fit(self, X_fvc, y_fvc, X_unc):
        """
        Trains the model using Decoupled Residual Regression.

        1. Train FVC model on (X_fvc, y_fvc).
        2. Predict on training set.
        3. Compute absolute residuals.
        4. Train Uncertainty model on (X_unc, residuals).
        """
        # 1. Train FVC Model
        print(f"Training FVC QuantileRegressor (q={self.quantile})...")
        self.fvc_model.fit(X_fvc, y_fvc)

        # 2. Generate Residuals
        y_pred = self.fvc_model.predict(X_fvc)
        residuals = np.abs(y_fvc - y_pred)

        # Gamma GLM requires strictly positive targets.
        # Add a small epsilon to zero residuals.
        epsilon = 1e-6
        residuals = np.maximum(residuals, epsilon)

        # 3. Train Uncertainty Model
        print("Training Uncertainty GammaRegressor...")
        self.unc_model.fit(X_unc, residuals)

        return self

    def predict(self, X_fvc, X_unc):
        """
        Predicts FVC and Confidence.

        Returns:
            fvc_pred (np.array): Predicted median FVC.
            sigma_pred (np.array): Predicted confidence (std dev).
        """
        # Predict Median FVC
        fvc_pred = self.fvc_model.predict(X_fvc)

        # Predict Expected Absolute Error (Delta)
        delta_pred = self.unc_model.predict(X_unc)

        # Convert Delta to Sigma (Standard Deviation)
        # For Laplace distribution: Sigma = Delta * sqrt(2)
        # This analytically maps the predicted Mean Absolute Deviation
        # to the optimal scale parameter for the Laplace metric.
        sigma_pred = delta_pred * np.sqrt(2)

        return fvc_pred, sigma_pred


def calculate_laplace_metric(y_true, y_pred, sigma):
    """
    Computes the competition metric: Modified Laplace Log Likelihood.

    metric = - (sqrt(2) * Delta / sigma_clipped) - ln(sqrt(2) * sigma_clipped)
    where Delta is clipped at 1000 and sigma is clipped at 70.
    """
    # Clip sigma
    sigma_clipped = np.maximum(sigma, MIN_CONFIDENCE)

    # Calculate Delta with thresholding
    delta = np.minimum(np.abs(y_true - y_pred), MAX_ERROR_METRIC)

    # Calculate Metric
    metric = -(np.sqrt(2) * delta) / sigma_clipped - np.log(np.sqrt(2) * sigma_clipped)

    return np.mean(metric)


def run_training_and_inference(load_cached_data=True, max_iter=1000):
    """
    Main execution function.

    1. Loads data via feature pipeline.
    2. Trains StratifiedQuantileGLM.
    3. Validates and prints exact metric.
    4. Generates and saves submission.
    """
    # 1. Load Data
    print("Loading data via Feature Pipeline...")
    train_data, val_data, test_data = run_feature_pipeline(
        load_cached_data=load_cached_data
    )

    X_fvc_train, y_fvc_train, X_unc_train, df_train = train_data
    X_fvc_val, y_fvc_val, X_unc_val, df_val = val_data
    X_fvc_test, _, X_unc_test, df_test = test_data

    # 2. Initialize and Train Model
    model = StratifiedQuantileGLM(quantile=QUANTILE_ALPHA, max_iter=max_iter)
    model.fit(X_fvc_train, y_fvc_train, X_unc_train)

    # 3. Validation
    print("Evaluating on Validation Set...")
    val_fvc_pred, val_sigma_pred = model.predict(X_fvc_val, X_unc_val)

    score = calculate_laplace_metric(y_fvc_val, val_fvc_pred, val_sigma_pred)
    print(f"Validation Laplace Log Likelihood: {score}")

    # 4. Inference on Test Set
    print("Generating Test Predictions...")
    test_fvc_pred, test_sigma_pred = model.predict(X_fvc_test, X_unc_test)

    # Create Submission DataFrame
    submission = pd.DataFrame(
        {
            "Patient_Week": df_test["Patient_Week"],
            "FVC": test_fvc_pred,
            "Confidence": test_sigma_pred,
        }
    )

    # Ensure FVC and Confidence are integers as per sample submission format
    # Note: While internal calculations use floats, submission usually expects ints.
    # We round to nearest integer.
    submission["FVC"] = submission["FVC"].round().astype(int)
    submission["Confidence"] = submission["Confidence"].round().astype(int)

    # Save
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
    print(f"Saving submission to {SUBMISSION_PATH}...")
    submission.to_csv(SUBMISSION_PATH, index=False)
    print("Done.")
