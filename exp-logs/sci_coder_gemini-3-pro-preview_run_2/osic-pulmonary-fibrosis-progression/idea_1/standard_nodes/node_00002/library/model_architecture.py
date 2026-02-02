import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from library.config import (
    XGB_PARAMS,
    MIN_CONFIDENCE,
    SUBMISSION_PATH,
)
from library.metrics import laplace_log_likelihood


class DualModel:
    """
    Radiomics-Enhanced Dual XGBoost.

    This architecture uses two parallel Gradient Boosting models:
    1. FVC Regressor: Predicts the target lung capacity (FVC).
    2. Uncertainty Regressor: Predicts the confidence (sigma) by learning the magnitude
       of the error (absolute residuals) from the FVC Regressor.
       Cite solution_lesson_node_00001: Decoupled Residual Regression.
    """

    def __init__(self, params=XGB_PARAMS):
        self.params = params

        # Model 1: Predicts FVC
        self.fvc_model = XGBRegressor(**self.params)

        # Model 2: Predicts Confidence (Sigma)
        self.sigma_model = XGBRegressor(**self.params)

    def fit(self, X, y):
        """
        Trains the dual models sequentially.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target vector (FVC).
        """
        # Stage 1: Train FVC Predictor
        self.fvc_model.fit(X, y)

        # Stage 2: Train Uncertainty Predictor
        # Generate predictions on the training set to calculate residuals
        y_pred_train = self.fvc_model.predict(X)

        # Calculate absolute residuals (modeling the error magnitude)
        abs_residuals = np.abs(y - y_pred_train)

        # Fit sigma model to predict these residuals
        self.sigma_model.fit(X, abs_residuals)

        return self

    def predict(self, X):
        """
        Predicts FVC and Confidence for new data.

        Args:
            X (np.ndarray): Feature matrix.

        Returns:
            tuple: (fvc_pred, sigma_clipped)
        """
        # Predict FVC
        fvc_pred = self.fvc_model.predict(X)

        # Predict Uncertainty (Sigma)
        # The model predicts Mean Absolute Error (MAE).
        # For Laplace distribution, Sigma = sqrt(2) * MAE.
        sigma_pred = self.sigma_model.predict(X)
        sigma_pred = sigma_pred * np.sqrt(2)

        # Clip Sigma to minimum threshold (70ml) as per task requirement
        sigma_clipped = np.maximum(sigma_pred, MIN_CONFIDENCE)

        return fvc_pred, sigma_clipped


def train_model(X_train, y_train, X_val, y_val):
    """
    Handles the training and evaluation loop.

    Args:
        X_train, y_train: Training data.
        X_val, y_val: Validation data.

    Returns:
        DualModel: The trained model instance.
    """
    print("Initializing Radiomics-Enhanced Dual XGBoost Model...")
    model = DualModel()

    print(f"Training on {len(X_train)} samples...")
    # Note: ElasticNet uses coordinate descent with an internal convergence check (tol).
    # This acts as the stopping criterion, so explicit epoch-based early stopping is not required.
    model.fit(X_train, y_train)

    print("Evaluating on Validation Set...")
    val_fvc_pred, val_sigma_pred = model.predict(X_val)

    # Compute Metric
    score = laplace_log_likelihood(y_val, val_fvc_pred, val_sigma_pred)

    # Print full precision metric
    print(f"Validation Laplace Log Likelihood: {score}")

    return model


def generate_submission(model, X_test, test_df):
    """
    Generates predictions for the test set and saves the submission file.

    Args:
        model (DualModel): Trained model.
        X_test (np.ndarray): Test features.
        test_df (pd.DataFrame): Test metadata containing 'Patient_Week'.
    """
    print("Generating predictions for test set...")
    fvc_pred, sigma_pred = model.predict(X_test)

    # Create submission DataFrame
    # Ensure we use the Patient_Week column from the test metadata
    submission = pd.DataFrame(
        {
            "Patient_Week": test_df["Patient_Week"],
            "FVC": fvc_pred,
            "Confidence": sigma_pred,
        }
    )

    # Save to disk
    print(f"Saving submission to {SUBMISSION_PATH}...")
    submission.to_csv(SUBMISSION_PATH, index=False)

    return submission
