import numpy as np
import pandas as pd
import os
from sklearn.linear_model import QuantileRegressor, ElasticNet
import warnings

from library.config import Config
from library.utils import mad_to_sigma, laplace_log_likelihood


class LaplaceSolver:
    """
    A wrapper class implementing the Decoupled Residual Regression strategy.
    Consists of:
    1. FVCPredictor: Linear Quantile Regressor (Median)
    2. UncertaintyPredictor: Elastic Net Regressor (Predicts MAD of residuals)
    """

    def __init__(self):
        # FVC Predictor: Linear Quantile Regression
        # Optimizes L1 Loss (Median), which aligns with the Laplace metric's location parameter.
        # We use a small alpha for stability and regularization.
        # solver='highs' is an efficient linear programming solver suitable for this scale.
        self.fvc_model = QuantileRegressor(
            quantile=Config.QUANTILE, alpha=0.01, solver="highs"  # Light regularization
        )

        # Uncertainty Predictor: Elastic Net
        # Predicts the expected absolute error (MAD).
        # Uses L1+L2 regularization to prevent overfitting to noisy residuals.
        self.unc_model = ElasticNet(
            alpha=0.1,  # Regularization strength
            l1_ratio=Config.ELASTIC_L1_RATIO,
            random_state=Config.SEED,
        )

    def fit(self, X_fvc, y, X_unc):
        """
        Trains the dual-model system.
        1. Fit FVC model on (X_fvc, y).
        2. Compute residuals r = |y - y_pred|.
        3. Fit Uncertainty model on (X_unc, r).

        Args:
            X_fvc (np.array): Features for FVC prediction (PCA + Interactions).
            y (np.array): True FVC targets.
            X_unc (np.array): Features for Uncertainty prediction (PCA + Horizon).
        """
        # 1. Train FVC Model
        self.fvc_model.fit(X_fvc, y)

        # 2. Compute Residuals
        y_pred = self.fvc_model.predict(X_fvc)
        residuals = np.abs(y - y_pred)

        # 3. Train Uncertainty Model
        self.unc_model.fit(X_unc, residuals)

        return self

    def predict(self, X_fvc, X_unc):
        """
        Generates predictions for FVC and Confidence.

        Args:
            X_fvc (np.array): Features for FVC prediction.
            X_unc (np.array): Features for Uncertainty prediction.

        Returns:
            fvc_pred (np.array): Predicted Median FVC.
            sigma_pred (np.array): Predicted Confidence (Std Dev).
        """
        # Predict Median FVC
        fvc_pred = self.fvc_model.predict(X_fvc)

        # Predict MAD (Mean Absolute Deviation)
        mad_pred = self.unc_model.predict(X_unc)

        # Ensure MAD is non-negative (linear models can extrapolate below 0)
        mad_pred = np.maximum(mad_pred, 0)

        # Convert MAD to Sigma (scaling by sqrt(2) for Laplace distribution)
        sigma_pred = mad_to_sigma(mad_pred)

        return fvc_pred, sigma_pred

    def evaluate(self, X_fvc, X_unc, y_true):
        """
        Computes the Laplace Log Likelihood on a given set.
        """
        fvc_pred, sigma_pred = self.predict(X_fvc, X_unc)
        score = laplace_log_likelihood(y_true, fvc_pred, sigma_pred)
        return score


def train_laplace_solver(data_dict_train, data_dict_val=None):
    """
    Orchestrates the training and evaluation of the LaplaceSolver.

    Args:
        data_dict_train (dict): Contains 'X_fvc', 'X_unc', 'y' for training.
        data_dict_val (dict, optional): Contains 'X_fvc', 'X_unc', 'y' for validation.

    Returns:
        model (LaplaceSolver): The trained model instance.
    """
    model = LaplaceSolver()

    X_fvc_train = data_dict_train["X_fvc"]
    X_unc_train = data_dict_train["X_unc"]
    y_train = data_dict_train["y"]

    # Fit the model
    model.fit(X_fvc_train, y_train, X_unc_train)

    # Evaluate on Train
    train_score = model.evaluate(X_fvc_train, X_unc_train, y_train)
    print(f"Training Score (Laplace LL): {train_score}")

    # Evaluate on Validation if provided
    if data_dict_val:
        X_fvc_val = data_dict_val["X_fvc"]
        X_unc_val = data_dict_val["X_unc"]
        y_val = data_dict_val["y"]

        val_score = model.evaluate(X_fvc_val, X_unc_val, y_val)
        print(f"Validation Score (Laplace LL): {val_score}")

    return model


def generate_submission(model, data_dict_test, test_metadata_path, output_path):
    """
    Generates the submission file for the test set.

    Args:
        model (LaplaceSolver): Trained model.
        data_dict_test (dict): Contains 'X_fvc', 'X_unc' for test set.
        test_metadata_path (str): Path to test_metadata.csv to retrieve Patient_Week IDs.
        output_path (str): Path to save the submission CSV.
    """
    # Load test metadata to get the Patient_Week identifiers
    df_test = pd.read_csv(test_metadata_path)

    # Ensure the order matches. The DataPipeline processes rows in order of the CSV.
    X_fvc_test = data_dict_test["X_fvc"]
    X_unc_test = data_dict_test["X_unc"]

    # Generate Predictions
    fvc_pred, sigma_pred = model.predict(X_fvc_test, X_unc_test)

    # Create Submission DataFrame
    submission = pd.DataFrame(
        {
            "Patient_Week": df_test["Patient_Week"],
            "FVC": fvc_pred,
            "Confidence": sigma_pred,
        }
    )

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
