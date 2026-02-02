import os
import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor, ElasticNet
from library.config import Config
from library.utils import score_func, seed_everything


class FVCPredictor:
    """
    Linear Quantile Regressor targeting the Median (q=0.5).
    Optimizes L1 loss (Mean Absolute Error) to align with the Laplace metric location parameter.
    """

    def __init__(self, alpha=1.0):
        """
        Args:
            alpha (float): Regularization strength.
        """
        # solver='highs' is efficient for linear programming problems in quantile regression
        self.model = QuantileRegressor(quantile=0.5, alpha=alpha, solver="highs")

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


class UncertaintyPredictor:
    """
    Elastic Net Regressor to predict the magnitude of errors (Absolute Residuals).
    Models temporal heteroscedasticity using the horizon feature.
    """

    def __init__(self, alpha=0.1, l1_ratio=0.5):
        """
        Args:
            alpha (float): Constant that multiplies the penalty terms.
            l1_ratio (float): The mixing parameter, with 0 <= l1_ratio <= 1.
        """
        self.model = ElasticNet(
            alpha=alpha, l1_ratio=l1_ratio, random_state=Config.SEED
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


class DualModel:
    """
    Orchestrates the two-stage Decoupled Residual Regression training.
    Stage 1: Predict Median FVC.
    Stage 2: Predict Uncertainty (MAD) from residuals.
    """

    def __init__(self, fvc_alpha=0.5, unc_alpha=0.1, unc_l1_ratio=0.5):
        seed_everything(Config.SEED)
        self.fvc_model = FVCPredictor(alpha=fvc_alpha)
        self.unc_model = UncertaintyPredictor(alpha=unc_alpha, l1_ratio=unc_l1_ratio)

    def fit(self, train_data, val_data=None):
        """
        Trains both the FVC and Uncertainty models.

        Args:
            train_data (dict): Dictionary containing training features and targets.
            val_data (dict, optional): Dictionary containing validation features and targets.
        """
        X_fvc = train_data["X_fvc"]
        y = train_data["y"]
        X_unc = train_data["X_unc"]

        # Stage 1: Train FVC Predictor
        print("Stage 1: Training FVC Predictor (Quantile Regression)...")
        self.fvc_model.fit(X_fvc, y)

        # Compute residuals on training set to train the uncertainty model
        y_pred_train = self.fvc_model.predict(X_fvc)
        residuals = np.abs(y - y_pred_train)

        # Stage 2: Train Uncertainty Predictor
        print("Stage 2: Training Uncertainty Predictor (Elastic Net)...")
        self.unc_model.fit(X_unc, residuals)

        # Validation
        if val_data:
            print("Evaluating on Validation Set...")
            score = self.evaluate(val_data)
            print(f"Validation Score: {score}")

    def predict(self, data):
        """
        Generates predictions for FVC and Confidence (Sigma).

        Args:
            data (dict): Dictionary containing feature matrices 'X_fvc' and 'X_unc'.

        Returns:
            tuple: (fvc_pred, sigma_pred)
        """
        X_fvc = data["X_fvc"]
        X_unc = data["X_unc"]

        # 1. Predict Median FVC
        fvc_pred = self.fvc_model.predict(X_fvc)

        # 2. Predict Expected MAD (Mean Absolute Deviation)
        mad_pred = self.unc_model.predict(X_unc)

        # Ensure MAD is non-negative
        mad_pred = np.maximum(mad_pred, 0)

        # 3. Convert MAD to Sigma
        # Analytically, for a Laplace distribution, Sigma = MAD * sqrt(2)
        sigma_pred = mad_pred * np.sqrt(2)

        return fvc_pred, sigma_pred

    def evaluate(self, val_data):
        """
        Evaluates the model using the modified Laplace Log Likelihood metric.
        """
        y_true = val_data["y"]
        y_pred, sigma_pred = self.predict(val_data)

        score = score_func(y_true, y_pred, sigma_pred)
        return score


def generate_submission(model, test_data):
    """
    Generates the submission file using the trained model and test data.

    Args:
        model (DualModel): Trained model instance.
        test_data (dict): Dictionary containing test features and identifiers.
    """
    print("Generating submission...")

    fvc_pred, sigma_pred = model.predict(test_data)
    patient_weeks = test_data["patient_week"]

    # Clip sigma at 70 as per metric definition for the submission file
    sigma_clipped = np.maximum(sigma_pred, Config.MIN_CONFIDENCE)

    submission_df = pd.DataFrame(
        {"Patient_Week": patient_weeks, "FVC": fvc_pred, "Confidence": sigma_clipped}
    )

    # Ensure output directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
