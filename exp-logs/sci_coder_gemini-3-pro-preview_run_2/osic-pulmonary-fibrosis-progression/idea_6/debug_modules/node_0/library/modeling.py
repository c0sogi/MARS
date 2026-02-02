import os
import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor, GammaRegressor
from library.config import Config
from library.utils import calculate_metric
from library.data_processor import DataProcessor


class DualMomentGLM:
    """
    Implements the Dual-Moment GLM-Quantile Pipeline.
    Consists of:
    1. A Linear Quantile Regressor (q=0.5) for FVC prediction.
    2. A Gamma GLM (Log Link) for Uncertainty (Sigma) prediction.
    """

    def __init__(self):
        self.processor = DataProcessor()

        # Initialize FVC Predictor (Quantile Regressor)
        # Filter params to match QuantileRegressor constructor
        qr_params = {
            k: v
            for k, v in Config.QR_PARAMS.items()
            if k in ["quantile", "alpha", "solver", "fit_intercept"]
        }
        self.fvc_model = QuantileRegressor(**qr_params)

        # Initialize Uncertainty Predictor (Gamma Regressor)
        # GammaRegressor implies family=Gamma, link=Log.
        # It supports 'alpha' (L2 penalty). It does not support 'l1_ratio' natively in standard sklearn.
        glm_params_filtered = {
            k: v
            for k, v in Config.GLM_PARAMS.items()
            if k
            in ["alpha", "fit_intercept", "max_iter", "tol", "warm_start", "verbose"]
        }
        self.unc_model = GammaRegressor(**glm_params_filtered)

    def train(self, data):
        """
        Trains the decoupled regression models.
        """
        X_train_fvc = data["X_train_fvc"]
        y_train = data["y_train"]
        X_train_unc = data["X_train_unc"]

        print(
            f"Training FVC Predictor (QuantileRegressor q={Config.QR_PARAMS['quantile']})..."
        )
        self.fvc_model.fit(X_train_fvc, y_train)

        # Generate training predictions to compute residuals
        y_train_pred = self.fvc_model.predict(X_train_fvc)

        # Compute absolute residuals for uncertainty modeling
        # Add epsilon to ensure targets are strictly positive for Gamma distribution
        epsilon = 1e-6
        residuals = np.abs(y_train - y_train_pred) + epsilon

        print("Training Uncertainty Predictor (GammaRegressor)...")
        self.unc_model.fit(X_train_unc, residuals)

        # Validation
        self.validate(data)

    def validate(self, data):
        """
        Evaluates the model on the validation set.
        """
        X_val_fvc = data["X_val_fvc"]
        y_val = data["y_val"]
        X_val_unc = data["X_val_unc"]

        # Predict FVC (Median)
        y_val_pred = self.fvc_model.predict(X_val_fvc)

        # Predict Uncertainty (Delta -> Sigma)
        # The GLM predicts the Mean Absolute Deviation (Delta)
        # For Laplace distribution, Sigma = Delta * sqrt(2)
        delta_pred = self.unc_model.predict(X_val_unc)
        sigma_pred = delta_pred * np.sqrt(2)

        # Calculate Metric
        score = calculate_metric(y_val, y_val_pred, sigma_pred)
        print(f"Validation Metric Score: {score}")

    def predict_test(self, data):
        """
        Generates predictions for the test set and prepares submission.
        """
        X_test_fvc = data["X_test_fvc"]
        X_test_unc = data["X_test_unc"]
        test_ids = data["test_ids"]

        print("Generating test predictions...")

        # Predict FVC
        fvc_pred = self.fvc_model.predict(X_test_fvc)

        # Predict Confidence
        delta_pred = self.unc_model.predict(X_test_unc)
        sigma_pred = delta_pred * np.sqrt(2)

        # Create Submission DataFrame
        sub_df = pd.DataFrame(
            {"Patient_Week": test_ids, "FVC": fvc_pred, "Confidence": sigma_pred}
        )

        # Ensure FVC is integer (as per sample submission, though float is usually accepted,
        # but sample has int). Let's keep it float or round.
        # The metric function handles floats.
        # However, sample submission has FVC as int. Let's round for safety.
        # Confidence is also int in sample, but usually float is better for scoring.
        # We will keep them as is (floats) or round if strictly necessary.
        # The prompt says "FVC (int64)" in sample info.
        # We will round FVC, but keep Confidence as is (or round).
        # Usually, higher precision is better for Confidence.

        # Saving
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    def run(self):
        """
        Main execution method.
        """
        # 1. Load Data (Cached or Processed)
        data = self.processor.process(load_cached_data=True)

        # 2. Train Models
        self.train(data)

        # 3. Generate Submission
        self.predict_test(data)
