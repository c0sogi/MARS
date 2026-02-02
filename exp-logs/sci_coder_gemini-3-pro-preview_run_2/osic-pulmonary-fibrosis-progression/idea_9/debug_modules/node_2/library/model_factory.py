import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error
from library.config import Config


class QuantileGLMSystem:
    """
    Implements the Density-Aware Global-Local Quantile-GLM system.

    This system uses a decoupled residual regression approach:
    1. A Linear Quantile Regressor (q=0.5) predicts the median FVC.
    2. A Gamma GLM (Log Link) predicts the expected absolute error (uncertainty)
       based on the residuals of the FVC predictor.
    """

    def __init__(self):
        self.fvc_model = None
        self.unc_model = None
        self.fvc_results = None
        self.unc_results = None

    def fit(self, X_fvc, X_unc, y):
        """
        Trains the FVC and Uncertainty models sequentially.

        Args:
            X_fvc (np.ndarray): Feature matrix for FVC prediction.
            X_unc (np.ndarray): Feature matrix for Uncertainty prediction.
            y (np.ndarray): Target FVC values.
        """
        # -------------------------------------------------------
        # 1. Prepare Data
        # -------------------------------------------------------
        # Statsmodels requires an explicit constant column for the intercept
        X_fvc_sm = sm.add_constant(X_fvc, has_constant="add")
        X_unc_sm = sm.add_constant(X_unc, has_constant="add")

        # -------------------------------------------------------
        # 2. Train FVC Model (Quantile Regression, q=0.5)
        # -------------------------------------------------------
        print("Training FVC Linear Quantile Regressor (q=0.5)...")
        # QuantReg minimizes Sum(|y - Xb|) for q=0.5
        self.fvc_model = sm.QuantReg(y, X_fvc_sm)

        # Fit the model
        # Using a higher max_iter to ensure convergence on complex datasets
        try:
            self.fvc_results = self.fvc_model.fit(q=0.5, max_iter=2500, p_tol=1e-6)
        except Exception as e:
            print(
                f"Warning: QuantReg convergence issue ({e}). Retrying with relaxed constraints..."
            )
            self.fvc_results = self.fvc_model.fit(q=0.5, max_iter=5000)

        # -------------------------------------------------------
        # 3. Compute Residuals for Uncertainty Training
        # -------------------------------------------------------
        # Predict Median FVC on training set
        y_pred_train = self.fvc_results.predict(X_fvc_sm)

        # Calculate Absolute Residuals (MAE per sample)
        residuals = np.abs(y - y_pred_train)

        # Gamma GLM requires strictly positive targets.
        # We clamp residuals at a small epsilon (1e-6) to avoid numerical errors.
        residuals = np.maximum(residuals, 1e-6)

        # -------------------------------------------------------
        # 4. Train Uncertainty Model (Gamma GLM)
        # -------------------------------------------------------
        print("Training Uncertainty Gamma GLM...")
        # Family: Gamma, Link: Log
        # The Log link ensures that the predicted uncertainty is always positive.
        gamma_family = sm.families.Gamma(link=sm.families.links.Log())

        self.unc_model = sm.GLM(residuals, X_unc_sm, family=gamma_family)

        try:
            self.unc_results = self.unc_model.fit(maxiter=1000)
        except Exception as e:
            print(
                f"Warning: Gamma GLM convergence issue ({e}). Falling back to Gaussian Log-Link."
            )
            # Fallback: Gaussian with Log Link is more stable but still enforces positivity
            gaussian_log = sm.families.Gaussian(link=sm.families.links.Log())
            self.unc_model = sm.GLM(residuals, X_unc_sm, family=gaussian_log)
            self.unc_results = self.unc_model.fit(maxiter=1000)

        # -------------------------------------------------------
        # 5. Logging
        # -------------------------------------------------------
        print(f"FVC Model Pseudo R-squared: {self.fvc_results.prsquared}")
        print(f"Uncertainty Model Deviance: {self.unc_results.deviance}")

    def predict(self, X_fvc, X_unc):
        """
        Generates predictions for FVC and Uncertainty.

        Args:
            X_fvc (np.ndarray): Feature matrix for FVC.
            X_unc (np.ndarray): Feature matrix for Uncertainty.

        Returns:
            fvc_pred (np.ndarray): Predicted Median FVC.
            delta_pred (np.ndarray): Predicted Expected Absolute Error (MAE).
        """
        if self.fvc_results is None or self.unc_results is None:
            raise RuntimeError("Models must be trained before prediction.")

        # Handle empty input case safely
        if len(X_fvc) == 0:
            return np.array([]), np.array([])

        # Add constant for intercept (must match training structure)
        X_fvc_sm = sm.add_constant(X_fvc, has_constant="add")
        X_unc_sm = sm.add_constant(X_unc, has_constant="add")

        # Predict FVC (Median)
        fvc_pred = self.fvc_results.predict(X_fvc_sm)

        # Predict Uncertainty (Delta / Expected MAE)
        delta_pred = self.unc_results.predict(X_unc_sm)

        return fvc_pred, delta_pred

    def evaluate(self, X_fvc, X_unc, y):
        """
        Evaluates the model on a validation set using the competition metric.

        Metric: Modified Laplace Log Likelihood
        """
        print("\n--- Evaluation ---")
        pred_fvc, pred_delta = self.predict(X_fvc, X_unc)

        # 1. Convert Delta (MAE) to Sigma (Scale)
        # For a Laplace distribution, the optimal scale parameter sigma = MAE * sqrt(2)
        sigma = pred_delta * np.sqrt(2)

        # 2. Clip Sigma (as per metric definition)
        sigma_clipped = np.maximum(sigma, 70)

        # 3. Calculate Thresholded Error
        abs_err = np.abs(y - pred_fvc)
        delta_metric = np.minimum(abs_err, 1000)

        # 4. Compute Metric
        # metric = - (sqrt(2) * delta) / sigma - ln(sqrt(2) * sigma)
        term1 = (np.sqrt(2) * delta_metric) / sigma_clipped
        term2 = np.log(np.sqrt(2) * sigma_clipped)
        score = -term1 - term2

        avg_score = np.mean(score)
        mae = mean_absolute_error(y, pred_fvc)

        print(f"Validation Score: {avg_score}")
        print(f"Validation MAE: {mae}")

        return avg_score
