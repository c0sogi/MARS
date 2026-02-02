import os
import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor, ElasticNet
from sklearn.preprocessing import StandardScaler
from library.config import Config, seed_everything

# Ensure reproducibility
seed_everything(Config.SEED)


def calculate_metric(y_true, y_pred, sigma):
    """
    Computes the modified Laplace Log Likelihood metric.

    metric = - (sqrt(2) * Delta / sigma_clipped) - ln(sqrt(2) * sigma_clipped)
    where Delta = min(|y_true - y_pred|, 1000)
    and sigma_clipped = max(sigma, 70)
    """
    sigma_clipped = np.maximum(sigma, Config.SIGMA_MIN)
    delta = np.abs(y_true - y_pred)
    delta = np.minimum(delta, Config.MAX_ERROR)

    metric = -(np.sqrt(2) * delta) / sigma_clipped - np.log(np.sqrt(2) * sigma_clipped)
    return np.mean(metric)


class FVCRegressor:
    """
    Linear Quantile Regressor (Median) with Interaction Terms.
    Predicts the central tendency of FVC.
    """

    def __init__(self):
        # Quantile=0.5 minimizes Mean Absolute Error (Median Regression)
        # alpha=0.0 or small value for L2 regularization.
        # solver='highs' is faster for linear programming in newer sklearn versions.
        # We use alpha=0.1 (L1 penalty) to prevent coefficient explosion when P > N.
        self.model = QuantileRegressor(
            quantile=Config.QUANTILE, alpha=0.1, solver="highs"
        )
        self.scaler = StandardScaler()
        self.y_scaler = StandardScaler()

    def _prepare_features(self, X_pca, relative_weeks, is_fitting=False):
        """
        Creates feature matrix: [X_pca, relative_weeks, X_pca * relative_weeks]
        """
        # Reshape weeks for broadcasting
        w = relative_weeks.reshape(-1, 1)

        # Interaction terms: PCA components * Weeks
        # This allows the slope of decline to vary based on image features
        interactions = X_pca * w

        # Concatenate: PCA (intercepts), Weeks (global slope), Interactions (patient-specific slope adjustments)
        X_combined = np.hstack([X_pca, w, interactions])

        # Scale features to help convergence of linear solvers
        if is_fitting:
            X_scaled = self.scaler.fit_transform(X_combined)
        else:
            X_scaled = self.scaler.transform(X_combined)

        return X_scaled

    def fit(self, X_pca, relative_weeks, y):
        X_features = self._prepare_features(X_pca, relative_weeks, is_fitting=True)
        print(f"Training FVC Regressor on shape {X_features.shape}...")

        # Scale target variable to ensure alpha=0.1 is effective
        y = y.reshape(-1, 1)
        y_scaled = self.y_scaler.fit_transform(y).ravel()

        self.model.fit(X_features, y_scaled)

    def predict(self, X_pca, relative_weeks):
        X_features = self._prepare_features(X_pca, relative_weeks, is_fitting=False)
        y_pred_scaled = self.model.predict(X_features)

        # Inverse scale predictions to original domain
        y_pred = self.y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

        return y_pred


class UncertaintyRegressor:
    """
    Elastic Net Regressor.
    Predicts the Mean Absolute Deviation (MAD) of the residuals.
    """

    def __init__(self):
        self.model = ElasticNet(
            alpha=Config.ENET_ALPHA,
            l1_ratio=Config.ENET_L1_RATIO,
            random_state=Config.SEED,
        )
        self.scaler = StandardScaler()

    def _prepare_features(self, X_pca, relative_weeks, is_fitting=False):
        """
        Creates feature matrix: [X_pca, abs(relative_weeks)]
        We use absolute time horizon because uncertainty grows with distance from baseline.
        """
        # Horizon = Absolute value of relative weeks
        horizon = np.abs(relative_weeks).reshape(-1, 1)

        X_combined = np.hstack([X_pca, horizon])

        if is_fitting:
            X_scaled = self.scaler.fit_transform(X_combined)
        else:
            X_scaled = self.scaler.transform(X_combined)

        return X_scaled

    def fit(self, X_pca, relative_weeks, y_residuals):
        X_features = self._prepare_features(X_pca, relative_weeks, is_fitting=True)
        print(f"Training Uncertainty Regressor on shape {X_features.shape}...")
        self.model.fit(X_features, y_residuals)

    def predict(self, X_pca, relative_weeks):
        X_features = self._prepare_features(X_pca, relative_weeks, is_fitting=False)
        # Predict MAD. Ensure non-negative.
        pred = self.model.predict(X_features)
        return np.maximum(pred, 0)


def run_modeling(data_dict):
    """
    Main execution function for the modeling phase.
    Args:
        data_dict: Dictionary containing tuples of (DataFrame, PCA_Features) for 'train', 'val', 'test'.
    """
    print("\n=== Starting Modeling Phase ===")

    # Unpack Data
    train_df, X_train_pca = data_dict["train"]
    val_df, X_val_pca = data_dict["val"]
    test_df, X_test_pca = data_dict["test"]

    # --- 1. Preprocessing: Calculate Relative Weeks ---
    # For Train/Val: 'Weeks' is already relative to baseline (per dataset description)
    train_weeks = train_df["Weeks"].values.astype(np.float32)
    val_weeks = val_df["Weeks"].values.astype(np.float32)

    # For Test: 'Weeks' is the target week, 'Baseline_Weeks' is the CT week.
    # Relative = Target - Baseline
    test_weeks_raw = test_df["Weeks"].values.astype(np.float32)
    test_baseline_weeks = test_df["Baseline_Weeks"].values.astype(np.float32)
    test_weeks = test_weeks_raw - test_baseline_weeks

    y_train = train_df["FVC"].values.astype(np.float32)
    y_val = val_df["FVC"].values.astype(np.float32)

    # --- 2. Train FVC Model (Median Regression) ---
    fvc_model = FVCRegressor()
    fvc_model.fit(X_train_pca, train_weeks, y_train)

    # --- 3. Train Uncertainty Model (Residual Regression) ---
    # Generate in-sample predictions to get residuals
    train_preds = fvc_model.predict(X_train_pca, train_weeks)
    train_residuals = np.abs(y_train - train_preds)

    unc_model = UncertaintyRegressor()
    unc_model.fit(X_train_pca, train_weeks, train_residuals)

    # --- 4. Validation ---
    print("\n--- Validation ---")
    val_preds_fvc = fvc_model.predict(X_val_pca, val_weeks)
    val_preds_mad = unc_model.predict(X_val_pca, val_weeks)

    # Convert MAD to Sigma (Analytical scaling for Laplace)
    # Sigma = MAD * sqrt(2)
    val_preds_sigma = val_preds_mad * np.sqrt(2)

    score = calculate_metric(y_val, val_preds_fvc, val_preds_sigma)
    print(f"Validation Laplace Log Likelihood: {score}")

    # Additional Metrics
    mae = np.mean(np.abs(y_val - val_preds_fvc))
    print(f"Validation MAE: {mae}")
    print(f"Mean Predicted Sigma: {np.mean(val_preds_sigma)}")

    # --- 5. Inference on Test Set ---
    print("\n--- Inference ---")
    test_preds_fvc = fvc_model.predict(X_test_pca, test_weeks)
    test_preds_mad = unc_model.predict(X_test_pca, test_weeks)

    # Convert to Sigma
    test_preds_sigma = test_preds_mad * np.sqrt(2)

    # Apply Clipping logic for submission
    # Note: The metric calculation handles clipping internally, but the submission file
    # expects raw values. The instructions say "confidence values are clipped at 70 ml
    # to reflect the approximate measurement uncertainty".
    # Usually, we submit the raw confidence, and the scoring system clips it.
    # However, the prompt says "Metric... sigma_clipped = max(sigma, 70)".
    # And "Submission Format... Confidence - a confidence value".
    # To be safe and consistent with the metric, we can clip here, or submit the raw value.
    # Given the prompt explicitly defines the metric clipping, submitting values < 70
    # will just be treated as 70. Submitting 70 is safe.
    test_preds_sigma = np.maximum(test_preds_sigma, Config.SIGMA_MIN)

    # --- 6. Create Submission File ---
    submission = pd.DataFrame(
        {
            "Patient_Week": test_df["Patient_Week"],
            "FVC": test_preds_fvc,
            "Confidence": test_preds_sigma,
        }
    )

    # Ensure FVC is integer (as per sample submission)
    submission["FVC"] = submission["FVC"].astype(int)
    # Confidence can be float or int, usually float is fine, but sample has int.
    # Let's keep it as float or round? Sample has 100 (int).
    # The prompt says "Confidence - a confidence value... (also has units of ml)".
    # Let's round Confidence to be clean.
    submission["Confidence"] = submission["Confidence"].round().astype(int)

    # Save
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(submission.head())
