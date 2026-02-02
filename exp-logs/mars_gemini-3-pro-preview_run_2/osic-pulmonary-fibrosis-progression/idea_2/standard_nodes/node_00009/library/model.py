import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from library.config import Config
from library.metrics import laplace_log_likelihood


class DualElasticNet:
    """
    Implements the Deep-Feature Varying-Coefficient Elastic Net strategy.

    This architecture consists of two distinct linear models:
    1. Primary Model (FVC Predictor):
       - Uses a Varying-Coefficient formulation: FVC(t) = Intercept(X) + Slope(X) * t
       - Input: Augmented feature matrix [X_static, t, X_static * t]
       - Target: FVC

    2. Secondary Model (Uncertainty Estimator):
       - Predicts the confidence (sigma) based on the expected error magnitude.
       - Input: Static features only (X_static)
       - Target: Absolute residuals of the Primary Model (|y_true - y_pred|)
    """

    def __init__(self):
        # Initialize the Primary FVC Predictor
        self.fvc_model = ElasticNet(
            alpha=Config.ENET_ALPHA,
            l1_ratio=Config.ENET_L1_RATIO,
            random_state=Config.SEED,
            max_iter=10000,  # Increased max_iter to ensure convergence
        )

        # Initialize the Secondary Uncertainty Estimator
        self.sigma_model = ElasticNet(
            alpha=Config.SIGMA_ALPHA,
            l1_ratio=Config.SIGMA_L1_RATIO,
            random_state=Config.SEED,
            max_iter=10000,
        )

    def _extract_static_features(self, X):
        """
        Helper method to extract the static feature block from the full interaction matrix.

        The input X is constructed as: [X_static, t, X_interactions]
        Where:
            - X_static has K columns
            - t has 1 column
            - X_interactions has K columns (X_static * t)
        Total columns = 2K + 1.

        Args:
            X (np.ndarray): The full feature matrix.

        Returns:
            np.ndarray: The subset of X corresponding to X_static.
        """
        n_cols = X.shape[1]
        # Calculate K (number of static features)
        n_static = (n_cols - 1) // 2

        # Return the first K columns
        return X[:, :n_static]

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Trains the dual-model system.

        Args:
            X_train (np.ndarray): Training features (including interactions).
            y_train (np.ndarray): Training targets (FVC).
            X_val (np.ndarray, optional): Validation features.
            y_val (np.ndarray, optional): Validation targets.
        """
        # -------------------------------------------------------
        # Stage 1: Train Primary FVC Model
        # -------------------------------------------------------
        print("Training Primary FVC Model (ElasticNet)...")
        self.fvc_model.fit(X_train, y_train)

        # -------------------------------------------------------
        # Stage 2: Train Uncertainty Model
        # -------------------------------------------------------
        print("Training Secondary Uncertainty Model (ElasticNet)...")

        # Generate predictions on training set to compute residuals
        train_preds = self.fvc_model.predict(X_train)

        # Compute absolute error (proxy for uncertainty)
        train_residuals = np.abs(y_train - train_preds)

        # Extract static features for the uncertainty model
        # We assume uncertainty is a property of the patient's condition (static),
        # rather than the specific time-point interaction.
        X_train_static = self._extract_static_features(X_train)

        # Fit the sigma model
        self.sigma_model.fit(X_train_static, train_residuals)

        # -------------------------------------------------------
        # Stage 3: Validation
        # -------------------------------------------------------
        if X_val is not None and y_val is not None:
            print("Evaluating on Validation Set...")
            val_fvc_pred, val_sigma_pred = self.predict(X_val)

            # Compute Metric
            score = laplace_log_likelihood(y_val, val_fvc_pred, val_sigma_pred)
            print(f"Validation Laplace Log Likelihood: {score}")

    def predict(self, X):
        """
        Generates predictions for FVC and Confidence.

        Args:
            X (np.ndarray): Feature matrix.

        Returns:
            tuple: (fvc_predictions, sigma_predictions)
        """
        # Predict FVC using the full feature set (Static + Time + Interactions)
        fvc_pred = self.fvc_model.predict(X)

        # Predict Sigma using only the Static features
        X_static = self._extract_static_features(X)
        sigma_pred = self.sigma_model.predict(X_static)

        # Scale the predicted residuals to optimal sigma for Laplace Likelihood
        # Optimal sigma = sqrt(2) * |Error|
        sigma_pred = sigma_pred * np.sqrt(2)

        return fvc_pred, sigma_pred


def generate_submission(model, X_test, df_test):
    """
    Runs inference on the test set and saves the submission file.

    Args:
        model (DualElasticNet): The trained model instance.
        X_test (np.ndarray): Test set features.
        df_test (pd.DataFrame): Test set metadata containing 'Patient_Week'.
    """
    print("Generating submission...")

    # 1. Generate Predictions
    fvc_pred, sigma_pred = model.predict(X_test)

    # 2. Post-process Confidence
    # Ensure confidence is not below the metric's clipping threshold
    # to provide realistic estimates in the CSV.
    sigma_pred = np.maximum(sigma_pred, Config.MIN_CONFIDENCE)

    # 3. Construct Submission DataFrame
    # df_test is guaranteed to be aligned with X_test by data_processor.py
    submission = df_test[["Patient_Week"]].copy()
    submission["FVC"] = fvc_pred
    submission["Confidence"] = sigma_pred

    # 4. Save to Disk
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")

    # Print sample for verification
    print("Sample Submission Rows:")
    print(submission.head())
