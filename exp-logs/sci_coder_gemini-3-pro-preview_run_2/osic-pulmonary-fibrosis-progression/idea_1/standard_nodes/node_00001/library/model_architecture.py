import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from library.config import (
    ELASTIC_NET_ALPHA,
    ELASTIC_NET_L1_RATIO,
    MAX_ITER,
    RANDOM_STATE,
    MIN_CONFIDENCE,
    SUBMISSION_PATH,
)
from library.metrics import laplace_log_likelihood


class DualModel:
    """
    Radiomics-Enhanced Dual Elastic Net.

    This architecture uses two parallel Linear Regression models with Elastic Net regularization:
    1. FVC Regressor: Predicts the target lung capacity (FVC).
    2. Uncertainty Regressor: Predicts the confidence (sigma) by learning the magnitude
       of the error (absolute residuals) from the FVC Regressor.
    """

    def __init__(
        self,
        alpha=ELASTIC_NET_ALPHA,
        l1_ratio=ELASTIC_NET_L1_RATIO,
        random_state=RANDOM_STATE,
    ):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.random_state = random_state

        # Model 1: Predicts FVC
        # ElasticNet minimizes the objective function: 1 / (2 * n_samples) * ||y - Xw||^2_2
        # + alpha * l1_ratio * ||w||_1 + 0.5 * alpha * (1 - l1_ratio) * ||w||^2_2
        self.fvc_model = ElasticNet(
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            max_iter=MAX_ITER,
            random_state=self.random_state,
        )

        # Model 2: Predicts Confidence (Sigma)
        self.sigma_model = ElasticNet(
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            max_iter=MAX_ITER,
            random_state=self.random_state,
        )

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
        sigma_pred = self.sigma_model.predict(X)

        # Clip Sigma to minimum threshold (70ml) as per task requirement
        # While ElasticNet might output negative values, confidence must be positive.
        # The metric also imposes a minimum of 70.
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
    print("Initializing Radiomics-Enhanced Dual Elastic Net Model...")
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
