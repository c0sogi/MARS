import numpy as np
import pandas as pd
import os
import warnings
from sklearn.linear_model import QuantileRegressor, ElasticNet
from sklearn.utils import resample
from sklearn.metrics import mean_absolute_error

from library.config import (
    N_BAGS,
    QUANTILES,
    ELASTIC_L1_RATIO,
    ELASTIC_ALPHA,
    SEED,
    MIN_CONFIDENCE,
    MAX_ERROR,
    SUBMISSION_PATH,
    CACHE_DIR,
)

# Suppress warnings for cleaner output (e.g. convergence warnings from linear solvers)
warnings.filterwarnings("ignore")


class BaggedQuantileRegressor:
    """
    Ensemble of Linear Quantile Regressors trained on bootstrap samples.
    Targets the median (quantile=0.5) to minimize L1 loss, which is robust to outliers.
    """

    def __init__(self, n_estimators=N_BAGS, quantile=0.5, alpha=0.01, seed=SEED):
        self.n_estimators = n_estimators
        self.quantile = quantile
        self.alpha = alpha  # Regularization strength
        self.seed = seed
        self.models = []

    def fit(self, X, y):
        self.models = []
        rng = np.random.RandomState(self.seed)

        for i in range(self.n_estimators):
            # Bootstrap sampling with replacement
            # We use a different random state for each bag derived from the main seed
            bag_seed = rng.randint(0, 10000)
            X_sample, y_sample = resample(X, y, replace=True, random_state=bag_seed)

            # Initialize QuantileRegressor
            # solver='highs' is efficient for linear programming problems in recent sklearn versions
            model = QuantileRegressor(
                quantile=self.quantile, alpha=self.alpha, solver="highs"
            )
            try:
                model.fit(X_sample, y_sample)
            except Exception:
                # Fallback to interior-point if highs solver encounters issues
                model = QuantileRegressor(
                    quantile=self.quantile, alpha=self.alpha, solver="interior-point"
                )
                model.fit(X_sample, y_sample)

            self.models.append(model)

    def predict(self, X):
        # Aggregate predictions by averaging
        if not self.models:
            return np.zeros(X.shape[0])

        preds = np.zeros((X.shape[0], len(self.models)))
        for i, model in enumerate(self.models):
            preds[:, i] = model.predict(X)
        return np.mean(preds, axis=1)


class BaggedElasticNet:
    """
    Ensemble of Elastic Net Regressors trained on bootstrap samples.
    Used to predict the expected absolute error (MAD) based on features and time horizon.
    """

    def __init__(
        self,
        n_estimators=N_BAGS,
        alpha=ELASTIC_ALPHA,
        l1_ratio=ELASTIC_L1_RATIO,
        seed=SEED,
    ):
        self.n_estimators = n_estimators
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.seed = seed
        self.models = []

    def fit(self, X, y):
        self.models = []
        rng = np.random.RandomState(self.seed)

        for i in range(self.n_estimators):
            # Bootstrap sampling
            bag_seed = rng.randint(0, 10000)
            X_sample, y_sample = resample(X, y, replace=True, random_state=bag_seed)

            model = ElasticNet(
                alpha=self.alpha, l1_ratio=self.l1_ratio, random_state=bag_seed
            )
            model.fit(X_sample, y_sample)
            self.models.append(model)

    def predict(self, X):
        if not self.models:
            return np.zeros(X.shape[0])

        preds = np.zeros((X.shape[0], len(self.models)))
        for i, model in enumerate(self.models):
            preds[:, i] = model.predict(X)
        return np.mean(preds, axis=1)


def laplace_log_likelihood(y_true, y_pred, sigma):
    """
    Computes the modified Laplace Log Likelihood metric.
    """
    # Clip sigma to reflect approximate measurement uncertainty
    sigma_clipped = np.maximum(sigma, MIN_CONFIDENCE)
    # Clip error to avoid large penalties
    delta = np.minimum(np.abs(y_true - y_pred), MAX_ERROR)

    metric = -(np.sqrt(2) * delta) / sigma_clipped - np.log(np.sqrt(2) * sigma_clipped)
    return np.mean(metric)


def train_and_predict(
    X_fvc_train,
    y_train,
    X_unc_train,
    X_fvc_val,
    y_val,
    X_unc_val,
    X_fvc_test,
    X_unc_test,
    test_ids,
):
    """
    Orchestrates the training and prediction pipeline.

    Args:
        X_fvc_*: Features for the FVC prediction model (includes interactions).
        y_*: True FVC values.
        X_unc_*: Features for the Uncertainty model (includes time horizon).
        test_ids: Patient_Week identifiers for the submission.

    Returns:
        submission_df: DataFrame containing the predictions.
    """
    print("Initializing Bagged Variance-Weighted Quantile-Elastic Pipeline...")

    # ---------------------------------------------------------
    # Stage 1: FVC Ensemble Training
    # ---------------------------------------------------------
    print(f"Stage 1: Training FVC Ensemble ({N_BAGS} Linear Quantile Regressors)...")
    # We use a small alpha to ensure numerical stability while keeping the model mostly linear
    fvc_model = BaggedQuantileRegressor(
        n_estimators=N_BAGS, quantile=0.5, alpha=0.01, seed=SEED
    )
    fvc_model.fit(X_fvc_train, y_train)

    # Generate FVC predictions
    y_train_pred = fvc_model.predict(X_fvc_train)
    y_val_pred = fvc_model.predict(X_fvc_val)
    y_test_pred = fvc_model.predict(X_fvc_test)

    train_mae = mean_absolute_error(y_train, y_train_pred)
    val_mae = mean_absolute_error(y_val, y_val_pred)
    print(f"FVC MAE - Train: {train_mae}, Val: {val_mae}")

    # ---------------------------------------------------------
    # Stage 2: Residual Computation
    # ---------------------------------------------------------
    print("Stage 2: Computing Residuals for Uncertainty Target...")
    # The uncertainty model targets the absolute error of the FVC model
    train_residuals = np.abs(y_train - y_train_pred)

    # ---------------------------------------------------------
    # Stage 3: Uncertainty Ensemble Training
    # ---------------------------------------------------------
    print(
        f"Stage 3: Training Uncertainty Ensemble ({N_BAGS} Elastic Net Regressors)..."
    )
    unc_model = BaggedElasticNet(
        n_estimators=N_BAGS, alpha=ELASTIC_ALPHA, l1_ratio=ELASTIC_L1_RATIO, seed=SEED
    )
    unc_model.fit(X_unc_train, train_residuals)

    # Predict Mean Absolute Deviation (MAD)
    mad_train = unc_model.predict(X_unc_train)
    mad_val = unc_model.predict(X_unc_val)
    mad_test = unc_model.predict(X_unc_test)

    # Convert MAD to Sigma (Confidence) for Laplace Metric
    # Assuming Laplace distribution: sigma = MAD * sqrt(2)
    sigma_train = mad_train * np.sqrt(2)
    sigma_val = mad_val * np.sqrt(2)
    sigma_test = mad_test * np.sqrt(2)

    # ---------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------
    train_score = laplace_log_likelihood(y_train, y_train_pred, sigma_train)
    val_score = laplace_log_likelihood(y_val, y_val_pred, sigma_val)

    print(f"Laplace Log Likelihood - Train: {train_score}")
    print(f"Laplace Log Likelihood - Val: {val_score}")

    # ---------------------------------------------------------
    # Submission Generation
    # ---------------------------------------------------------
    print(f"Generating submission for {len(test_ids)} patient-weeks...")

    submission_df = pd.DataFrame(
        {"Patient_Week": test_ids, "FVC": y_test_pred, "Confidence": sigma_test}
    )

    # Ensure directory exists and save
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")

    return submission_df
