import os
import sys
import numpy as np
import pandas as pd
import random
import torch
import warnings

# Add current directory to path to ensure library imports work
sys.path.append(os.getcwd())

from library.config import SEED, SUBMISSION_PATH
from library.dicom_feature_extractor import run_extraction
from library.data_processor import process_data
from library.model_zoo import (
    BaggedQuantileRegressor,
    BaggedElasticNet,
    laplace_log_likelihood,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    # 1. Setup
    set_seed(SEED)
    warnings.filterwarnings("ignore")

    # 2. Data Loading & Processing
    # We use load_cached_data=True to leverage any existing artifacts and speed up execution
    train_feats, val_feats, test_feats = run_extraction(load_cached_data=True)

    (
        X_fvc_train,
        y_train,
        X_unc_train,
        X_fvc_val,
        y_val,
        X_unc_val,
        X_fvc_test,
        X_unc_test,
        test_ids,
    ) = process_data(train_feats, val_feats, test_feats, load_cached_data=True)

    # 3. Model Training
    # FVC Model: Bagged Quantile Regressor (Target: Median)
    # This model predicts the central tendency of the lung capacity
    fvc_model = BaggedQuantileRegressor(seed=SEED)
    fvc_model.fit(X_fvc_train, y_train)

    # Compute residuals for Uncertainty Model
    # We predict on the training set to see how much the model deviates
    y_train_pred = fvc_model.predict(X_fvc_train)
    train_residuals = np.abs(y_train - y_train_pred)

    # Uncertainty Model: Bagged ElasticNet (Target: Absolute Residuals)
    # This model predicts the expected error (MAD) based on features and time horizon
    unc_model = BaggedElasticNet(seed=SEED)
    unc_model.fit(X_unc_train, train_residuals)

    # 4. Validation & Metric
    # Predict FVC on validation set
    y_val_pred = fvc_model.predict(X_fvc_val)

    # Predict Uncertainty (MAD) on validation set
    mad_val = unc_model.predict(X_unc_val)

    # Convert MAD to Sigma for Laplace Metric
    # For a Laplace distribution, sigma = MAD * sqrt(2)
    sigma_val = mad_val * np.sqrt(2)

    # Compute Metric
    val_metric = laplace_log_likelihood(y_val, y_val_pred, sigma_val)

    # REQUIRED OUTPUT
    print(f"Final Validation Metric: {val_metric}")

    # 5. Failure Analysis
    print("Failure Analysis (Top 5 Feature Correlations with Absolute Error):")
    val_errors = np.abs(y_val - y_val_pred)

    # Correlate with X_unc_val features (Base features + Horizon)
    # This helps identify if errors are driven by specific clinical features or time duration
    correlations = []
    n_features = X_unc_val.shape[1]

    for i in range(n_features):
        feature_col = X_unc_val[:, i]
        # Avoid division by zero in correlation if feature is constant
        if np.std(feature_col) > 1e-9:
            corr = np.corrcoef(feature_col, val_errors)[0, 1]
            correlations.append((i, corr))
        else:
            correlations.append((i, 0.0))

    # Sort by absolute correlation to find strongest associations
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    for idx, corr in correlations[:5]:
        print(f"Feature Index {idx}: Correlation {corr}")

    # 6. Submission Generation
    # Threshold logic: Metric must be > -6.805292148096688
    # Note: Since metric is negative, "higher" means closer to 0 (e.g., -6.5 > -6.8)
    THRESHOLD = -6.805292148096688

    if val_metric > THRESHOLD:
        # Predict on Test Set
        y_test_pred = fvc_model.predict(X_fvc_test)
        mad_test = unc_model.predict(X_unc_test)
        sigma_test = mad_test * np.sqrt(2)

        submission_df = pd.DataFrame(
            {"Patient_Week": test_ids, "FVC": y_test_pred, "Confidence": sigma_test}
        )

        # Ensure directory exists
        os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

        # Save Submission
        submission_df.to_csv(SUBMISSION_PATH, index=False)
        print(f"Submission saved to {SUBMISSION_PATH}")
    else:
        print(
            f"Validation metric {val_metric} is not higher than threshold {THRESHOLD}. No submission generated."
        )


if __name__ == "__main__":
    main()
