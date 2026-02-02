import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from library.config import (
    SUBMISSION_PATH,
    SEED,
    N_BAGS,
    ELASTIC_ALPHA,
    ELASTIC_L1_RATIO,
)
from library.dicom_feature_extractor import run_extraction
from library.data_processor import process_data
from library.model_zoo import (
    BaggedQuantileRegressor,
    BaggedElasticNet,
    laplace_log_likelihood,
)


def train_pipeline(load_cached_data=True):
    """
    Orchestrates the training process:
    1. Loads and processes data (features + tabular).
    2. Trains the FVC Ensemble (Quantile Regression).
    3. Computes residuals and trains the Uncertainty Ensemble (ElasticNet).
    4. Evaluates on the validation set.

    Returns:
        fvc_model: Trained BaggedQuantileRegressor
        unc_model: Trained BaggedElasticNet
        X_fvc_test, X_unc_test, test_ids: Processed test data for inference
    """
    print("Starting Train Pipeline...")

    # 1. Feature Extraction (Images)
    # run_extraction handles caching internally via load_cached_data
    train_feats, val_feats, test_feats = run_extraction(
        load_cached_data=load_cached_data
    )

    # 2. Data Processing (Tabular + PCA)
    # process_data handles caching internally via load_cached_data
    data_tuple = process_data(
        train_feats, val_feats, test_feats, load_cached_data=load_cached_data
    )
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
    ) = data_tuple

    # 3. Train FVC Model (Bagged Quantile Regression)
    print(f"Training FVC Ensemble with {N_BAGS} estimators...")
    fvc_model = BaggedQuantileRegressor(
        n_estimators=N_BAGS, quantile=0.5, alpha=0.01, seed=SEED
    )
    fvc_model.fit(X_fvc_train, y_train)

    # 4. Train Uncertainty Model (Bagged ElasticNet on Residuals)
    print("Computing residuals and training Uncertainty Ensemble...")
    # Predict on train to get residuals
    y_train_pred = fvc_model.predict(X_fvc_train)
    train_residuals = np.abs(y_train - y_train_pred)

    unc_model = BaggedElasticNet(
        n_estimators=N_BAGS, alpha=ELASTIC_ALPHA, l1_ratio=ELASTIC_L1_RATIO, seed=SEED
    )
    unc_model.fit(X_unc_train, train_residuals)

    # 5. Validation Evaluation
    print("Evaluating on Validation Set...")
    # FVC Predictions
    y_val_pred = fvc_model.predict(X_fvc_val)
    val_mae = mean_absolute_error(y_val, y_val_pred)

    # Uncertainty Predictions (MAD -> Sigma)
    mad_val = unc_model.predict(X_unc_val)
    sigma_val = mad_val * np.sqrt(2)

    # Metric Calculation
    val_score = laplace_log_likelihood(y_val, y_val_pred, sigma_val)

    # Also calc train score for reference
    mad_train = unc_model.predict(X_unc_train)
    sigma_train = mad_train * np.sqrt(2)
    train_score = laplace_log_likelihood(y_train, y_train_pred, sigma_train)

    print(f"Validation MAE (FVC): {val_mae}")
    print(f"Train Laplace Log Likelihood: {train_score}")
    print(f"Validation Laplace Log Likelihood: {val_score}")

    return fvc_model, unc_model, X_fvc_test, X_unc_test, test_ids


def predict_pipeline(fvc_model, unc_model, X_fvc_test, X_unc_test, test_ids):
    """
    Orchestrates the inference process:
    1. Generates FVC predictions.
    2. Generates Uncertainty predictions (MAD) and scales to Sigma.
    3. Formats and saves the submission file.
    """
    print(f"Starting Predict Pipeline for {len(test_ids)} samples...")

    # 1. Predict FVC
    y_test_pred = fvc_model.predict(X_fvc_test)

    # 2. Predict Uncertainty
    mad_test = unc_model.predict(X_unc_test)
    # Convert Mean Absolute Deviation to Sigma (Confidence)
    # For Laplace distribution, sigma = MAD * sqrt(2)
    sigma_test = mad_test * np.sqrt(2)

    # 3. Generate Submission
    submission_df = pd.DataFrame(
        {"Patient_Week": test_ids, "FVC": y_test_pred, "Confidence": sigma_test}
    )

    # Save
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")

    return submission_df


def run_workflow(load_cached_data=True):
    """
    Convenience function to run the full end-to-end workflow.
    """
    # Run Training
    fvc_model, unc_model, X_fvc_test, X_unc_test, test_ids = train_pipeline(
        load_cached_data
    )

    # Run Inference
    predict_pipeline(fvc_model, unc_model, X_fvc_test, X_unc_test, test_ids)
