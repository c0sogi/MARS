import os
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Suppress warnings
warnings.filterwarnings("ignore")

# Import library components
from library.config import (
    RANDOM_STATE,
    ENSEMBLE_WEIGHTS,
    SUBMISSION_FILE_PATH,
    NUMERIC_FEATURES,
    MLP_PARAMS,
)
from library.utils import set_seed
from library.data_loader import load_data
from library.feature_engineering import FeaturePipeline
from library.rf_learner import RFPredictor
from library.mlp_learner import MLPTrainer


def run():
    # 1. Setup and Configuration
    set_seed(RANDOM_STATE)

    # 2. Data Loading
    # Using load_cached_data=True to utilize pre-processed Parquet files if available
    train_df, val_df, test_df = load_data(load_cached_data=True)

    # Extract targets
    y_train = train_df["requester_received_pizza"].values
    y_val = val_df["requester_received_pizza"].values

    # 3. Feature Engineering
    # This step generates dictionary containing features for both RF and MLP streams
    print("Running Feature Engineering Pipeline...")
    pipeline = FeaturePipeline()
    features = pipeline.fit_transform(train_df, val_df, test_df, load_cached_data=True)

    # 4. Model Training

    # --- Stream A: Random Forest ---
    print("Training Random Forest (Stream A)...")
    rf_model = RFPredictor()
    rf_model.train(features["train"]["rf"], y_train, features["val"]["rf"], y_val)

    # --- Stream B: Skip-Gated MLP ---
    print("Training Skip-Gated MLP (Stream B)...")
    mlp_model = MLPTrainer()
    # We use the default parameters defined in config, but ensure we pass the validation set
    # for early stopping and scheduler updates.
    mlp_model.train(features["train"]["mlp"], y_train, features["val"]["mlp"], y_val)

    # 5. Validation and Ensemble
    print("Evaluating Ensemble...")

    # Generate probabilities
    rf_val_probs = rf_model.predict(features["val"]["rf"])
    mlp_val_probs = mlp_model.predict(features["val"]["mlp"])

    # Weighted Average Ensemble
    val_preds = (ENSEMBLE_WEIGHTS["rf"] * rf_val_probs) + (
        ENSEMBLE_WEIGHTS["mlp"] * mlp_val_probs
    )

    # Calculate Metric
    final_auc = roc_auc_score(y_val, val_preds)
    print(f"Final Validation Metric: {final_auc}")

    # 6. Failure Analysis
    print("Performing Failure Analysis...")
    # Calculate absolute error
    errors = np.abs(y_val - val_preds)

    # Prepare analysis dataframe with numeric features from validation set
    # We select only the numeric features used in modeling to see which correlate with error
    analysis_df = val_df[NUMERIC_FEATURES].copy()

    # Simple imputation for analysis purposes only (to handle any potential NaNs in raw data)
    analysis_df = analysis_df.fillna(analysis_df.median())

    # Add error column
    analysis_df["error_magnitude"] = errors

    # Compute correlations
    correlations = analysis_df.corr()["error_magnitude"].drop("error_magnitude")

    # Sort by absolute correlation to find strongest signals (positive or negative)
    sorted_corrs = correlations.abs().sort_values(ascending=False)

    print("Top 5 Features correlated with Error Magnitude:")
    for feature_name in sorted_corrs.head(5).index:
        corr_val = correlations[feature_name]
        print(f"{feature_name}: {corr_val:.4f}")

    # 7. Submission Generation
    threshold = 0.7135451153926904

    if final_auc > threshold:
        print(
            f"Validation AUC ({final_auc}) exceeds threshold ({threshold}). Generating submission..."
        )

        # Predict on Test Set
        rf_test_probs = rf_model.predict(features["test"]["rf"])
        mlp_test_probs = mlp_model.predict(features["test"]["mlp"])

        # Ensemble
        test_preds = (ENSEMBLE_WEIGHTS["rf"] * rf_test_probs) + (
            ENSEMBLE_WEIGHTS["mlp"] * mlp_test_probs
        )

        # Create Submission DataFrame
        submission = pd.DataFrame(
            {
                "request_id": test_df["request_id"],
                "requester_received_pizza": test_preds,
            }
        )

        # Save
        os.makedirs(os.path.dirname(SUBMISSION_FILE_PATH), exist_ok=True)
        submission.to_csv(SUBMISSION_FILE_PATH, index=False)
        print(f"Submission saved to {SUBMISSION_FILE_PATH}")

    else:
        print(
            f"Validation AUC ({final_auc}) does not exceed threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    run()
