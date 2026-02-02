import sys
import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

# Ensure library modules are importable
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed, load_object
from library.trainer import run_cv_training
from library.inference import generate_submission
from library.data_loader import DataLoader


def main():
    # Set reproducibility
    set_seed(Config.SEED)

    # ---------------------------------------------------------
    # 1. Training Phase
    # ---------------------------------------------------------
    # The trainer runs 5-fold CV on the training set (train.csv)
    # and saves the fitted pipelines to Config.WORKING_DIR
    print("Starting Training Phase...")
    run_cv_training()

    # ---------------------------------------------------------
    # 2. Validation Phase
    # ---------------------------------------------------------
    print("\nStarting Validation Phase...")

    # Load Hold-out Validation Set
    loader = DataLoader()
    # load_merged_data handles reading metadata and raw json
    df_val = loader.load_merged_data(split="val")

    # Extract Target
    y_val = df_val["requester_received_pizza"].values

    # Prepare for Ensemble Inference
    fold_preds = []

    # Use torch.no_grad() to save memory/compute during SBERT encoding
    # Although SBERT is inside the pipeline, this context manager affects torch ops
    with torch.no_grad():
        for fold in range(Config.N_FOLDS):
            model_path = os.path.join(
                Config.WORKING_DIR, f"fold_{fold}_pipeline.joblib"
            )

            if not os.path.exists(model_path):
                print(f"Warning: Model for fold {fold} not found at {model_path}")
                continue

            # Load pipeline
            # This loads the full pipeline including SBERT, TF-IDF, and the Classifier
            pipeline = load_object(model_path)

            # Predict
            # Pipeline expects the full DataFrame as input
            # predict_proba returns (n_samples, 2), we take column 1 (probability of success)
            preds = pipeline.predict_proba(df_val)[:, 1]
            fold_preds.append(preds)

    if not fold_preds:
        raise RuntimeError("No models were loaded for validation.")

    # Average predictions (Ensemble of Ensembles)
    avg_preds = np.mean(fold_preds, axis=0)

    # Compute Metric
    val_score = roc_auc_score(y_val, avg_preds)

    # PRINT REQUIRED METRIC
    print(f"Final Validation Metric: {val_score}")

    # ---------------------------------------------------------
    # 3. Failure Analysis
    # ---------------------------------------------------------
    print("\nPerforming Failure Analysis...")

    # Compute error magnitude
    errors = np.abs(y_val - avg_preds)

    # Get numeric features from validation dataframe for correlation
    # We exclude the target and any non-numeric columns
    numeric_df = df_val.select_dtypes(include=[np.number])
    if "requester_received_pizza" in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=["requester_received_pizza"])

    # Calculate correlations
    feature_corrs = {}
    for col in numeric_df.columns:
        # Fill NaNs just in case
        feat_values = numeric_df[col].fillna(0).values

        # Avoid constant columns to prevent warnings
        if np.std(feat_values) == 0:
            continue

        corr = np.corrcoef(feat_values, errors)[0, 1]
        if not np.isnan(corr):
            feature_corrs[col] = corr

    # Sort by absolute correlation
    sorted_corrs = sorted(feature_corrs.items(), key=lambda x: abs(x[1]), reverse=True)

    print("Top Correlations between Error and Features:")
    for feat, corr in sorted_corrs[:5]:
        print(f"  {feat}: {corr:.6f}")

    # ---------------------------------------------------------
    # 4. Submission
    # ---------------------------------------------------------
    TARGET_THRESHOLD = 0.713561265524314

    if val_score > TARGET_THRESHOLD:
        print(
            f"\nValidation Score ({val_score}) exceeds threshold ({TARGET_THRESHOLD})."
        )
        print("Generating Submission...")

        # Run inference on test set
        # We wrap in no_grad for efficiency
        with torch.no_grad():
            generate_submission()

    else:
        print(
            f"\nValidation Score ({val_score}) does not exceed threshold ({TARGET_THRESHOLD})."
        )
        print("Submission generation skipped.")


if __name__ == "__main__":
    main()
