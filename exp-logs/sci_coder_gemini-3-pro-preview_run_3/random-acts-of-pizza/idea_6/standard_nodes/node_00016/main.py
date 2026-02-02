import sys
import os
import numpy as np
import pandas as pd
import torch

# Ensure the current directory is in the path for module imports
sys.path.append(os.getcwd())

import library.config as config
from library.utils import set_seed, calculate_auc, save_submission, timer
from library.data_loader import load_data
from library.feature_pipeline import generate_features
from library.stacking_engine import TriViewStackingEnsemble


def main():
    # 1. Setup and Reproducibility
    set_seed(config.SEED)

    # Detect device for any potential PyTorch operations (e.g., SBERT)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Runtime Device: {device}")

    # 2. Data Loading
    # Load data using the provided loader. We use cached data if available for speed.
    # We use the full dataset (debug=False) to maximize performance.
    print("Loading datasets...")
    X_train, y_train, X_val, y_val, X_test, test_ids = load_data(
        load_cached_data=True, debug=False
    )

    # 3. Feature Generation
    # Transform raw data into Lexical, Semantic, and Community feature views.
    print("Generating feature views...")
    train_feats, val_feats, test_feats = generate_features(
        X_train, X_val, X_test, load_cached_data=True
    )

    # 4. Model Training
    # Initialize and train the Stacking Ensemble
    print("Training Tri-View Stacking Ensemble...")
    ensemble = TriViewStackingEnsemble()

    with timer("Ensemble Training"):
        ensemble.fit(train_feats, y_train)

    # 5. Validation Assessment
    print("\nPerforming Validation Inference...")
    # Using torch.no_grad() is good practice for inference to save memory/compute,
    # though the ensemble prediction here is primarily CPU-bound (sklearn/xgb).
    with torch.no_grad():
        val_probs = ensemble.predict_proba(val_feats)

    # Calculate and Print Metric
    # The calculate_auc function prints the score, but we also need to print
    # the specific format required by the task.
    auc_score = calculate_auc(y_val, val_probs, label="Hold-out Validation")
    print(f"Final Validation Metric: {auc_score}")

    # 6. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate error magnitude (absolute difference between truth and probability)
    errors = np.abs(y_val - val_probs)

    # Identify numerical columns in the validation set for correlation analysis
    # We exclude ID and target columns
    numeric_cols = X_val.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = [
        "request_id",
        "requester_received_pizza",
        "unix_timestamp_of_request",
        "unix_timestamp_of_request_utc",
    ]
    numeric_cols = [c for c in numeric_cols if c not in exclude_cols]

    correlations = {}
    for col in numeric_cols:
        # Ensure we have valid data for correlation
        if X_val[col].nunique() > 1:
            # Impute any remaining NaNs with median for this analysis
            feat_values = X_val[col].fillna(X_val[col].median())
            # Calculate correlation with error
            corr = np.corrcoef(feat_values, errors)[0, 1]
            if not np.isnan(corr):
                correlations[col] = corr

    # Sort features by the strength of their correlation with the error
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    print("Top features correlated with prediction error:")
    for feat, corr in sorted_corr[:5]:
        print(f"  {feat}: {corr:.4f}")

    # 7. Submission Generation
    # Only submit if the model meets the performance threshold
    THRESHOLD = 0.6913548345419015

    if auc_score > THRESHOLD:
        print(
            f"\nValidation score ({auc_score}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )
        with torch.no_grad():
            test_probs = ensemble.predict_proba(test_feats)
        save_submission(test_ids, test_probs, path=config.SUBMISSION_PATH)
    else:
        print(
            f"\nValidation score ({auc_score}) does not exceed threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
