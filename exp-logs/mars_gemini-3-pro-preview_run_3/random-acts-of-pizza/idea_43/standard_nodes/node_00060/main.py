import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr
import warnings

# Ensure the current directory is in the path to import from library
sys.path.append(os.getcwd())

from library.config import Config
from library.data_loader import load_datasets
from library.stacking_manager import HexStackEnsemble

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    """Sets random seeds for reproducibility."""
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    # 1. Configuration and Setup
    set_seed(Config.SEED)
    print("Initializing Hex-View Stacking Ensemble Pipeline...")

    # 2. Load Data Metadata to determine split boundaries
    # We load the data here primarily to know the size of the train and validation sets
    # so we can correctly slice the OOF predictions later.
    print("Loading metadata to determine split boundaries...")
    train_df, val_df, _ = load_datasets(debug=False)
    n_train = len(train_df)
    n_val = len(val_df)
    print(f"Train size: {n_train}, Validation size: {n_val}")

    # 3. Initialize the Ensemble Manager
    ensemble = HexStackEnsemble()

    # 4. Train Level 1 (OOF Generation)
    # This method performs Stratified K-Fold CV on the combined (Train + Val) dataset.
    # It returns predictions for every sample in the combined dataset.
    print("\nStep 1: Generating OOF Predictions (Level 1)...")
    oof_preds_l1, y_full = ensemble.train_oof(debug=False)

    # oof_preds_l1 shape: (n_train + n_val, 6)
    # y_full shape: (n_train + n_val,)

    # 5. Train Level 2 (Meta Learner)
    # We train the meta-learner on the full OOF predictions.
    print("\nStep 2: Training Meta-Learner (Level 2)...")
    meta_learner = ensemble.train_meta(oof_preds_l1, y_full)

    # 6. Validation Evaluation
    # We extract the predictions corresponding to the hold-out validation set.
    # Since the stacking manager concatenates [train, val], the validation samples are at the end.
    print("\nStep 3: Evaluating on Hold-out Validation Set...")

    # Get the final ensemble probability predictions for the whole dataset
    ensemble_probs_full = meta_learner.predict_proba(oof_preds_l1)[:, 1]

    # Slice to get only the validation set predictions
    val_probs = ensemble_probs_full[n_train:]
    y_val_actual = y_full[n_train:]

    # Sanity check for alignment
    if len(val_probs) != n_val:
        raise ValueError(
            f"Validation slice length mismatch. Expected {n_val}, got {len(val_probs)}"
        )

    # Compute and print the required metric
    val_auc = roc_auc_score(y_val_actual, val_probs)
    print(f"Final Validation Metric: {val_auc}")

    # 7. Failure Analysis
    print("\nStep 4: Performing Failure Analysis...")
    # Calculate absolute error magnitude
    errors = np.abs(y_val_actual - val_probs)

    # Identify numerical columns for correlation analysis
    # We exclude ID columns, timestamps, and the target itself
    numerical_cols = val_df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = [
        "requester_received_pizza",
        "request_id",
        "unix_timestamp_of_request",
        "unix_timestamp_of_request_utc",
    ]
    # Also exclude retrieval-time features if any leaked into val_df (though data_loader handles this)
    numerical_cols = [
        c for c in numerical_cols if c not in exclude_cols and "retrieval" not in c
    ]

    correlations = []
    for col in numerical_cols:
        # Fill NaNs with median for correlation calculation to be robust
        feat_vals = val_df[col].fillna(val_df[col].median())

        # Calculate Pearson correlation if the feature has variance
        if len(feat_vals.unique()) > 1:
            corr, _ = pearsonr(feat_vals, errors)
            correlations.append((col, corr))

    # Sort features by the absolute value of their correlation with error
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features correlated with Error Magnitude:")
    for name, corr in correlations[:5]:
        print(f"  {name}: {corr:.4f}")

    # 8. Submission Logic
    # Check against the provided threshold
    threshold = 0.7138293787137718

    if val_auc > threshold:
        print(
            f"\nValidation metric {val_auc} exceeds threshold {threshold}. Proceeding to submission."
        )

        # 9. Retrain Final Model
        # This fits transformers on the full dataset and retrains base learners
        print("\nStep 5: Retraining Final Ensemble on Full Data...")
        ensemble.retrain_final(debug=False)

        # 10. Generate Submission
        # Predicts on the test set and saves to ./submission/submission.csv
        print("\nStep 6: Generating Submission for Test Set...")
        ensemble.predict(debug=False)

        print("Pipeline Completed Successfully.")
    else:
        print(
            f"\nValidation metric {val_auc} did not exceed threshold {threshold}. Skipping submission."
        )


if __name__ == "__main__":
    main()
