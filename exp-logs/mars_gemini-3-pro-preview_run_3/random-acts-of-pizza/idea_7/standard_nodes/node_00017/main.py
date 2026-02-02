import sys
import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

# Ensure the current directory is in the python path for module imports
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed, setup_logger
from library.feature_extraction import extract_features
from library.model_pipeline import TriViewStackingClassifier


def main():
    # 1. Setup
    logger = setup_logger("runfile")
    set_seed(Config.SEED)

    logger.info("Starting runfile execution...")

    # 2. Data Loading & Feature Extraction
    # We use debug=False to train on the full provided training set (approx 2k samples).
    # load_cache=True will attempt to use pre-computed features in ./working/idea_7 if they exist.
    logger.info("Loading data and extracting features...")
    (X_train, y_train), (X_val, y_val), (X_test, test_ids) = extract_features(
        debug=False, load_cache=True
    )

    # 3. Model Initialization & Training
    logger.info("Initializing Tri-View Stacking Ensemble...")
    model = TriViewStackingClassifier()

    logger.info("Fitting model with Cross-Validation Stacking...")
    # fit_cv handles OOF generation, meta-learner training, and final base-learner retraining
    model.fit_cv(X_train, y_train)

    # 4. Validation
    logger.info("Predicting on validation set...")
    val_probs = model.predict_proba(X_val)

    # Calculate Metric
    val_auc = roc_auc_score(y_val, val_probs)
    print(f"Final Validation Metric: {val_auc}")

    # 5. Failure Analysis
    logger.info("Running failure analysis...")
    # Calculate absolute error
    errors = np.abs(y_val - val_probs)

    # Reconstruct meta feature names based on Config and FeaturePipeline logic
    # The order in X_val['meta'] corresponds to Config.NUMERICAL_COLS + derived temporal features
    meta_cols = Config.NUMERICAL_COLS.copy()
    if "unix_timestamp_of_request" in meta_cols:
        meta_cols.append("request_hour")
        meta_cols.append("request_day_of_week")

    X_val_meta = X_val["meta"]

    # Verify shape matches expected columns
    if X_val_meta.shape[1] == len(meta_cols):
        correlations = []
        for i, col_name in enumerate(meta_cols):
            feat_values = X_val_meta[:, i]
            # Calculate Pearson correlation between feature value and error magnitude
            # We handle potential constant columns (std=0) which would cause pearsonr to warn/fail
            if np.std(feat_values) > 1e-9:
                corr, _ = pearsonr(errors, feat_values)
                correlations.append((col_name, corr))
            else:
                correlations.append((col_name, 0.0))

        # Sort by absolute correlation to find most impactful features on error
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)

        print("\nFailure Analysis - Correlation with Error Magnitude (Top 5):")
        for name, corr in correlations[:5]:
            print(f"{name}: {corr:.4f}")
    else:
        logger.warning(
            f"Meta feature shape {X_val_meta.shape} mismatch with expected {len(meta_cols)} columns. Skipping correlation analysis."
        )

    # 6. Submission
    THRESHOLD = 0.6913548345419015

    if val_auc > THRESHOLD:
        logger.info(
            f"Validation AUC ({val_auc}) meets threshold ({THRESHOLD}). Generating submission..."
        )

        # Predict on Test Set
        test_probs = model.predict_proba(X_test)

        # Create Submission DataFrame
        submission_df = pd.DataFrame(
            {"request_id": test_ids, "requester_received_pizza": test_probs}
        )

        # Save
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        logger.warning(
            f"Validation AUC ({val_auc}) did not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
