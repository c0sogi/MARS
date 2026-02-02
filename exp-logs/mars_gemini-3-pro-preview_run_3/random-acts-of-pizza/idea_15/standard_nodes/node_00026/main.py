import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Ensure library modules can be imported
sys.path.append(os.getcwd())

from library.config import (
    RANDOM_SEED,
    RAW_NUMERICAL_COLS,
    SUBMISSION_PATH,
    ID_COL,
    TARGET_COL,
)
from library.utils import set_seed, timer, print_header
from library.feature_engineering import DataPipeline
from library.stacking_engine import PentViewStackingEnsemble


def main():
    # 1. Setup
    set_seed(RANDOM_SEED)

    # 2. Data Loading & Processing
    # We use load_cached_data=True to leverage any existing preprocessed files
    pipeline = DataPipeline()
    data = pipeline.process_data(load_cached_data=True)

    # 3. Model Training
    # The ensemble handles CV stacking and retraining internally
    ensemble = PentViewStackingEnsemble()
    ensemble.fit(data)

    # 4. Validation
    print_header("Validation Analysis")
    # Predict on the hold-out validation set
    val_preds = ensemble.predict(data["val"])
    y_val = data["val"]["y"]

    # Calculate and print metric
    val_auc = roc_auc_score(y_val, val_preds)
    print(f"Final Validation Metric: {val_auc}")

    # 5. Failure Analysis
    print_header("Failure Analysis")
    # Calculate absolute error |y_true - y_pred|
    errors = np.abs(y_val - val_preds)

    # Correlate errors with numerical metadata features
    # data['val']['metadata'] corresponds to RAW_NUMERICAL_COLS (scaled)
    X_val_meta = data["val"]["metadata"]

    correlations = []
    # Iterate through features to compute correlation with error
    for i, col_name in enumerate(RAW_NUMERICAL_COLS):
        if i < X_val_meta.shape[1]:
            feature_values = X_val_meta[:, i]
            # Avoid division by zero if feature is constant
            if np.std(feature_values) == 0 or np.std(errors) == 0:
                corr = 0.0
            else:
                corr = np.corrcoef(feature_values, errors)[0, 1]
            correlations.append((col_name, corr))

    # Sort by absolute correlation strength
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features correlated with Prediction Error:")
    for name, corr in correlations[:5]:
        print(f"  {name}: {corr:.4f}")

    # 6. Submission Generation
    THRESHOLD = 0.6913548345419015

    if val_auc > THRESHOLD:
        print_header("Generating Submission")
        # Generate predictions for test set
        test_preds = ensemble.predict(data["test"])
        test_ids = data["test"]["ids"]

        # Create submission DataFrame
        submission_df = pd.DataFrame({ID_COL: test_ids, TARGET_COL: test_preds})

        # Save to disk
        os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
        submission_df.to_csv(SUBMISSION_PATH, index=False)
        print(f"Submission saved to {SUBMISSION_PATH}")
        print(f"Submission shape: {submission_df.shape}")
    else:
        print(
            f"\nValidation AUC ({val_auc}) did not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
