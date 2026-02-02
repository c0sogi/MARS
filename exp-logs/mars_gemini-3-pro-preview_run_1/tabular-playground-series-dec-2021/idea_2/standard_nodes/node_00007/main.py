import os
import sys
import numpy as np
import pandas as pd
import warnings
import random
from sklearn.metrics import accuracy_score

# Add current directory to path to ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.data_loader import load_and_process_data
from library.model_factory import XGBoostWrapper
from library.ensemble_utils import save_submission

# Suppress warnings
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def perform_failure_analysis(X_val, y_val, pred_labels):
    """
    Analyzes failure modes by correlating error magnitude with features.
    """
    print("\n--- Failure Analysis ---")

    # Calculate error vector (1 for incorrect, 0 for correct)
    # Ensure indices align
    if isinstance(y_val, pd.Series):
        y_val_arr = y_val.values
    else:
        y_val_arr = y_val

    errors = (y_val_arr != pred_labels).astype(int)

    print(f"Total Validation Samples: {len(errors)}")
    print(f"Total Errors: {errors.sum()}")
    print(f"Error Rate: {errors.mean():.6f}")

    # Create a Series for errors with the same index as X_val to ensure alignment
    error_series = pd.Series(errors, index=X_val.index)

    # Calculate correlation with numerical features
    # Select only numeric columns for correlation
    numeric_cols = X_val.select_dtypes(include=[np.number]).columns

    correlations = X_val[numeric_cols].corrwith(error_series)

    # Sort by absolute correlation
    abs_corrs = correlations.abs().sort_values(ascending=False)

    print("\nTop 10 Features correlated with Error (Magnitude):")
    for feat in abs_corrs.head(10).index:
        corr_val = correlations[feat]
        print(f"{feat}: {corr_val:.6f}")


def main():
    # 1. Setup
    set_seed(Config.SEED)

    # 2. Load Data
    # Using max_samples=None to use full dataset for best performance within time limit (A100 is fast enough)
    print("Loading and processing data...")
    X_train, y_train, X_val, y_val, X_test, test_ids = load_and_process_data(
        load_cached_data=True, max_samples=None
    )

    # 3. Model Training
    print("\n--- Training XGBoost ---")
    xgb_model = XGBoostWrapper()
    xgb_model.train(X_train, y_train, X_val, y_val)

    # 4. Validation Inference
    print("\n--- Running Validation Inference ---")
    # Get probabilities
    val_probs = xgb_model.predict_proba(X_val)

    # Convert to labels
    class_labels = xgb_model.le.classes_
    val_pred_indices = np.argmax(val_probs, axis=1)
    val_pred_labels = class_labels[val_pred_indices]

    # 5. Metrics
    val_acc = accuracy_score(y_val, val_pred_labels)
    # Print full precision metric as required
    print(f"Final Validation Metric: {val_acc}")

    # 6. Failure Analysis
    perform_failure_analysis(X_val, y_val, val_pred_labels)

    # 7. Submission
    # Threshold check
    THRESHOLD = 0.9612388888888889

    if val_acc > THRESHOLD:
        print(
            f"\nValidation accuracy ({val_acc}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        # Inference on Test
        test_probs = xgb_model.predict_proba(X_test)

        # Save
        save_submission(
            test_ids=test_ids,
            probabilities=test_probs,
            class_labels=class_labels,
            output_path=Config.SUBMISSION_PATH,
        )
    else:
        print(
            f"\nValidation accuracy ({val_acc}) did not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
