import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error

# Import from provided library files
from library.config import SEED, N_FOLDS, WORKING_DIR
from library.data_manager import get_val_data
from library.workflow_orchestrator import run_cross_validation, generate_submission
from library.model_handler import predict_ensemble

# Set random seeds for reproducibility
np.random.seed(SEED)


def main():
    """
    Main execution function for the volcanic eruption prediction task.
    Orchestrates training, validation, failure analysis, and submission.
    """

    # -------------------------------------------------------------------------
    # 1. Training Phase
    # -------------------------------------------------------------------------
    print("=== Starting Training Phase ===")
    # We use the full dataset (debug_size=None) because the dataset size (approx 4000 files)
    # is manageable within the time limit given the parallelized feature engineering.
    # High-capacity LightGBM requires sufficient data to generalize effectively.
    try:
        run_cross_validation(load_cached_data=True, debug_size=None)
    except Exception as e:
        print(f"Training failed: {e}")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # 2. Validation Phase
    # -------------------------------------------------------------------------
    print("\n=== Starting Validation Phase ===")
    try:
        # Load the hold-out validation dataset
        # We evaluate on the specific validation set defined in metadata/val.csv
        X_val, y_val = get_val_data(load_cached_data=True, debug_size=None)

        # Load the trained models from the working directory
        models = []
        for i in range(N_FOLDS):
            model_path = os.path.join(WORKING_DIR, f"model_fold_{i}.joblib")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            models.append(joblib.load(model_path))

        # Generate predictions on validation set
        # The ensemble averages predictions from all fold models
        val_preds = predict_ensemble(models, X_val)

        # Compute Metric (Mean Absolute Error)
        final_metric = mean_absolute_error(y_val, val_preds)

        # Print the required metric string
        print(f"Final Validation Metric: {final_metric}")

    except Exception as e:
        print(f"Validation failed: {e}")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # 3. Failure Analysis
    # -------------------------------------------------------------------------
    print("\n=== Starting Failure Analysis ===")
    try:
        # Calculate error magnitude
        errors = np.abs(y_val - val_preds)

        # Create analysis dataframe
        analysis_df = X_val.copy()
        analysis_df["error_magnitude"] = errors

        # Compute correlations between features and error magnitude
        feature_cols = [c for c in analysis_df.columns if c != "error_magnitude"]
        correlations = (
            analysis_df[feature_cols].corrwith(analysis_df["error_magnitude"]).abs()
        )

        # Sort and print top correlations to identify sources of error
        top_correlations = correlations.sort_values(ascending=False).head(5)
        print("Top 5 features correlated with error magnitude:")
        print(top_correlations)

    except Exception as e:
        print(f"Failure analysis failed: {e}")
        # Proceed to submission check even if analysis fails

    # -------------------------------------------------------------------------
    # 4. Submission Phase
    # -------------------------------------------------------------------------
    print("\n=== Checking Submission Criteria ===")
    THRESHOLD = 2617304.0647319085

    if final_metric < THRESHOLD:
        print(
            f"Metric {final_metric} is below threshold {THRESHOLD}. Generating submission..."
        )
        try:
            generate_submission(load_cached_data=True, debug_size=None)
            print("Submission generation complete.")
        except Exception as e:
            print(f"Submission generation failed: {e}")
            sys.exit(1)
    else:
        print(
            f"Metric {final_metric} is NOT below threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
