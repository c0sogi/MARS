import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_log_error

# Import from provided libraries
from library.config import SUBMISSION_PATH, RANDOM_SEED
from library.preprocessor import load_and_preprocess_data
from library.model_wrapper import XGBRegressorWrapper


def calculate_rmsle(y_true, y_pred):
    """
    Calculates Root Mean Squared Logarithmic Error.
    """
    # Ensure no negative values for log calculation
    y_pred_safe = np.maximum(y_pred, 0)
    return np.sqrt(mean_squared_log_error(y_true, y_pred_safe))


def main():
    print("Starting runfile.py execution...")

    # 1. Load and Preprocess Data
    # We use load_cached_data=True to leverage any existing features in ./working/idea_41
    # If features don't exist, they will be computed.
    print("\n[Step 1] Loading and preprocessing data...")
    (X_train, y_train), (X_val, y_val), (X_test, test_ids) = load_and_preprocess_data(
        load_cached_data=True
    )

    print(f"Training shape: {X_train.shape}")
    print(f"Validation shape: {X_val.shape}")
    print(f"Test shape: {X_test.shape}")

    # 2. Model Training
    print("\n[Step 2] Training XGBoost models...")
    model_wrapper = XGBRegressorWrapper()

    # Train returns a dict of RMSLE scores on the validation set (calculated in log space)
    # The wrapper handles log1p transformation of targets internally during fit/predict logic context
    # but the .train() method specifically calculates metrics on the transformed data.
    # RMSLE on original data ~= RMSE on log(1+y) data.
    metrics = model_wrapper.train(X_train, y_train, X_val, y_val)

    # 3. Validation Assessment
    print("\n[Step 3] Assessing Validation Performance...")

    # Get predictions in original units
    val_preds = model_wrapper.predict(X_val)

    # Ensure non-negative predictions for RMSLE calculation (physics constraint)
    val_preds[val_preds < 0] = 0

    # Calculate competition metric: Mean of column-wise RMSLE
    target_cols = ["formation_energy_ev_natom", "bandgap_energy_ev"]
    rmsle_scores = []

    for col in target_cols:
        score = calculate_rmsle(y_val[col], val_preds[col])
        rmsle_scores.append(score)
        print(f"Target: {col}, RMSLE: {score:.6f}")

    final_metric = np.mean(rmsle_scores)
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    print("\n[Step 4] Performing Failure Analysis...")
    # Calculate error magnitude per sample
    # We use mean absolute error of log-transformed values as a proxy for "difficulty"
    # Error = | log(1+y_true) - log(1+y_pred) |

    y_val_log = np.log1p(y_val)
    val_preds_log = np.log1p(val_preds)

    # Average error across targets
    error_magnitude = np.mean(np.abs(y_val_log - val_preds_log), axis=1)

    # Correlate error with features
    # We concat X_val and error to compute correlation
    analysis_df = X_val.copy()
    analysis_df["error_magnitude"] = error_magnitude

    # Drop columns with NaN correlations (constant columns)
    correlations = (
        analysis_df.corr()["error_magnitude"].drop("error_magnitude").dropna()
    )
    top_correlations = correlations.abs().sort_values(ascending=False).head(10)

    print("Top 10 features correlated with prediction error:")
    print(top_correlations)

    # 5. Submission Generation
    # Threshold check
    THRESHOLD = 0.05095
    if final_metric < THRESHOLD:
        print(
            f"\n[Step 5] Metric {final_metric:.6f} < {THRESHOLD}. Generating submission..."
        )

        test_preds = model_wrapper.predict(X_test)

        # Ensure non-negative
        test_preds[test_preds < 0] = 0

        # Construct submission DataFrame
        submission_df = pd.DataFrame(
            {
                "id": test_ids,
                "formation_energy_ev_natom": test_preds["formation_energy_ev_natom"],
                "bandgap_energy_ev": test_preds["bandgap_energy_ev"],
            }
        )

        # Save
        submission_df.to_csv(SUBMISSION_PATH, index=False)
        print(f"Submission saved to {SUBMISSION_PATH}")
        print(submission_df.head())
    else:
        print(
            f"\n[Step 5] Metric {final_metric:.6f} >= {THRESHOLD}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
