import sys
import os
import numpy as np
import pandas as pd

# Ensure local library imports work
sys.path.append(os.getcwd())

from library.data_loader import NFLDataLoader
from library.model import DualStreamXGB
from library.utils import seed_everything, calc_mcc
from library.config import SEED


def main():
    # 1. Setup
    seed_everything(SEED)
    print("Starting execution...")

    # 2. Data Loading
    loader = NFLDataLoader()

    print("Loading Training Data...")
    # load_cached_data=True allows using pre-computed features if available
    train_data = loader.prepare_streams(split="train", load_cached_data=True)

    print("Loading Validation Data...")
    val_data = loader.prepare_streams(split="validation", load_cached_data=True)

    # 3. Model Training
    print("Initializing and Training Model...")
    model = DualStreamXGB()

    # The model uses XGBoost with GPU support and early stopping.
    # Training data is already undersampled by the data loader.
    model.fit(train_data, val_data)

    # 4. Threshold Optimization
    print("Optimizing Thresholds...")
    model.optimize_thresholds(val_data)

    # 5. Validation Metric Calculation
    print("Calculating Final Validation Metric...")

    # Extract validation data and predictions for Stream A
    X_val_a = val_data["stream_a"]["X"]
    y_val_a = val_data["stream_a"]["y"]
    if len(X_val_a) > 0:
        probs_a = model.model_a.predict_proba(X_val_a)[:, 1]
        preds_a = (probs_a >= model.threshold_a).astype(int)
    else:
        y_val_a = np.array([])
        preds_a = np.array([])

    # Extract validation data and predictions for Stream B
    X_val_b = val_data["stream_b"]["X"]
    y_val_b = val_data["stream_b"]["y"]
    if len(X_val_b) > 0:
        probs_b = model.model_b.predict_proba(X_val_b)[:, 1]
        preds_b = (probs_b >= model.threshold_b).astype(int)
    else:
        y_val_b = np.array([])
        preds_b = np.array([])

    # Combine streams
    all_true = np.concatenate([y_val_a, y_val_b])
    all_preds = np.concatenate([preds_a, preds_b])

    final_mcc = calc_mcc(all_true, all_preds)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_mcc}")

    # 6. Failure Analysis
    print("\n--- Failure Analysis ---")

    def analyze_errors(X, y_true, y_pred, stream_name):
        if len(y_true) == 0:
            print(f"{stream_name}: No data for analysis.")
            return

        # Calculate binary error (0 for correct, 1 for incorrect)
        errors = np.abs(y_true - y_pred)

        correlations = {}
        # Iterate over columns to find correlation with error
        for col in X.columns:
            # Skip non-numeric if any (though features should be numeric)
            if not pd.api.types.is_numeric_dtype(X[col]):
                continue

            try:
                # Handle constant columns which produce NaN correlation
                if X[col].std() == 0:
                    corr = 0
                else:
                    corr = np.corrcoef(X[col].values, errors)[0, 1]

                if np.isnan(corr):
                    corr = 0
                correlations[col] = corr
            except Exception:
                correlations[col] = 0

        # Sort by absolute correlation
        sorted_corrs = sorted(
            correlations.items(), key=lambda x: abs(x[1]), reverse=True
        )[:5]

        print(f"{stream_name} - Top Feature Correlations with Error:")
        for feat, corr in sorted_corrs:
            print(f"  {feat}: {corr:.4f}")

    analyze_errors(X_val_a, y_val_a, preds_a, "Stream A (Player-Player)")
    analyze_errors(X_val_b, y_val_b, preds_b, "Stream B (Player-Ground)")

    # 7. Submission
    THRESHOLD = 0.6533

    if final_mcc > THRESHOLD:
        print(
            f"\nValidation Metric ({final_mcc}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )

        print("Loading Test Data...")
        test_data = loader.prepare_streams(split="test", load_cached_data=True)

        print("Generating Submission File...")
        model.generate_submission(test_data)

        # Save models for persistence
        model.save_models()
    else:
        print(
            f"\nValidation Metric ({final_mcc}) did not exceed threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
