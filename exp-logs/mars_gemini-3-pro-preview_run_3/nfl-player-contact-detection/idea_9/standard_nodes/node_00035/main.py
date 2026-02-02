import sys
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import matthews_corrcoef

# Import library components
from library.pipeline import run_training, run_inference
from library.features import FeatureEngineer
from library.utils import seed_everything, compute_mcc
import library.config as config


def main():
    # 1. Setup
    seed_everything(config.SEED)
    print("Initializing Asymmetric Modality-Selective Dual-Stream GBDT Pipeline...")

    # 2. Training Phase
    # We limit n_estimators to 800 to ensure the run completes well within the 2-hour limit
    # while providing sufficient capacity for the gradient boosting to converge.
    print("\n--- Phase 1: Training ---")
    model = run_training(load_cached_data=True, n_estimators=800, debug=False)

    # 3. Validation & Failure Analysis Phase
    print("\n--- Phase 2: Validation & Failure Analysis ---")
    fe_val = FeatureEngineer(mode="validation")
    val_data = fe_val.generate_features(load_cached_data=True)

    all_y_true = []
    all_y_pred = []

    # Iterate through streams to predict and analyze
    for stream in ["A", "B"]:
        if stream not in val_data:
            continue

        if stream not in model.models:
            print(f"Warning: Model for Stream {stream} not found.")
            continue

        print(
            f"\nAnalyzing Stream {stream} ({config.STREAM_CONFIG[stream]['description']})..."
        )

        X_val, y_val, ids_val = val_data[stream]
        xgb_model = model.models[stream]
        threshold = model.thresholds.get(stream, 0.5)

        # Generate Predictions
        # XGBoost on GPU handles this efficiently
        proba = xgb_model.predict_proba(X_val)[:, 1]
        preds = (proba >= threshold).astype(int)

        # Aggregate for Global Metric
        all_y_true.append(y_val)
        all_y_pred.append(preds)

        # --- Failure Analysis ---
        # Calculate error magnitude (0 for correct, 1 for incorrect)
        errors = np.abs(y_val - preds)

        # We calculate correlation between features and the error signal
        # This helps identify which features are associated with model failure
        if len(errors) > 0:
            # Create a temporary series for correlation calculation
            error_series = pd.Series(errors, index=X_val.index)

            # Compute correlation with all features
            # abs() because we care about the strength of association with error
            correlations = (
                X_val.corrwith(error_series).abs().sort_values(ascending=False)
            )

            print(f"Top 5 Features correlated with Error in Stream {stream}:")
            print(correlations.head(5).to_string())

    # 4. Global Evaluation
    if all_y_true:
        y_true_global = np.concatenate(all_y_true)
        y_pred_global = np.concatenate(all_y_pred)

        val_mcc = compute_mcc(y_true_global, y_pred_global)
        print(f"\nFinal Validation Metric: {val_mcc}")
    else:
        print("\nError: No validation data processed.")
        val_mcc = -1.0

    # 5. Submission Phase
    # Threshold defined in task requirements
    SUBMISSION_THRESHOLD = 0.6565613438092561

    print("\n--- Phase 3: Submission Decision ---")
    if val_mcc > SUBMISSION_THRESHOLD:
        print(f"Validation MCC ({val_mcc}) exceeds threshold ({SUBMISSION_THRESHOLD}).")
        print("Proceeding with Test Inference and Submission Generation...")
        run_inference(model=model, load_cached_data=True)
    else:
        print(
            f"Validation MCC ({val_mcc}) does not exceed threshold ({SUBMISSION_THRESHOLD})."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
