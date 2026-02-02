import sys
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import matthews_corrcoef

# Import library components
from library.config import Config
from library.utils import setup_seed, calc_mcc
from library.feature_engineering import process_data
from library.model import ContactModel
from library.pipeline import train_pipeline, inference_pipeline


def main():
    # 1. Setup Reproducibility
    setup_seed(Config.SEED)

    # 2. Override Config for Fast Baseline Execution
    # Reducing tree count and undersampling ratio to ensure completion within time limit
    print("Overriding Config for fast baseline execution...")
    Config.STREAM_A_PARAMS["n_estimators"] = 100
    Config.STREAM_B_PARAMS["n_estimators"] = 100
    Config.UNDERSAMPLE_RATIO = 1.0

    # 3. Run Training Pipeline
    # This generates features, trains models, optimizes thresholds, and saves artifacts.
    train_pipeline(load_cached_data=True)

    # 4. Load Validation Data for Analysis
    print("\nLoading validation data for post-training analysis...")
    X_val_a, ids_val_a, y_val_a, X_val_b, ids_val_b, y_val_b = process_data(
        "validation", load_cached_data=True
    )

    # 5. Load Trained Models and Thresholds
    model_a_path = os.path.join(Config.WORKING_DIR, "model_stream_a.joblib")
    model_b_path = os.path.join(Config.WORKING_DIR, "model_stream_b.joblib")
    thresholds_path = os.path.join(Config.WORKING_DIR, "thresholds.joblib")

    if not (
        os.path.exists(model_a_path)
        and os.path.exists(model_b_path)
        and os.path.exists(thresholds_path)
    ):
        raise FileNotFoundError("Training artifacts not found.")

    model_a = ContactModel.load(model_a_path)
    model_b = ContactModel.load(model_b_path)
    thresholds = joblib.load(thresholds_path)

    thresh_a = thresholds["stream_a"]
    thresh_b = thresholds["stream_b"]

    # 6. Generate Validation Predictions
    # Stream A (Interaction)
    if len(y_val_a) > 0:
        pred_proba_a = model_a.predict(X_val_a)
        pred_bin_a = (pred_proba_a >= thresh_a).astype(int)
    else:
        pred_proba_a = np.array([])
        pred_bin_a = np.array([])

    # Stream B (Impact)
    if len(y_val_b) > 0:
        pred_proba_b = model_b.predict(X_val_b)
        pred_bin_b = (pred_proba_b >= thresh_b).astype(int)
    else:
        pred_proba_b = np.array([])
        pred_bin_b = np.array([])

    # 7. Compute and Print Global Validation Metric
    y_val_total = np.concatenate([y_val_a, y_val_b])
    pred_bin_total = np.concatenate([pred_bin_a, pred_bin_b])

    final_mcc = calc_mcc(y_val_total, pred_bin_total)
    print(f"Final Validation Metric: {final_mcc}")

    # 8. Perform Failure Analysis
    print("\nPerforming Failure Analysis...")

    # Analyze Stream A
    if len(y_val_a) > 0:
        print("Stream A (Interaction) Error Correlations:")
        errors_a = np.abs(y_val_a - pred_proba_a)
        corrs_a = []
        # X_val_a is a DataFrame
        for col in X_val_a.columns:
            # Calculate correlation between feature and error magnitude
            if X_val_a[col].std() > 0:  # Avoid constant columns
                c = np.corrcoef(X_val_a[col], errors_a)[0, 1]
                if not np.isnan(c):
                    corrs_a.append((col, c))

        corrs_a.sort(key=lambda x: abs(x[1]), reverse=True)
        for feat, corr in corrs_a[:5]:
            print(f"  {feat}: {corr:.4f}")

    # Analyze Stream B
    if len(y_val_b) > 0:
        print("Stream B (Impact) Error Correlations:")
        errors_b = np.abs(y_val_b - pred_proba_b)
        corrs_b = []
        for col in X_val_b.columns:
            if X_val_b[col].std() > 0:
                c = np.corrcoef(X_val_b[col], errors_b)[0, 1]
                if not np.isnan(c):
                    corrs_b.append((col, c))

        corrs_b.sort(key=lambda x: abs(x[1]), reverse=True)
        for feat, corr in corrs_b[:5]:
            print(f"  {feat}: {corr:.4f}")

    # 9. Conditional Submission Generation
    threshold_score = 0.7008
    if final_mcc > threshold_score:
        print(
            f"\nValidation score ({final_mcc:.4f}) > {threshold_score}. Generating submission..."
        )
        inference_pipeline(load_cached_data=True)
    else:
        print(
            f"\nValidation score ({final_mcc:.4f}) <= {threshold_score}. Skipping submission."
        )


if __name__ == "__main__":
    main()
