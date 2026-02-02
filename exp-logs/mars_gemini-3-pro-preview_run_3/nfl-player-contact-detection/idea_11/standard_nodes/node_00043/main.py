import sys
import os
import pandas as pd
import numpy as np
import gc

# Import from library
from library.config import Config
from library.utils import seed_everything, compute_mcc
from library.feature_builder import FeatureBuilder
from library.model_factory import DualStreamModel


def run_pipeline():
    # 1. Setup
    seed_everything(Config.SEED)

    print("Initializing Feature Builder...")
    fb = FeatureBuilder()

    # 2. Load/Build Features
    # We use load_cached_data=True to leverage any existing work
    print("Loading/Building Train Features...")
    X_train_A, ids_train_A, y_train_A = fb.build_stream_a_features(
        "train", load_cached_data=True
    )
    X_train_B, ids_train_B, y_train_B = fb.build_stream_b_features(
        "train", load_cached_data=True
    )

    print("Loading/Building Validation Features...")
    X_val_A, ids_val_A, y_val_A = fb.build_stream_a_features(
        "val", load_cached_data=True
    )
    X_val_B, ids_val_B, y_val_B = fb.build_stream_b_features(
        "val", load_cached_data=True
    )

    # 4. Train Models
    print("Initializing Model...")
    model = DualStreamModel()

    data_bundle = {
        "X_train_A": X_train_A,
        "y_train_A": y_train_A,
        "X_val_A": X_val_A,
        "y_val_A": y_val_A,
        "X_train_B": X_train_B,
        "y_train_B": y_train_B,
        "X_val_B": X_val_B,
        "y_val_B": y_val_B,
    }

    model.train(data_bundle)

    # 5. Optimize Thresholds
    model.optimize_thresholds(X_val_A, y_val_A, X_val_B, y_val_B)

    # 6. Final Validation Evaluation
    print("Performing Final Validation...")
    # Get probabilities
    # Note: XGBoost on GPU is used, inference is efficient
    prob_A = model.model_a.predict_proba(X_val_A)[:, 1]
    prob_B = model.model_b.predict_proba(X_val_B)[:, 1]

    # Apply thresholds
    pred_A = (prob_A >= model.threshold_a).astype(int)
    pred_B = (prob_B >= model.threshold_b).astype(int)

    # Combine
    y_true_total = np.concatenate([y_val_A, y_val_B])
    pred_total = np.concatenate([pred_A, pred_B])

    final_mcc = compute_mcc(y_true_total, pred_total)
    print(f"Final Validation Metric: {final_mcc}")

    # 7. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Analyze Stream A (Interaction) as it's the primary complexity
    # Calculate error vector (0 for correct, 1 for incorrect)
    errors_A = np.abs(y_val_A - pred_A)

    # Calculate correlation with features
    # We'll pick a subset of numeric features to check
    numeric_cols = X_val_A.select_dtypes(include=[np.number]).columns

    # Compute correlations
    correlations = {}
    # Create a Series for errors aligned with X_val_A index
    error_series = pd.Series(errors_A, index=X_val_A.index)

    for col in numeric_cols:
        # Replace sentinel with NaN for correlation calculation to be meaningful
        feat_vals = X_val_A[col].replace(Config.SENTINEL_VALUE, np.nan)
        if feat_vals.std() > 0:  # Avoid constant columns
            corr = feat_vals.corr(error_series)
            correlations[col] = corr

    # Sort by absolute correlation
    sorted_corr = sorted(
        correlations.items(),
        key=lambda x: abs(x[1]) if pd.notnull(x[1]) else 0,
        reverse=True,
    )

    print("Top 5 Features correlated with Error (Stream A):")
    for name, val in sorted_corr[:5]:
        print(f"{name}: {val:.4f}")

    # 8. Submission
    THRESHOLD_SCORE = 0.6565613438092561

    if final_mcc > THRESHOLD_SCORE:
        print(
            f"\nValidation Metric ({final_mcc}) > Threshold ({THRESHOLD_SCORE}). Generating Submission..."
        )

        print("Loading/Building Test Features...")
        X_test_A, ids_test_A, _ = fb.build_stream_a_features(
            "test", load_cached_data=True
        )
        X_test_B, ids_test_B, _ = fb.build_stream_b_features(
            "test", load_cached_data=True
        )

        # Predict
        submission_df = model.predict(X_test_A, ids_test_A, X_test_B, ids_test_B)

        # Ensure format matches sample submission
        # Load sample submission to check alignment
        sample_sub = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)

        # Merge predictions onto sample submission to guarantee order and rows
        # submission_df has [contact_id, contact]
        final_sub = sample_sub[["contact_id"]].merge(
            submission_df, on="contact_id", how="left"
        )

        # Fill missing (if any) with 0
        final_sub["contact"] = final_sub["contact"].fillna(0).astype(int)

        # Save
        final_sub.to_csv(Config.FINAL_SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.FINAL_SUBMISSION_PATH}")

    else:
        print(
            f"\nValidation Metric ({final_mcc}) <= Threshold ({THRESHOLD_SCORE}). Skipping Submission."
        )


if __name__ == "__main__":
    run_pipeline()
