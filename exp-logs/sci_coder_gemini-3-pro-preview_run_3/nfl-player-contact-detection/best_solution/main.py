import pandas as pd
import numpy as np
import gc
import sys
from sklearn.metrics import matthews_corrcoef

# Import from provided libraries
from library.config import Config
from library.utils import seed_everything
from library.dataset_builder import DatasetBuilder
from library.model_trainer import (
    apply_undersampling,
    train_stream_model,
    optimize_threshold,
    DualStreamPredictor,
    generate_submission,
)


def perform_failure_analysis(X, y_true, y_prob, stream_name):
    """
    Calculates correlation between prediction error and input features.
    """
    print(f"\n--- Failure Analysis: Stream {stream_name} ---")

    if len(X) == 0:
        print("No data for failure analysis.")
        return

    # Calculate absolute error
    errors = np.abs(y_true - y_prob)

    # Calculate correlations
    correlations = {}
    feature_cols = X.columns.tolist()

    # Use a subset if validation set is huge to save time, though 220GB RAM is plenty
    # We'll use full set for accuracy
    for col in feature_cols:
        # Handle potential constant columns or NaNs safely
        if X[col].std() == 0:
            continue

        try:
            # simple correlation
            corr = np.corrcoef(X[col].values, errors)[0, 1]
            if not np.isnan(corr):
                correlations[col] = corr
        except Exception:
            continue

    # Sort by correlation (descending)
    sorted_corr = sorted(correlations.items(), key=lambda x: x[1], reverse=True)

    print(f"Top 5 Features associated with Error in Stream {stream_name}:")
    for feat, corr in sorted_corr[:5]:
        print(f"  {feat}: {corr:.4f}")


def main():
    # 1. Initialization
    seed_everything(Config.SEED)
    print("Starting Hybrid-Context Dual-Stream GBDT Pipeline...")

    # 2. Data Loading
    print("\nLoading Datasets...")
    train_builder = DatasetBuilder("train", load_cached_data=True)
    val_builder = DatasetBuilder("validation", load_cached_data=True)

    # --- Stream A: Interaction (Player-Player) ---
    print("\n=== Processing Stream A (Interaction) ===")
    # Load Data
    X_train_a, ids_train_a, y_train_a = train_builder.build_dataset("A")
    X_val_a, ids_val_a, y_val_a = val_builder.build_dataset("A")

    # Undersample
    X_train_a_res, y_train_a_res, _ = apply_undersampling(
        X_train_a, y_train_a, ids_train_a, Config.NEGATIVE_SAMPLING_RATIO, Config.SEED
    )

    # Train
    model_a = train_stream_model("A", X_train_a_res, y_train_a_res, X_val_a, y_val_a)

    # Optimize Threshold
    thresh_a = optimize_threshold(model_a, X_val_a, y_val_a, "A")

    # Get Validation Probabilities for Analysis
    probs_val_a = model_a.predict_proba(X_val_a)[:, 1]
    preds_val_a = (probs_val_a >= thresh_a).astype(int)

    # Failure Analysis A
    perform_failure_analysis(X_val_a, y_val_a, probs_val_a, "A")

    # Cleanup A Training Data to free memory
    del X_train_a, X_train_a_res, y_train_a, y_train_a_res, ids_train_a
    gc.collect()

    # --- Stream B: Impact (Player-Ground) ---
    print("\n=== Processing Stream B (Impact) ===")
    # Load Data
    X_train_b, ids_train_b, y_train_b = train_builder.build_dataset("B")
    X_val_b, ids_val_b, y_val_b = val_builder.build_dataset("B")

    # Undersample
    X_train_b_res, y_train_b_res, _ = apply_undersampling(
        X_train_b, y_train_b, ids_train_b, Config.NEGATIVE_SAMPLING_RATIO, Config.SEED
    )

    # Train
    model_b = train_stream_model("B", X_train_b_res, y_train_b_res, X_val_b, y_val_b)

    # Optimize Threshold
    thresh_b = optimize_threshold(model_b, X_val_b, y_val_b, "B")

    # Get Validation Probabilities for Analysis
    probs_val_b = model_b.predict_proba(X_val_b)[:, 1]
    preds_val_b = (probs_val_b >= thresh_b).astype(int)

    # Failure Analysis B
    perform_failure_analysis(X_val_b, y_val_b, probs_val_b, "B")

    # Cleanup B Training Data
    del X_train_b, X_train_b_res, y_train_b, y_train_b_res, ids_train_b
    gc.collect()

    # --- Global Validation ---
    print("\n=== Global Validation ===")
    # Concatenate Truth and Predictions
    y_true_total = np.concatenate([y_val_a, y_val_b])
    y_pred_total = np.concatenate([preds_val_a, preds_val_b])

    # Calculate Metric
    final_metric = matthews_corrcoef(y_true_total, y_pred_total)

    # Print EXACTLY as requested
    print(f"Final Validation Metric: {final_metric}")

    # --- Submission ---
    TARGET_METRIC = 0.6968
    if final_metric > TARGET_METRIC:
        print(
            f"\nMetric ({final_metric:.5f}) > Threshold ({TARGET_METRIC}). Generating Submission..."
        )

        # Create Wrapper
        predictor = DualStreamPredictor(model_a, model_b, thresh_a, thresh_b)

        # Generate Submission
        generate_submission(predictor, load_cached_data=True)

    else:
        print(
            f"\nMetric ({final_metric:.5f}) <= Threshold ({TARGET_METRIC}). Skipping Submission."
        )


if __name__ == "__main__":
    main()
