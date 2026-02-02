import os
import gc
import numpy as np
import pandas as pd
from library.config import WORKING_DIR, RANDOM_STATE
from library.utils import seed_everything, compute_mcc
from library.feature_engineering import generate_features
from library.training_pipeline import (
    train_scout_model,
    mine_hard_negatives,
    train_expert_models,
)
from library.inference import predict_and_submit

# Set random seed for reproducibility
seed_everything(RANDOM_STATE)


def main():
    print("Starting Vectorized Flow-Field Mining Cascade (VFF-MC) Pipeline...")

    # =========================================================================
    # 1. Configuration for Fast Baseline
    # =========================================================================
    # Limit rows to ensure execution within 2 hours
    N_TRAIN_SAMPLES = 500000
    N_VAL_SAMPLES = 100000

    # Create models directory if it doesn't exist (needed for saving threshold)
    os.makedirs(os.path.join(WORKING_DIR, "models"), exist_ok=True)

    # =========================================================================
    # 2. Data Loading & Feature Engineering
    # =========================================================================
    print("\n[Step 1/6] Generating Features...")

    # Generate Training Features (with Gating)
    df_train = generate_features(
        split="train", load_cached_data=True, nrows=N_TRAIN_SAMPLES, gating=True
    )

    # Generate Validation Features (with Gating)
    df_val = generate_features(
        split="val", load_cached_data=True, nrows=N_VAL_SAMPLES, gating=True
    )

    # Force garbage collection
    gc.collect()

    # =========================================================================
    # 3. Scout Training
    # =========================================================================
    print("\n[Step 2/6] Training Scout Model...")
    scout_model, feature_cols = train_scout_model(df_train, df_val)

    gc.collect()

    # =========================================================================
    # 4. Hard Negative Mining
    # =========================================================================
    print("\n[Step 3/6] Mining Hard Negatives...")
    hard_neg_indices = mine_hard_negatives(
        scout_model, df_train, feature_cols, load_cached_data=True
    )

    # Free up memory
    del scout_model
    gc.collect()

    # =========================================================================
    # 5. Expert Training
    # =========================================================================
    print("\n[Step 4/6] Training Expert Ensemble...")
    ensemble = train_expert_models(df_train, df_val, hard_neg_indices, feature_cols)

    # Free up memory (keep df_val for evaluation)
    del df_train
    gc.collect()

    # =========================================================================
    # 6. Validation & Metric Calculation
    # =========================================================================
    print("\n[Step 5/6] Validating Model...")

    X_val = df_val[feature_cols]
    y_val = df_val["contact"].values

    # Get probability predictions
    val_probs = ensemble.predict(X_val)

    # Optimize Threshold
    best_mcc = -1.0
    best_th = 0.5
    thresholds = np.linspace(0.05, 0.95, 91)

    for th in thresholds:
        pred_labels = (val_probs >= th).astype(int)
        mcc = compute_mcc(y_val, pred_labels)
        if mcc > best_mcc:
            best_mcc = mcc
            best_th = th

    # REQUIRED: Print Final Validation Metric
    print(f"Final Validation Metric: {best_mcc:.16f}")

    # Save the best threshold for inference
    thresh_path = os.path.join(WORKING_DIR, "models", "best_threshold.npy")
    np.save(thresh_path, np.array([best_th]))
    print(f"Optimal threshold {best_th:.4f} saved to {thresh_path}")

    # =========================================================================
    # 7. Failure Analysis
    # =========================================================================
    print("\n--- Failure Analysis ---")

    # Calculate Error Magnitude
    # For binary classification, error is |y_true - y_prob|
    errors = np.abs(y_val - val_probs)

    # Calculate correlation between features and error
    # We only check numeric columns
    numeric_cols = X_val.select_dtypes(include=[np.number]).columns
    correlations = {}

    for col in numeric_cols:
        # Simple correlation
        try:
            corr = np.corrcoef(X_val[col].fillna(0), errors)[0, 1]
            if not np.isnan(corr):
                correlations[col] = corr
        except Exception:
            continue

    # Sort by absolute correlation
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features correlated with Error Magnitude:")
    for name, val in sorted_corr[:5]:
        print(f"  {name}: {val:.4f}")

    # =========================================================================
    # 8. Conditional Submission
    # =========================================================================
    print("\n[Step 6/6] Checking Submission Criteria...")

    TARGET_METRIC = 0.6782

    if best_mcc > TARGET_METRIC:
        print(
            f"Metric ({best_mcc:.4f}) > Target ({TARGET_METRIC}). Generating submission..."
        )

        # Clean up memory before inference
        del df_val, X_val, y_val, val_probs, ensemble
        gc.collect()

        # Run Inference Pipeline
        # We pass nrows=None to ensure we predict on the FULL test set
        predict_and_submit(load_cached_data=True, nrows=None)

    else:
        print(
            f"Metric ({best_mcc:.4f}) <= Target ({TARGET_METRIC}). Skipping submission."
        )

    print("\nPipeline Complete.")


if __name__ == "__main__":
    main()
