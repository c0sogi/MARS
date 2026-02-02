import sys
import os
import pandas as pd
import numpy as np
import joblib
import torch
from scipy.stats import spearmanr, pearsonr

# -----------------------------------------------------------------------------
# 1. Configuration & Data Patching
# -----------------------------------------------------------------------------
# We import library.config first to modify paths/params before other modules load them
import library.config


def setup_data_and_config():
    """
    Checks training data size and subsamples if necessary to ensure fast execution.
    Monkey-patches library.config to use the subsampled data.
    """
    print("Checking training data size...")
    train_path = library.config.TRAIN_METADATA_PATH

    if not os.path.exists(train_path):
        print(f"Warning: Train metadata not found at {train_path}")
        return

    # Load only the first column to check length quickly, or just read it
    df = pd.read_csv(train_path)
    n_rows = len(df)
    print(f"Original training set size: {n_rows}")

    # Limit to 6000 samples for the 'fast baseline' requirement
    MAX_SAMPLES = 6000

    if n_rows > MAX_SAMPLES:
        print(f"Dataset exceeds {MAX_SAMPLES} samples. Subsampling...")
        df_subset = df.sample(n=MAX_SAMPLES, random_state=42)

        subset_path = os.path.join(
            library.config.WORKING_DIR, "train_metadata_subset.csv"
        )
        df_subset.to_csv(subset_path, index=False)

        # Patch the config path
        print(f"Patching TRAIN_METADATA_PATH to {subset_path}")
        library.config.TRAIN_METADATA_PATH = subset_path
    else:
        print("Dataset size is within limits. Using full training set.")


# Execute setup before importing other library modules
setup_data_and_config()

# -----------------------------------------------------------------------------
# 2. Import Library Modules
# -----------------------------------------------------------------------------
# Now it is safe to import modules that depend on library.config
from library import trainer
from library import feature_caching
from library import refinement
from library import inference
from library.trainer import compute_spearman_metric


# -----------------------------------------------------------------------------
# 3. Main Execution Flow
# -----------------------------------------------------------------------------
def main():
    print("=" * 60)
    print(" STARTING PIPELINE")
    print("=" * 60)

    # --- Step 1: Stage 1 Training (Fine-tuning) ---
    print("\n[Step 1] Running Stage 1: End-to-End Fine-tuning...")
    # debug=False ensures we use the data defined in config (potentially subsampled), not the 100-row debug set
    trainer.train_stage_1(debug=False, load_cached_data=True)

    # --- Step 2: Feature Caching ---
    print("\n[Step 2] Caching Interaction Features...")
    # Force extraction (load_cached_data=False) to ensure we use the features
    # from the model we just fine-tuned in Step 1.
    feature_caching.cache_features(debug=False, load_cached_data=False)

    # --- Step 3: Stage 2 Training (Ridge Refinement) ---
    print("\n[Step 3] Running Stage 2: Ridge Regression Refinement...")
    refinement.train_ridge_head(load_cached_model=False)

    # --- Step 4: Validation Assessment & Failure Analysis ---
    print("\n[Step 4] Validation Assessment & Failure Analysis...")

    # Load artifacts for manual analysis
    X_val = np.load(library.config.VAL_FEATURES_PATH)
    y_val = np.load(library.config.VAL_TARGETS_PATH)
    ridge_model = joblib.load(library.config.RIDGE_MODEL_PATH)

    # Predict
    val_preds = ridge_model.predict(X_val)
    val_preds = np.clip(val_preds, 0, 1)

    # Compute Final Metric
    final_metric = compute_spearman_metric(val_preds, y_val)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate Mean Absolute Error per sample (averaged across 30 targets)
    # Shape: (N_val, 30) -> (N_val,)
    row_errors = np.mean(np.abs(val_preds - y_val), axis=1)

    # Load validation metadata for feature correlation
    val_df = pd.read_csv(library.config.VAL_METADATA_PATH)

    # Ensure alignment (metadata should match cached features order)
    if len(val_df) == len(row_errors):
        # Extract features
        q_lens = val_df["question_body"].fillna("").astype(str).str.len()
        a_lens = val_df["answer"].fillna("").astype(str).str.len()

        # Compute correlations
        # We use Pearson for linear correlation between error magnitude and length
        corr_q, _ = pearsonr(row_errors, q_lens)
        corr_a, _ = pearsonr(row_errors, a_lens)

        print(f"Correlation between Error and Question Length: {corr_q:.4f}")
        print(f"Correlation between Error and Answer Length:   {corr_a:.4f}")

        if abs(corr_q) > 0.1 or abs(corr_a) > 0.1:
            print(
                "Observation: Model performance is noticeably correlated with text length."
            )
    else:
        print(
            "Warning: Validation metadata length mismatch. Skipping detailed failure analysis."
        )

    # --- Step 5: Submission ---
    THRESHOLD = 0.39286571872869314

    if final_metric > THRESHOLD:
        print(
            f"\n[Step 5] Metric ({final_metric:.6f}) > Threshold ({THRESHOLD:.6f}). Generating Submission..."
        )
        inference.predict_and_submit(load_cached_data=True, debug=False)
    else:
        print(
            f"\n[Step 5] Metric ({final_metric:.6f}) <= Threshold ({THRESHOLD:.6f}). Skipping Submission."
        )

    print("\nPipeline Completed.")


if __name__ == "__main__":
    # Ensure reproducibility
    library.config.seed_everything()
    main()
