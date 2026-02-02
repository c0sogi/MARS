import sys
import os
import pandas as pd
import numpy as np
import torch
from library.config import Config
from library.utils import set_seed
from library.hfbb_engine import HFBBModel
from library.trainer import train_model
from library.inference import HybridPredictor, generate_submission
from library.text_processing import is_semiotic


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Override Config for a fast baseline execution that fits within time limits
    # while leveraging the A100 GPU capability.
    Config.EPOCHS = 3  # Reduced from 20 to ensure completion
    Config.N_FOLDS = 3  # Reduced from 5 to speed up Jackknife process
    Config.BATCH_SIZE = 32  # Reduced to fit V100 memory

    # Setup directories and seeds
    Config.setup_dirs()
    set_seed()

    print("=== Configuration ===")
    print(
        f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}"
    )
    print(f"Epochs: {Config.EPOCHS}")
    print(f"Batch Size: {Config.BATCH_SIZE}")
    print(f"Folds: {Config.N_FOLDS}")
    print("=====================\n")

    # ==========================================
    # 2. Tier 1: HFBB Model Initialization
    # ==========================================
    print("=== Step 1: Building/Loading Tier 1 (HFBB) Model ===")
    # The HFBB model is the statistical backbone. We need it loaded/built
    # before we can rely on it for inference or residual analysis.
    hfbb = HFBBModel()
    hfbb.build(load_cached_data=True)

    # ==========================================
    # 3. Tier 2: Transformer Training
    # ==========================================
    print("\n=== Step 2: Training Tier 2 (Transformer) on Residuals ===")
    # train_model handles the end-to-end process:
    # - Checks for cached tokenizers/datasets
    # - If missing, runs DatasetBuilder (Jackknife) to find residuals
    # - Trains the Transformer on those residuals
    # - Saves the best model checkpoint
    train_model(load_cached_data=True)

    # ==========================================
    # 4. Validation & Failure Analysis
    # ==========================================
    print("\n=== Step 3: Validation & Failure Analysis ===")

    # Load validation metadata
    if not os.path.exists(Config.VAL_FILE):
        raise FileNotFoundError(f"Validation file not found at {Config.VAL_FILE}")

    val_df = pd.read_csv(Config.VAL_FILE)
    print(f"Loaded {len(val_df)} validation samples.")

    # Initialize the Hybrid Predictor (loads HFBB and Best Transformer)
    predictor = HybridPredictor()
    predictor.load_resources()

    print("Running inference on validation set...")
    # Predict using the hybrid cascade logic
    # Note: predict() handles context generation and routing internally
    val_preds_series = predictor.predict(val_df)

    # Align predictions with targets
    # Ensure we handle potential NaNs by converting to empty strings (though pipeline handles this)
    val_preds = val_preds_series.fillna("").astype(str)
    val_targets = val_df["after"].fillna("").astype(str)

    # Calculate Accuracy (Exact String Match)
    # Pandas alignment ensures indices match even if predictor sorted internally
    correct_mask = val_preds == val_targets
    accuracy = correct_mask.mean()

    # PRINT REQUIRED METRIC
    print(f"Final Validation Metric: {accuracy}")

    # --- Failure Analysis ---
    print("\n--- Failure Analysis ---")

    # Prepare analysis dataframe
    analysis_df = val_df.copy()
    analysis_df["is_error"] = ~correct_mask
    analysis_df["error_int"] = analysis_df["is_error"].astype(int)

    # 1. Error Rate by Class
    if "class" in analysis_df.columns:
        print("Error Rate by Class (Top 10):")
        class_errors = (
            analysis_df.groupby("class")["is_error"].mean().sort_values(ascending=False)
        )
        print(class_errors.head(10))

    # 2. Correlation: Error vs Input Length
    analysis_df["len_before"] = analysis_df["before"].fillna("").astype(str).apply(len)
    corr_len = analysis_df["len_before"].corr(analysis_df["error_int"])
    print(f"Correlation (Input Length vs Error): {corr_len}")

    # 3. Correlation: Error vs Semiotic Status
    # Semiotic tokens (digits/latin) are the main targets for normalization
    analysis_df["is_semiotic"] = (
        analysis_df["before"].fillna("").astype(str).apply(is_semiotic).astype(int)
    )
    corr_semiotic = analysis_df["is_semiotic"].corr(analysis_df["error_int"])
    print(f"Correlation (Is Semiotic vs Error): {corr_semiotic}")

    # ==========================================
    # 5. Submission Generation
    # ==========================================
    THRESHOLD = 0.9784022349361615

    if accuracy > THRESHOLD:
        print(
            f"\nValidation accuracy {accuracy} > {THRESHOLD}. Proceeding to submission generation..."
        )
        # generate_submission handles loading test data, predicting, and saving to CSV
        generate_submission(load_cached_data=True)
    else:
        print(f"\nValidation accuracy {accuracy} <= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
