import pandas as pd
import numpy as np
import os
import sys

# Import provided library modules
from library.config import Config
from library.utils import set_seed, load_data
from library.train_eval import Trainer
from library.pipeline import HybridInference


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Override Config for Fast Baseline execution
    # Reducing epochs and upsampling count to ensure completion within strict time limits
    Config.EPOCHS = 3
    Config.UPSAMPLE_TARGET_COUNT = 50000

    # Ensure working directories exist
    Config.setup_directories()
    set_seed(Config.SEED)

    print("=== Text Normalization Pipeline ===")
    print(f"Device: {Config.DEVICE}")
    print(
        f"Configuration: Epochs={Config.EPOCHS}, Upsample Target={Config.UPSAMPLE_TARGET_COUNT}"
    )

    # ==========================================
    # 2. Training
    # ==========================================
    print("\n--- Step 1: Training Model ---")
    trainer = Trainer(device=Config.DEVICE)
    # Execute training. load_cached_data=True allows using existing artifacts if available,
    # but since we are in a new working dir (idea_5), this will likely trigger processing and training.
    trainer.run(epochs=Config.EPOCHS, load_cached_data=True)

    # ==========================================
    # 3. Validation & Evaluation
    # ==========================================
    print("\n--- Step 2: Validation ---")
    val_df = load_data("val")

    # Initialize Inference Engine (Loads HFBB and the trained Neural Model)
    inference = HybridInference(device=Config.DEVICE)
    inference.load_resources(load_cached_data=True)

    print(f"Running inference on {len(val_df)} validation tokens...")
    # Predict on validation set
    val_preds = inference.predict(val_df, batch_size=Config.BATCH_SIZE)

    # Construct 'id' column in ground truth for merging
    val_df["id"] = (
        val_df["sentence_id"].astype(str) + "_" + val_df["token_id"].astype(str)
    )

    # Merge predictions with ground truth
    merged = pd.merge(
        val_df[["id", "before", "after", "class"]],
        val_preds,
        on="id",
        how="inner",
        suffixes=("_true", "_pred"),
    )

    # Calculate Accuracy (Exact Match)
    merged["is_correct"] = merged["after_true"] == merged["after_pred"]
    accuracy = merged["is_correct"].mean()

    # Print Required Metric
    print(f"Final Validation Metric: {accuracy}")

    # ==========================================
    # 4. Failure Analysis
    # ==========================================
    print("\n--- Step 3: Failure Analysis ---")
    # Define Error (1 = Incorrect, 0 = Correct)
    merged["error"] = (~merged["is_correct"]).astype(int)

    # Feature: Input Length
    merged["len_before"] = merged["before"].astype(str).apply(len)

    # Correlation: Error vs Input Length
    corr_len = merged["error"].corr(merged["len_before"])
    print(f"Correlation (Error vs Input Length): {corr_len}")

    # Analysis by Class
    print("Top 5 Classes by Error Rate:")
    class_stats = (
        merged.groupby("class")
        .agg(error_rate=("error", "mean"), count=("error", "count"))
        .sort_values("error_rate", ascending=False)
    )
    print(class_stats.head(5))

    # ==========================================
    # 5. Submission
    # ==========================================
    print("\n--- Step 4: Submission Generation ---")
    THRESHOLD = 0.9784022349361615

    if accuracy > THRESHOLD:
        print(
            f"Validation accuracy {accuracy} exceeds threshold {THRESHOLD}. Generating submission..."
        )

        # Load Test Data
        test_df = load_data("test")

        # Run Inference
        submission_df = inference.predict(test_df, batch_size=Config.BATCH_SIZE)

        # Save Submission
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"Validation accuracy {accuracy} does not exceed threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
