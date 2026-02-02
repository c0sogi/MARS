import os
import sys
import shutil
import pandas as pd
import numpy as np
import torch

# Ensure the current directory is in the python path to allow library imports
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything
from library.engine import train_symbolic, train_neural
from library.pipeline import HybridPredictor


def main():
    # --- 1. Configuration ---
    # Initialize configuration with overrides for a fast but effective baseline.
    # - plain_subset_ratio=0.05: Limits the amount of 'background' text to 5% to speed up data processing
    #   and training, while still providing enough context for the neural model.
    # - epochs=3: Sufficient for the character-level model to learn basic normalization patterns.
    # - batch_size=1024: Maximizes throughput on the available A100 GPU.
    config = Config(
        working_dir="./working/run_baseline",
        plain_subset_ratio=0.05,
        epochs=3,
        batch_size=256,
        learning_rate=3e-4,
        seed=42,
    )

    # Ensure submission directory exists
    os.makedirs("./submission", exist_ok=True)

    # Set reproducibility seeds
    seed_everything(config.seed)

    print("--- Starting Fast Baseline Pipeline ---")

    # --- 2. Training Components ---

    # Step 2a: Train Symbolic Memory
    # This builds the N-gram stats (Trigram -> Bigram -> Unigram) which act as the primary solver.
    print("\n[1/5] Training Symbolic Memory...")
    train_symbolic(config)

    # Step 2b: Train Neural Model
    # This trains the Char-Transformer for handling complex semiotic classes (numbers, dates).
    print("\n[2/5] Training Neural Model...")
    train_neural(config)

    # --- 3. Validation ---
    print("\n[3/5] Validating Hybrid Pipeline...")

    # Load validation metadata
    val_path = os.path.join(config.metadata_dir, "val.csv")
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Validation metadata not found at {val_path}")

    df_val = pd.read_csv(val_path)

    # Initialize the Hybrid Predictor (loads both Symbolic Memory and Neural Model)
    predictor = HybridPredictor(config)

    # Generate predictions on the validation set
    print(f"Running inference on {len(df_val)} validation samples...")
    df_preds = predictor.predict(df_val)

    # Prepare for comparison
    # Construct the unique ID for merging: sentence_id + "_" + token_id
    df_val["id"] = (
        df_val["sentence_id"].astype(str) + "_" + df_val["token_id"].astype(str)
    )

    # Merge predictions (df_preds has 'id', 'after') with ground truth (df_val has 'after')
    comparison = df_val.merge(df_preds, on="id", suffixes=("_true", "_pred"))

    # Calculate Accuracy (Exact String Match)
    comparison["correct"] = comparison["after_true"] == comparison["after_pred"]
    accuracy = comparison["correct"].mean()

    # Print the required metric
    print(f"Final Validation Metric: {accuracy}")

    # --- 4. Failure Analysis ---
    print("\n[4/5] Performing Failure Analysis...")

    # Create Error flag (1 = Incorrect, 0 = Correct)
    comparison["error"] = (~comparison["correct"]).astype(int)

    # Feature Engineering for Analysis
    # 1. Input Token Length
    comparison["len_before"] = comparison["before"].astype(str).str.len()
    # 2. Presence of Digits (Proxy for semiotic complexity)
    comparison["has_digit"] = (
        comparison["before"].astype(str).str.contains(r"\d").astype(int)
    )

    # Calculate Correlations
    corr_len = comparison["error"].corr(comparison["len_before"])
    corr_digit = comparison["error"].corr(comparison["has_digit"])

    print(f"Correlation between Error and Input Length: {corr_len}")
    print(f"Correlation between Error and Has Digit: {corr_digit}")

    # --- 5. Submission Generation ---
    print("\n[5/5] Checking Submission Criteria...")

    threshold = 0.9798075665557208

    if accuracy > threshold:
        print(
            f"Validation accuracy ({accuracy}) exceeds threshold ({threshold}). Generating submission..."
        )

        # Generate submission file (saved to config.submission_path by default)
        predictor.generate_submission()

        # Move the file to the required location: ./submission/submission.csv
        src_path = config.submission_path
        dst_path = "./submission/submission.csv"

        if os.path.exists(src_path):
            shutil.move(src_path, dst_path)
            print(f"Submission successfully saved to {dst_path}")
        else:
            print(f"Error: Generated submission file not found at {src_path}")

    else:
        print(
            f"Validation accuracy ({accuracy}) does not exceed threshold ({threshold}). Submission skipped."
        )

    print("\nPipeline execution complete.")


if __name__ == "__main__":
    main()
