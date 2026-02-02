import sys
import os
import pandas as pd
import numpy as np
import torch
import tqdm


# --- 1. Environment Setup & Patching ---
# Patch tqdm to suppress progress bars as required
def silent_tqdm(iterable=None, *args, **kwargs):
    if iterable is None:
        # Return a dummy object that has update/close methods for manual tqdm usage
        class DummyTqdm:
            def update(self, n=1):
                pass

            def close(self):
                pass

        return DummyTqdm()
    return iterable


tqdm.tqdm = silent_tqdm

# Import library modules
from library.config import Config
from library.trainer import run_training
from library.hybrid_system import HybridPredictor

# --- 2. Configuration Override ---
# Adjust settings for a fast baseline execution within 2 hours
# We train on a subset of the 'difficult' (digit-containing) tokens
Config.DEBUG_SUBSET_SIZE = 150000
Config.NUM_EPOCHS = 3
Config.BATCH_SIZE = 512  # Utilize A100 memory
Config.NUM_WORKERS = 4
Config.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    # Set seeds for reproducibility (already handled in trainer, but good practice)
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)

    # --- 3. Training Phase ---
    print("Starting Training Phase...")
    # run_training handles tokenizer building, N-gram stats generation, and neural model training
    run_training(load_cached_data=True)

    # --- 4. Validation Phase ---
    print("Starting Validation Phase...")

    # Initialize Predictor (loads cached artifacts)
    predictor = HybridPredictor(load_cached_data=True)

    # Load Validation Data
    print(f"Loading validation data from {Config.VAL_FILE}...")
    val_df = pd.read_csv(Config.VAL_FILE, dtype=str)

    # Preprocess validation data
    val_df["before"] = val_df["before"].fillna("")
    val_df["after"] = val_df["after"].fillna("")
    val_df["sentence_id"] = val_df["sentence_id"].astype(int)
    val_df["token_id"] = val_df["token_id"].astype(int)

    # Prepare context columns (prev, next, etc.)
    val_df = predictor._prepare_context(val_df)

    # Determine routing (Neural vs N-gram)
    neural_mask = predictor._get_neural_candidates(val_df)
    ngram_mask = ~neural_mask

    print(f"Validation Split: {neural_mask.sum()} Neural vs {ngram_mask.sum()} N-gram")

    # Initialize prediction column
    val_df["pred"] = ""

    # Inference: Path A (N-gram)
    if ngram_mask.any():
        val_df.loc[ngram_mask, "pred"] = predictor._predict_ngrams(val_df[ngram_mask])

    # Inference: Path B (Neural)
    if neural_mask.any():
        val_df.loc[neural_mask, "pred"] = predictor._predict_neural(val_df[neural_mask])

    # Compute Metric
    correct_mask = val_df["pred"] == val_df["after"]
    accuracy = correct_mask.mean()

    # Print required metric
    print(f"Final Validation Metric: {accuracy}")

    # --- 5. Failure Analysis ---
    print("\n--- Failure Analysis ---")
    val_df["is_error"] = (~correct_mask).astype(int)
    val_df["len_before"] = val_df["before"].str.len()

    # Correlation Analysis
    corr_len = val_df["is_error"].corr(val_df["len_before"])
    print(f"Correlation (Error vs Input Length): {corr_len}")

    # Class Analysis
    if "class" in val_df.columns:
        print("\nError Rate by Class (Top 5):")
        class_stats = val_df.groupby("class")["is_error"].agg(["mean", "sum", "count"])
        print(class_stats.sort_values("mean", ascending=False).head(5))

        print("\nMost Frequent Error Classes:")
        print(class_stats.sort_values("sum", ascending=False).head(5))

    # --- 6. Submission ---
    TARGET_METRIC = 0.9784130832472395

    if accuracy > TARGET_METRIC:
        print(f"\nMetric {accuracy} > {TARGET_METRIC}. Generating submission...")
        predictor.generate_submission(
            test_file=Config.TEST_FILE, submission_path=Config.SUBMISSION_PATH
        )
    else:
        print(f"\nMetric {accuracy} <= {TARGET_METRIC}. Skipping submission.")


if __name__ == "__main__":
    main()
