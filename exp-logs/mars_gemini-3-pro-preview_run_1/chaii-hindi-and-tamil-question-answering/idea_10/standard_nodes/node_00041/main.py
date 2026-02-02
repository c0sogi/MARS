import os
import sys
import pandas as pd
import numpy as np
import torch
import glob
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Import provided library modules
from library.config import Config, set_seed
from library.model import MultiTaskXLMR
from library.trainer import Trainer
from library.data import create_loaders, QADataset, get_processed_data
from library.inference import predict_ensemble, post_process_predictions

# =============================================================================
# Configuration Overrides for Fast Baseline
# =============================================================================
# Limit training to ensure completion within 2 hours
Config.EPOCHS = 1
Config.SEEDS = [42, 43]  # Use 2 seeds for the baseline ensemble
Config.DEBUG = True
Config.DEBUG_SAMPLE_SIZE = 2000  # Cap samples to ensure speed while keeping enough data
Config.BATCH_SIZE = 4

# Ensure working directory is clean of previous caches
if os.path.exists(Config.WORKING_DIR):
    for f in glob.glob(os.path.join(Config.WORKING_DIR, "*.parquet")):
        os.remove(f)


# =============================================================================
# Helper Functions
# =============================================================================
def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    denom = len(a) + len(b) - len(c)
    return float(len(c)) / denom if denom > 0 else 0.0


def main():
    print("Initializing Fast Baseline Run...")

    # =========================================================================
    # 1. Data Setup (Train on Train only, Hold-out Val)
    # =========================================================================
    # The library is designed to merge Train+Val for the "Full-Data" idea.
    # For this baseline, we must validate on Val, so we prevent the merge.
    # We create an empty CSV and point VAL_META_PATH to it during training setup.

    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    empty_val_path = os.path.join(Config.WORKING_DIR, "empty_val.csv")
    pd.DataFrame(
        columns=["id", "context", "question", "answer_text", "answer_start", "language"]
    ).to_csv(empty_val_path, index=False)

    real_val_path = Config.VAL_META_PATH
    Config.VAL_META_PATH = empty_val_path

    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_CHECKPOINT)

    # Load training data (Train Only)
    # Force load_cached_data=False to ensure we use the modified path configuration
    print("Loading training data...")
    train_loader, _, _ = create_loaders(tokenizer, load_cached_data=False)

    # =========================================================================
    # 2. Training Loop
    # =========================================================================
    device = Config.DEVICE
    print(f"Training on device: {device}")

    for seed in Config.SEEDS:
        set_seed(seed)
        print(f"\n--- Training Seed {seed} ---")
        model = MultiTaskXLMR(Config.MODEL_CHECKPOINT)
        trainer = Trainer(model, train_loader, device, seed)
        trainer.train()

        # Cleanup to save memory
        del model, trainer
        torch.cuda.empty_cache()

    # =========================================================================
    # 3. Validation
    # =========================================================================
    print("\n--- Starting Validation ---")

    # Restore Val path to point to the real validation set
    # We treat the validation set as a "test" set for the inference pipeline
    # to generate predictions without using labels during the forward pass.
    Config.TEST_META_PATH = real_val_path

    # Process Validation Data
    # Force reload to overwrite cache with validation data
    val_features_df = get_processed_data(tokenizer, mode="test", load_cached_data=False)

    # Create DataLoader manually
    val_data = {col: val_features_df[col].tolist() for col in val_features_df.columns}
    val_dataset = QADataset(val_data, mode="test")
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Run Inference
    start_logits, end_logits, rel_logits = predict_ensemble(val_loader, device)

    # Post-process to get prediction strings
    val_preds_df = post_process_predictions(
        val_features_df, start_logits, end_logits, rel_logits
    )

    # Load Ground Truth
    val_gt_df = pd.read_csv(real_val_path)

    # Merge and Evaluate
    merged = pd.merge(val_gt_df, val_preds_df, on="id", how="left")
    merged["PredictionString"] = merged["PredictionString"].fillna("")

    scores = []
    for _, row in merged.iterrows():
        gt = str(row["answer_text"])
        dt = str(row["PredictionString"])
        scores.append(jaccard(gt, dt))

    final_metric = np.mean(scores)
    print(f"Final Validation Metric: {final_metric}")

    # =========================================================================
    # 4. Failure Analysis
    # =========================================================================
    print("\n--- Failure Analysis ---")
    merged["jaccard"] = scores
    merged["error"] = 1.0 - merged["jaccard"]

    # Compute lengths for correlation analysis
    merged["context_len"] = merged["context"].astype(str).apply(len)
    merged["question_len"] = merged["question"].astype(str).apply(len)

    corr_ctx = merged["error"].corr(merged["context_len"])
    corr_que = merged["error"].corr(merged["question_len"])

    print(f"Correlation (Error vs Context Len): {corr_ctx:.4f}")
    print(f"Correlation (Error vs Question Len): {corr_que:.4f}")

    # =========================================================================
    # 5. Submission
    # =========================================================================
    if final_metric > 0.60025:
        print("\nMetric check passed. Generating submission...")

        # Point TEST_META_PATH back to the actual test file
        Config.TEST_META_PATH = os.path.join(Config.METADATA_DIR, "test.csv")

        # Process Test Data
        # Force reload to overwrite cache with actual test data
        test_features_df = get_processed_data(
            tokenizer, mode="test", load_cached_data=False
        )

        test_data = {
            col: test_features_df[col].tolist() for col in test_features_df.columns
        }
        test_dataset = QADataset(test_data, mode="test")
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        )

        # Run Inference
        start_logits, end_logits, rel_logits = predict_ensemble(test_loader, device)

        # Post-process
        sub_df = post_process_predictions(
            test_features_df, start_logits, end_logits, rel_logits
        )

        # Save
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(f"\nMetric {final_metric} <= 0.60025. Skipping submission.")


if __name__ == "__main__":
    main()
