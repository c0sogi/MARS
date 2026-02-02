import os
import sys
import torch
import numpy as np
import pandas as pd
import shutil
import warnings

# Filter warnings for cleaner output
warnings.filterwarnings("ignore")

# Import library components
from library.config import Config
from library.utils import seed_everything, compute_spearmanr
from library.dataset import get_dataloaders, QuestDataset
from library.model import SymmetricDualEncoder
from library.engine import run_training


def run_demo():
    print("=== Starting Library Usage Demonstration ===\n")

    # ---------------------------------------------------------
    # 1. Configuration & Setup
    # ---------------------------------------------------------
    print("[1] Configuring environment for rapid demo execution...")

    # Patch Config for speed and isolation
    Config.WORKING_DIR = "./working/demo_execution"
    Config.SUBMISSION_DIR = "./working/demo_execution/submission"
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Reduce computational load
    Config.MAX_LEN = 32  # Short sequence length for speed
    Config.TRAIN_BATCH_SIZE = 4  # Small batch size
    Config.VALID_BATCH_SIZE = 8
    Config.ACCUMULATION_STEPS = 1
    Config.ACTUAL_EPOCHS = 1  # Only 1 epoch
    Config.PHANTOM_EPOCHS = 1

    # Ensure directories exist
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    seed_everything(Config.SEED)
    print("    Configuration patched successfully.")

    # ---------------------------------------------------------
    # 2. Metric Verification
    # ---------------------------------------------------------
    print("\n[2] Verifying Metric Logic (Spearman's Correlation)...")

    # Case A: Perfect Positive Correlation
    y_true = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    y_pred_perfect = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    score_perfect = compute_spearmanr(y_true, y_pred_perfect)

    # Case B: Perfect Negative Correlation
    y_pred_neg = np.array([[0.5, 0.6], [0.3, 0.4], [0.1, 0.2]])
    score_neg = compute_spearmanr(y_true, y_pred_neg)

    print(f"    Perfect Score: {score_perfect:.4f} (Expected: 1.0)")
    print(f"    Negative Score: {score_neg:.4f} (Expected: -1.0)")

    assert np.isclose(
        score_perfect, 1.0
    ), "Metric check failed: Perfect correlation should be 1.0"
    assert np.isclose(
        score_neg, -1.0
    ), "Metric check failed: Negative correlation should be -1.0"
    print("    Metric verification passed.")

    # ---------------------------------------------------------
    # 3. Data Pipeline Verification
    # ---------------------------------------------------------
    print("\n[3] Verifying Data Pipeline...")

    # Get debug dataloaders (loads subset of 100 rows)
    train_loader, val_loader, test_loader = get_dataloaders(
        debug=True, load_cached_data=False
    )

    # Fetch one batch to verify structure
    batch = next(iter(train_loader))

    required_keys = [
        "input_ids_q",
        "attention_mask_q",
        "input_ids_a",
        "attention_mask_a",
        "labels",
    ]
    for key in required_keys:
        assert key in batch, f"Batch missing key: {key}"

    # Check shapes
    # Input IDs: [Batch, SeqLen]
    assert batch["input_ids_q"].shape == (
        Config.TRAIN_BATCH_SIZE,
        Config.MAX_LEN,
    ), f"Incorrect input shape: {batch['input_ids_q'].shape}"
    # Labels: [Batch, 30]
    assert batch["labels"].shape == (
        Config.TRAIN_BATCH_SIZE,
        30,
    ), f"Incorrect label shape: {batch['labels'].shape}"

    print(
        f"    Batch loaded successfully. Input shape: {batch['input_ids_q'].shape}, Targets: {batch['labels'].shape}"
    )

    # ---------------------------------------------------------
    # 4. Model Architecture Verification
    # ---------------------------------------------------------
    print("\n[4] Verifying Model Architecture...")

    device = torch.device(Config.DEVICE)
    model = SymmetricDualEncoder()
    model.to(device)
    model.eval()

    # Move batch to device
    inputs = {
        k: v.to(device)
        for k, v in batch.items()
        if k in ["input_ids_q", "attention_mask_q", "input_ids_a", "attention_mask_a"]
    }

    with torch.no_grad():
        outputs = model(**inputs)

    # Check output shape [Batch, 30]
    assert outputs.shape == (
        Config.TRAIN_BATCH_SIZE,
        30,
    ), f"Model output shape mismatch. Expected ({Config.TRAIN_BATCH_SIZE}, 30), got {outputs.shape}"

    print("    Model forward pass successful. Output shape verified.")

    # ---------------------------------------------------------
    # 5. Full Engine Execution (Train/Eval/Infer)
    # ---------------------------------------------------------
    print("\n[5] Executing Full Training Engine (Debug Mode)...")

    # run_training handles the loop, saving, and submission generation
    # We use debug=True to ensure it uses the subset logic
    best_score = run_training(debug=True)

    print(f"    Training complete. Best Validation Score: {best_score:.4f}")

    # ---------------------------------------------------------
    # 6. Submission Verification
    # ---------------------------------------------------------
    print("\n[6] Verifying Submission Artifacts...")

    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)

    # Check dimensions
    # Debug mode uses 100 rows for test set as well
    expected_rows = 100
    assert (
        len(sub_df) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(sub_df)}"

    # Check columns
    expected_cols = ["qa_id"] + Config.TARGET_COLS
    assert (
        list(sub_df.columns) == expected_cols
    ), "Submission columns do not match requirements."

    # Check value range [0, 1]
    # We exclude qa_id from this check
    preds = sub_df[Config.TARGET_COLS].values
    assert (preds >= 0).all() and (
        preds <= 1
    ).all(), "Predictions contain values outside [0, 1] range."

    print("    Submission format verified successfully.")
    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
