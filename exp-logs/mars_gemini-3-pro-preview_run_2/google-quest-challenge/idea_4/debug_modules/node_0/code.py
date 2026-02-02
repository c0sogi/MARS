import os
import sys
import shutil
import warnings
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

# Import from the provided library
from library.config import Config, set_seed
from library.utils import compute_spearman_metric
from library.dataset import get_dataloader
from library.model import UnifiedDebertaSiamese
from library.engine import run_training, predict_and_submit


def main():
    # ==========================================
    # 1. Setup & Configuration Override
    # ==========================================
    print("Initializing demonstration...")

    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    # Override Config for speed and demonstration purposes
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 16  # Small sample size for quick execution
    Config.EPOCHS = 1  # Single epoch
    Config.TRAIN_BATCH_SIZE = 4
    Config.VALID_BATCH_SIZE = 4
    Config.MAX_LEN = 64  # Reduced sequence length for speed

    # Define and clean working directory for this run
    Config.WORKING_DIR = "./working/demo_run"
    Config.OUTPUT_DIR = "./working/demo_run"
    Config.SUBMISSION_PATH = "./working/demo_run/submission.csv"

    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seed for reproducibility
    set_seed(Config.SEED)
    device = Config.DEVICE
    print(f"Configuration set. Device: {device}")

    # ==========================================
    # 2. Verify Metric Logic
    # ==========================================
    print("\n[1/4] Verifying Metric Logic...")

    # Create dummy data
    # Case A: Perfect Correlation
    preds_perfect = np.random.rand(10, 30)
    targets_perfect = preds_perfect.copy()
    score_perfect = compute_spearman_metric(preds_perfect, targets_perfect)

    # Case B: Perfect Negative Correlation
    # (Spearman of x and -x is -1)
    targets_inverse = 1 - preds_perfect
    score_inverse = compute_spearman_metric(preds_perfect, targets_inverse)

    print(f"  Perfect Correlation Score: {score_perfect:.4f}")
    print(f"  Inverse Correlation Score: {score_inverse:.4f}")

    # Assertions
    assert np.isclose(
        score_perfect, 1.0
    ), "Metric failed: Expected 1.0 for identical inputs"
    assert np.isclose(
        score_inverse, -1.0
    ), "Metric failed: Expected -1.0 for inverse inputs"
    print("  Metric verification passed.")

    # ==========================================
    # 3. Verify Dataset & Tokenizer
    # ==========================================
    print("\n[2/4] Verifying Dataset and Tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # Load training data (load_cached_data=False ensures we process from scratch)
    train_loader = get_dataloader("train", tokenizer, load_cached_data=False)

    # Fetch one batch
    batch = next(iter(train_loader))

    # Verify keys
    expected_keys = [
        "q_input_ids",
        "q_attention_mask",
        "a_input_ids",
        "a_attention_mask",
        "labels",
    ]
    assert all(
        k in batch for k in expected_keys
    ), f"Missing keys in batch. Found: {batch.keys()}"

    # Verify shapes
    q_shape = batch["q_input_ids"].shape
    label_shape = batch["labels"].shape

    print(f"  Batch Q-Input Shape: {q_shape}")
    print(f"  Batch Label Shape: {label_shape}")

    assert q_shape == (Config.TRAIN_BATCH_SIZE, Config.MAX_LEN), "Incorrect input shape"
    assert label_shape == (
        Config.TRAIN_BATCH_SIZE,
        Config.NUM_LABELS,
    ), "Incorrect label shape"
    print("  Dataset verification passed.")

    # ==========================================
    # 4. Verify Model Architecture
    # ==========================================
    print("\n[3/4] Verifying Model Architecture...")

    model = UnifiedDebertaSiamese()
    model.to(device)
    model.eval()

    # Perform forward pass with the batch from step 3
    with torch.no_grad():
        q_ids = batch["q_input_ids"].to(device)
        q_mask = batch["q_attention_mask"].to(device)
        a_ids = batch["a_input_ids"].to(device)
        a_mask = batch["a_attention_mask"].to(device)

        logits = model(q_ids, q_mask, a_ids, a_mask)

    print(f"  Output Logits Shape: {logits.shape}")

    assert logits.shape == (
        Config.TRAIN_BATCH_SIZE,
        Config.NUM_LABELS,
    ), "Model output shape mismatch"
    print("  Model verification passed.")

    # ==========================================
    # 5. Verify Training & Inference Engine
    # ==========================================
    print("\n[4/4] Verifying Training and Inference Engine...")

    # Run the full training loop (uses the modified Config)
    # This will train for 1 epoch on 16 samples
    trained_model, tokenizer = run_training()

    # Check if best model was saved
    best_model_path = os.path.join(Config.OUTPUT_DIR, "best_model.pth")
    assert os.path.exists(best_model_path), "Best model checkpoint not found."
    print(f"  Training complete. Model saved to {best_model_path}")

    # Run inference
    predict_and_submit(trained_model, tokenizer)

    # Check submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found."

    submission_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"  Submission File Shape: {submission_df.shape}")

    # Validation:
    # 1. Check columns (qa_id + 30 targets)
    expected_cols = ["qa_id"] + Config.TARGET_COLS
    assert list(submission_df.columns) == expected_cols, "Submission columns mismatch"

    # 2. Check row count
    # Since DEBUG=True, the test set in get_dataset is also truncated to DEBUG_SAMPLE_SIZE (16)
    assert (
        len(submission_df) == Config.DEBUG_SAMPLE_SIZE
    ), f"Expected {Config.DEBUG_SAMPLE_SIZE} rows in debug submission, found {len(submission_df)}"

    # 3. Check value range [0, 1]
    # Drop qa_id for check
    preds_values = submission_df.drop(columns=["qa_id"]).values
    assert (preds_values >= 0).all() and (
        preds_values <= 1
    ).all(), "Predictions outside [0,1] range"

    print("  Inference verification passed.")
    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    main()
