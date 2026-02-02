import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings
from transformers import get_linear_schedule_with_warmup

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import library components
from library.config import Config
from library.utils import seed_everything, compute_spearman_correlation
from library.data import get_dataloaders
from library.model import DistilRobertaDualEncoder
from library.engine import get_optimizer_params, train_fn, eval_fn, generate_submission


def main():
    print("=== Starting Demonstration Script ===")

    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Modify Config for a fast demonstration run
    print("[Setup] Configuring parameters for fast execution...")
    Config.EPOCHS = 1
    Config.DEBUG = True  # Truncates data: Train=100, Val=50, Test=50
    Config.TRAIN_BATCH_SIZE = 8
    Config.VALID_BATCH_SIZE = 16
    Config.NUM_WORKERS = 2  # Reduce workers for simple demo

    # Set seeds for reproducibility
    seed_everything(Config.SEED)

    device = Config.DEVICE
    print(f"[Setup] Device selected: {device}")

    # ==========================================
    # 2. Logic Verification: Metric
    # ==========================================
    print("[Verification] Testing Spearman Correlation metric logic...")
    # Create dummy predictions and targets
    # Case 1: Perfect correlation
    dummy_preds = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
    dummy_targets = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
    score_perfect = compute_spearman_correlation(dummy_preds, dummy_targets)

    # Case 2: Inverse correlation (should be -1)
    dummy_preds_inv = np.array([[0.9], [0.8], [0.7], [0.6]])
    dummy_targets_inv = np.array([[0.1], [0.2], [0.3], [0.4]])
    score_inv = compute_spearman_correlation(dummy_preds_inv, dummy_targets_inv)

    assert np.isclose(
        score_perfect, 1.0
    ), f"Metric Logic Error: Expected 1.0, got {score_perfect}"
    assert np.isclose(
        score_inv, -1.0
    ), f"Metric Logic Error: Expected -1.0, got {score_inv}"
    print("[Verification] Metric logic verified.")

    # ==========================================
    # 3. Data Loading
    # ==========================================
    print("[Data] Loading and processing data (Debug Mode)...")
    # load_cached_data=True attempts to use existing parquet files.
    # debug=True will truncate the dataframes after loading/processing.
    train_loader, val_loader, test_loader, target_cols = get_dataloaders(
        load_cached_data=False, debug=True
    )

    print(f"[Data] Train batches: {len(train_loader)}")
    print(f"[Data] Val batches: {len(val_loader)}")
    print(f"[Data] Test batches: {len(test_loader)}")

    # ==========================================
    # 4. Model Initialization
    # ==========================================
    print("[Model] Initializing DistilRobertaDualEncoder...")
    model = DistilRobertaDualEncoder()
    model.to(device)

    # Verify Forward Pass Dimensions
    print("[Verification] Verifying model output dimensions...")
    dummy_batch = next(iter(train_loader))
    b_ids_q = dummy_batch["input_ids_q"].to(device)
    b_mask_q = dummy_batch["attention_mask_q"].to(device)
    b_ids_a = dummy_batch["input_ids_a"].to(device)
    b_mask_a = dummy_batch["attention_mask_a"].to(device)

    with torch.no_grad():
        logits = model(b_ids_q, b_mask_q, b_ids_a, b_mask_a)

    expected_shape = (b_ids_q.size(0), 30)
    assert (
        logits.shape == expected_shape
    ), f"Model Output Error: Expected shape {expected_shape}, got {logits.shape}"
    print("[Verification] Model forward pass verified.")

    # ==========================================
    # 5. Optimizer & Training Setup
    # ==========================================
    optimizer_params = get_optimizer_params(model)
    optimizer = torch.optim.AdamW(optimizer_params)

    num_training_steps = len(train_loader) * Config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_training_steps * Config.WARMUP_RATIO),
        num_training_steps=num_training_steps,
    )

    # ==========================================
    # 6. Training Loop
    # ==========================================
    print("[Training] Starting training loop...")
    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_fn(train_loader, model, optimizer, device, scheduler)
        print(f"  Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.4f}")

        # Validation
        val_loss, val_spearman = eval_fn(val_loader, model, device)
        print(
            f"  Epoch {epoch+1}/{Config.EPOCHS} | Val Loss: {val_loss:.4f} | Val Spearman: {val_spearman:.4f}"
        )

    # ==========================================
    # 7. Submission Generation
    # ==========================================
    print("[Inference] Generating submission...")
    generate_submission(test_loader, model, device)

    # ==========================================
    # 8. Submission Verification
    # ==========================================
    print("[Verification] Verifying submission file...")
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)

    # Check 1: Row count (Debug mode uses 50 test samples)
    expected_rows = 50
    assert (
        len(sub_df) == expected_rows
    ), f"Submission Error: Expected {expected_rows} rows, got {len(sub_df)}"

    # Check 2: Columns
    expected_columns = ["qa_id"] + target_cols
    assert (
        list(sub_df.columns) == expected_columns
    ), "Submission Error: Column names mismatch."

    # Check 3: Value Range [0, 1]
    # We select only target columns
    pred_values = sub_df[target_cols].values
    if not ((pred_values >= 0) & (pred_values <= 1)).all():
        raise AssertionError(
            "Submission Error: Predictions contain values outside [0, 1]."
        )

    print("[Verification] Submission file is valid.")
    print("=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
