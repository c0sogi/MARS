import sys
import os
import torch
import numpy as np
import pandas as pd
import shutil

# Ensure the library is in the path
sys.path.append(".")

# Import library components
from library.config import Config
from library.utils import set_seed, compute_spearmanr_metric, setup_logger
from library.data import get_dataloaders
from library.model import MultiScaleDualEncoder
from library.trainer import Trainer, create_submission


def main():
    print("=== Starting Demonstration Script ===")

    # 1. Setup and Configuration Override
    # We override Config attributes to ensure the script runs quickly (demo mode)
    print("\n[1] Configuring environment for rapid execution...")
    set_seed(42)

    # Override Config for speed
    Config.EPOCHS = 1
    Config.SCHEDULER_TOTAL_EPOCHS = 1
    Config.TRAIN_BATCH_SIZE = 4
    Config.VALID_BATCH_SIZE = 4
    Config.ACCUMULATION_STEPS = 1
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data

    # Use a specific working directory for this execution
    Config.WORKING_DIR = "./working/demo_execution"
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Update submission path to be inside the demo working dir
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Device: {Config.DEVICE}")

    # 2. Metric Verification
    print("\n[2] Verifying Metric Calculation...")
    # Case 1: Perfect correlation
    preds_perfect = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    targets_perfect = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    score_perfect = compute_spearmanr_metric(preds_perfect, targets_perfect)
    print(f"Perfect Correlation Score: {score_perfect}")
    assert np.isclose(score_perfect, 1.0), "Metric failed on perfect correlation"

    # Case 2: Negative correlation (for one column)
    # Col 0: 0.1, 0.3, 0.5 vs 0.5, 0.3, 0.1 -> -1.0
    # Col 1: 0.2, 0.4, 0.6 vs 0.2, 0.4, 0.6 -> +1.0
    # Mean: 0.0
    preds_mixed = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    targets_mixed = np.array([[0.5, 0.2], [0.3, 0.4], [0.1, 0.6]])
    score_mixed = compute_spearmanr_metric(preds_mixed, targets_mixed)
    print(f"Mixed Correlation Score: {score_mixed}")
    assert np.isclose(score_mixed, 0.0), "Metric failed on mixed correlation"
    print("Metric verification passed.")

    # 3. Data Loading
    print("\n[3] Loading Data (Debug Mode)...")
    # debug=True loads a small subset (100 rows for train, 50 for val/test)
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=False, debug=True
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Verify batch structure
    sample_batch = next(iter(train_loader))
    required_keys = [
        "q_input_ids",
        "q_attention_mask",
        "q_title_mask",
        "q_body_mask",
        "a_input_ids",
        "a_attention_mask",
        "qa_id",
        "labels",
    ]
    for key in required_keys:
        assert key in sample_batch, f"Missing key {key} in batch"
    print("Data loading and batch structure verified.")

    # 4. Model Initialization and Forward Pass
    print("\n[4] Initializing Model and Verifying Forward Pass...")
    model = MultiScaleDualEncoder()
    model.to(Config.DEVICE)
    model.eval()

    # Move sample batch to device
    for k, v in sample_batch.items():
        if isinstance(v, torch.Tensor):
            sample_batch[k] = v.to(Config.DEVICE)

    with torch.no_grad():
        logits = model(sample_batch)

    print(f"Logits Shape: {logits.shape}")
    assert logits.shape == (
        Config.TRAIN_BATCH_SIZE,
        Config.NUM_LABELS,
    ), f"Expected shape ({Config.TRAIN_BATCH_SIZE}, {Config.NUM_LABELS}), got {logits.shape}"
    print("Model forward pass verified.")

    # 5. Training Loop
    print("\n[5] Running Training Loop (1 Epoch)...")
    trainer = Trainer(model, train_loader, val_loader)
    trainer.fit()

    # Verify model checkpoint creation
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    assert os.path.exists(best_model_path), "best_model.pth was not created."
    print("Training loop completed and model saved.")

    # 6. Submission Generation
    print("\n[6] Generating Submission...")
    # We need to reload the best model or just use the current one (which is the best in this 1-epoch case)
    # The trainer logic saves to disk but keeps the model in memory.
    # For strict correctness based on the library, create_submission uses the trainer instance.
    create_submission(trainer, test_loader, output_path=Config.SUBMISSION_PATH)

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found."

    # Verify submission content
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission shape: {sub_df.shape}")

    # Expected rows: 50 (from debug=True in get_dataloaders for test set)
    # Expected cols: 1 (qa_id) + 30 (targets) = 31
    expected_rows = 50
    expected_cols = 31

    assert sub_df.shape == (
        expected_rows,
        expected_cols,
    ), f"Submission shape mismatch. Expected ({expected_rows}, {expected_cols}), got {sub_df.shape}"

    # Check values are within [0, 1] - though model outputs logits, Trainer.predict applies sigmoid
    preds_values = sub_df.iloc[:, 1:].values
    assert (
        preds_values.min() >= 0.0 and preds_values.max() <= 1.0
    ), "Predictions are not in probability range [0, 1]"

    print("Submission generation verified.")

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
