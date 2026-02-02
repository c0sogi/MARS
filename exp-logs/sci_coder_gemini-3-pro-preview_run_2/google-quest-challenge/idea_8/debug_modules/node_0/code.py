import os
import torch
import numpy as np
import pandas as pd
import shutil
from library.config import Config
from library.utils import compute_spearmanr, seed_everything
from library.data import get_dataloaders
from library.model import SiameseCoAttentionNetwork
from library.engine import run_training, generate_submission


def main():
    print("=== Starting Demonstration of QA Labeling Pipeline ===")

    # ---------------------------------------------------------
    # 1. Configuration Override for Speed and Isolation
    # ---------------------------------------------------------
    print("\n[1] Configuring environment for demo run...")

    # Override Config attributes to run a fast, lightweight demo
    Config.debug = True
    Config.debug_sample_size = 50  # Small subset for speed
    Config.epochs = 1
    Config.train_batch_size = 4
    Config.valid_batch_size = 4
    Config.working_dir = "./working/demo_run"
    Config.output_dir = "./working/demo_run"
    Config.submission_path = os.path.join(Config.working_dir, "submission.csv")

    # Clean up demo directory if it exists to ensure a fresh run
    if os.path.exists(Config.working_dir):
        shutil.rmtree(Config.working_dir)
    os.makedirs(Config.working_dir, exist_ok=True)

    print(f"Debug Mode: {Config.debug}")
    print(f"Working Directory: {Config.working_dir}")
    print(f"Device: {Config.device}")

    # ---------------------------------------------------------
    # 2. Verify Utility Functions
    # ---------------------------------------------------------
    print("\n[2] Verifying Utility Functions...")

    # Test Spearman Correlation
    y_true = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    y_pred = np.array([[0.15, 0.25], [0.25, 0.35], [0.55, 0.65]])
    # Perfectly correlated ranks should yield 1.0
    score = compute_spearmanr(y_true, y_pred)
    print(f"Computed Spearman Score on dummy data: {score}")

    if abs(score - 1.0) > 1e-6:
        raise AssertionError(
            "compute_spearmanr failed validation: Expected ~1.0 for correlated data."
        )

    # ---------------------------------------------------------
    # 3. Verify Data Pipeline
    # ---------------------------------------------------------
    print("\n[3] Verifying Data Pipeline...")

    # Load dataloaders (this triggers tokenization and dataset creation)
    train_loader, val_loader, test_loader = get_dataloaders(
        debug=Config.debug, load_cached_data=False
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Fetch one batch to inspect structure
    batch = next(iter(train_loader))
    required_keys = [
        "q_input_ids",
        "q_attention_mask",
        "q_token_type_ids",
        "a_input_ids",
        "a_attention_mask",
        "cats",
        "labels",
    ]

    for key in required_keys:
        if key not in batch:
            raise KeyError(f"Batch missing required key: {key}")

    # Check shapes
    batch_size = batch["q_input_ids"].shape[0]
    if batch_size != Config.train_batch_size:
        raise AssertionError(
            f"Batch size mismatch. Expected {Config.train_batch_size}, got {batch_size}"
        )

    if batch["q_input_ids"].shape[1] != Config.max_len_q:
        raise AssertionError(
            f"Question seq length mismatch. Expected {Config.max_len_q}, got {batch['q_input_ids'].shape[1]}"
        )

    if batch["a_input_ids"].shape[1] != Config.max_len_a:
        raise AssertionError(
            f"Answer seq length mismatch. Expected {Config.max_len_a}, got {batch['a_input_ids'].shape[1]}"
        )

    if batch["labels"].shape[1] != Config.num_targets:
        raise AssertionError(
            f"Target mismatch. Expected {Config.num_targets}, got {batch['labels'].shape[1]}"
        )

    print("Data batch structure verified successfully.")

    # ---------------------------------------------------------
    # 4. Verify Model Architecture
    # ---------------------------------------------------------
    print("\n[4] Verifying Model Architecture...")

    model = SiameseCoAttentionNetwork()
    model.to(Config.device)

    # Move batch to device
    inputs = {
        "q_input_ids": batch["q_input_ids"].to(Config.device),
        "q_attention_mask": batch["q_attention_mask"].to(Config.device),
        "q_token_type_ids": batch["q_token_type_ids"].to(Config.device),
        "a_input_ids": batch["a_input_ids"].to(Config.device),
        "a_attention_mask": batch["a_attention_mask"].to(Config.device),
        "cats": batch["cats"].to(Config.device),
    }

    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)

    print(f"Model output shape: {outputs.shape}")

    # Check output shape
    if outputs.shape != (batch_size, Config.num_targets):
        raise AssertionError(
            f"Model output shape mismatch. Expected {(batch_size, Config.num_targets)}, got {outputs.shape}"
        )

    # Check output range (Sigmoid)
    if outputs.min() < 0.0 or outputs.max() > 1.0:
        raise AssertionError(
            "Model outputs out of range [0, 1]. Sigmoid activation might be missing."
        )

    print("Model forward pass verified successfully.")

    # ---------------------------------------------------------
    # 5. Verify Training Engine
    # ---------------------------------------------------------
    print("\n[5] Verifying Training Loop...")

    # Run training for 1 epoch (as configured)
    best_score = run_training(train_loader, val_loader)

    print(f"Training complete. Best Validation Score: {best_score:.4f}")

    # Verify checkpoint existence
    best_model_path = os.path.join(Config.output_dir, "best_model.pth")
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {best_model_path}")

    print("Training loop and checkpointing verified successfully.")

    # ---------------------------------------------------------
    # 6. Verify Inference and Submission
    # ---------------------------------------------------------
    print("\n[6] Verifying Inference and Submission Generation...")

    generate_submission(test_loader)

    if not os.path.exists(Config.submission_path):
        raise FileNotFoundError(
            f"Submission file not found at {Config.submission_path}"
        )

    # Validate submission file content
    sub_df = pd.read_csv(Config.submission_path)
    print(f"Submission shape: {sub_df.shape}")

    # Check columns
    expected_cols = ["qa_id"] + Config.target_cols
    if list(sub_df.columns) != expected_cols:
        raise AssertionError("Submission columns do not match expected target columns.")

    # Check row count (should match debug sample size for test set)
    # Note: In debug mode, test set is also sliced to debug_sample_size
    if len(sub_df) != Config.debug_sample_size:
        raise AssertionError(
            f"Submission row count mismatch. Expected {Config.debug_sample_size}, got {len(sub_df)}"
        )

    # Check value range
    numeric_cols = sub_df.columns[1:]
    if (sub_df[numeric_cols].values < 0).any() or (
        sub_df[numeric_cols].values > 1
    ).any():
        raise AssertionError("Submission contains values outside range [0, 1].")

    print("Inference and submission verified successfully.")

    print("\n=== Demonstration Complete: All Systems Go ===")


if __name__ == "__main__":
    main()
