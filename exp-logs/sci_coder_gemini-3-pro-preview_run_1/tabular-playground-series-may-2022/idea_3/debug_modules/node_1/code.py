import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import seed_everything, save_submission
from library.dataset import Tokenizer, get_dataloaders
from library.model import HybridTransformerModel
from library.train import run_training


def main():
    print("=== Starting Demonstration and Verification Script ===")

    # ------------------------------------------------------------------------
    # 1. Configuration Setup for Fast Execution
    # ------------------------------------------------------------------------
    print("\n[1] Configuring environment for fast demonstration...")

    # Override Config defaults to run quickly on a small subset
    Config.DEBUG = True
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 64
    # Force data processing from scratch to verify the processing logic
    Config.LOAD_CACHED_DATA = False

    # Initialize directories
    Config.setup()

    # Set seeds for reproducibility
    seed_everything(42)
    print("    Configuration complete. Debug mode: ON, Epochs: 1.")

    # ------------------------------------------------------------------------
    # 2. Verify Utility Functions
    # ------------------------------------------------------------------------
    print("\n[2] Verifying Utility Functions...")

    # Test save_submission
    dummy_ids = [999000, 999001, 999002]
    dummy_preds = [0.1, 0.5, 0.9]
    dummy_output_path = os.path.join(Config.WORKING_DIR, "test_utils_submission.csv")

    save_submission(dummy_ids, dummy_preds, output_path=dummy_output_path)

    if not os.path.exists(dummy_output_path):
        raise AssertionError("save_submission failed to create file.")

    df_dummy = pd.read_csv(dummy_output_path)
    if df_dummy.shape != (3, 2):
        raise AssertionError(
            f"Dummy submission shape mismatch. Expected (3, 2), got {df_dummy.shape}"
        )
    if list(df_dummy.columns) != ["id", "target"]:
        raise AssertionError(
            f"Dummy submission columns mismatch. Got {list(df_dummy.columns)}"
        )

    print("    Utils verified successfully.")

    # ------------------------------------------------------------------------
    # 3. Verify Dataset and Tokenizer
    # ------------------------------------------------------------------------
    print("\n[3] Verifying Dataset and Tokenizer...")

    # Test Tokenizer logic
    tokenizer = Tokenizer()
    # 'A' -> 1, 'Z' -> 26, Padding -> 0
    test_strings = ["AB", "Z"]
    tokens = tokenizer.transform(test_strings)

    expected_shape = (2, Config.MAX_SEQ_LEN)
    if tokens.shape != expected_shape:
        raise AssertionError(
            f"Tokenizer output shape mismatch. Expected {expected_shape}, got {tokens.shape}"
        )

    # Check specific mappings
    # A=1, B=2
    if tokens[0, 0] != 1 or tokens[0, 1] != 2:
        raise AssertionError("Tokenizer mapping incorrect for 'AB'.")
    # Z=26
    if tokens[1, 0] != 26:
        raise AssertionError("Tokenizer mapping incorrect for 'Z'.")
    # Check padding (index 2 onwards should be 0 for "AB")
    if tokens[0, 2] != 0:
        raise AssertionError("Tokenizer padding incorrect.")

    print("    Tokenizer logic verified.")

    # Test DataLoaders
    print("    Initializing DataLoaders (this triggers data processing)...")
    train_loader, val_loader, test_loader = get_dataloaders(
        debug=True, batch_size=Config.BATCH_SIZE
    )

    # Fetch one batch to verify structure
    batch = next(iter(train_loader))
    seq_batch = batch["sequence"]
    num_batch = batch["numerical"]
    target_batch = batch["target"]

    print(
        f"    Batch shapes - Sequence: {seq_batch.shape}, Numerical: {num_batch.shape}, Target: {target_batch.shape}"
    )

    if seq_batch.shape != (Config.BATCH_SIZE, Config.MAX_SEQ_LEN):
        raise AssertionError("Sequence batch shape mismatch.")

    if num_batch.dim() != 2 or num_batch.shape[0] != Config.BATCH_SIZE:
        raise AssertionError("Numerical batch shape mismatch.")

    if target_batch.shape != (Config.BATCH_SIZE,):
        raise AssertionError("Target batch shape mismatch.")

    num_features = num_batch.shape[1]
    print(f"    DataLoaders verified. Detected {num_features} numerical features.")

    # ------------------------------------------------------------------------
    # 4. Verify Model Architecture
    # ------------------------------------------------------------------------
    print("\n[4] Verifying Model Architecture...")

    model = HybridTransformerModel(num_numerical_features=num_features)
    model.to(Config.DEVICE)
    model.train()

    # Move batch to device
    seq_batch = seq_batch.to(Config.DEVICE)
    num_batch = num_batch.to(Config.DEVICE)
    target_batch = target_batch.to(Config.DEVICE)

    # Forward pass
    logits = model(seq_batch, num_batch)

    if logits.shape != (Config.BATCH_SIZE, 1):
        raise AssertionError(
            f"Model output shape mismatch. Expected ({Config.BATCH_SIZE}, 1), got {logits.shape}"
        )

    # Backward pass check (Gradient flow)
    criterion = torch.nn.BCEWithLogitsLoss()
    loss = criterion(logits, target_batch.unsqueeze(1))
    loss.backward()

    # Check if gradients exist in the head layer
    if model.head.weight.grad is None:
        raise AssertionError("Gradients not calculated for model head.")

    print("    Model forward and backward pass verified.")

    # ------------------------------------------------------------------------
    # 5. Verify Full Training Pipeline
    # ------------------------------------------------------------------------
    print("\n[5] Verifying Full Training Pipeline...")

    # This runs the training loop, validation, and generates submission
    # We use debug=True and epochs=1 (set in Config)
    trained_model = run_training(debug=True)

    if trained_model is None:
        raise AssertionError("Training function returned None.")

    # Verify final submission file
    submission_path = Config.SUBMISSION_FILE
    if not os.path.exists(submission_path):
        raise AssertionError(f"Final submission file not found at {submission_path}")

    df_sub = pd.read_csv(submission_path)

    # In debug mode, the test set is clipped to DEBUG_SAMPLE_SIZE
    expected_len = min(100000, Config.DEBUG_SAMPLE_SIZE)  # 100000 is total test rows
    if len(df_sub) != expected_len:
        raise AssertionError(
            f"Submission file length mismatch. Expected {expected_len}, got {len(df_sub)}"
        )

    print(f"    Pipeline finished. Submission generated with {len(df_sub)} rows.")
    print("    Full pipeline verified.")

    print("\n=== All Tests Passed Successfully ===")


if __name__ == "__main__":
    main()
