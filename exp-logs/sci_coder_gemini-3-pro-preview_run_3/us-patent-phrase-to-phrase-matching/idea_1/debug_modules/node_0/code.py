import sys
import os
import torch
import pandas as pd
import warnings
import shutil

# Ensure we can import from the library directory
sys.path.append(".")

from library.config import Config, set_seed
from library.data_loader import get_dataloaders
from library.model import SiameseDAN
from library.trainer import Trainer


def run_demonstration():
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    print("=== Starting Phrase Matching Task Demonstration ===")

    # ---------------------------------------------------------
    # 1. Configuration Setup for Fast Execution
    # ---------------------------------------------------------
    print("\n[1] Configuring environment...")

    # Modify Config for a quick debug run
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 200  # Use only 200 samples for speed
    Config.EPOCHS = 2  # Run only 2 epochs
    Config.BATCH_SIZE = 16  # Small batch size
    Config.PATIENCE = 2  # Early stopping patience

    # Set custom working directories for this demo to avoid conflicts
    Config.WORKING_DIR = "./working/demo_run/"
    Config.SUBMISSION_DIR = "./submission/"
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "model.pth")

    # Initialize directories
    Config.setup()

    # Set seed for reproducibility
    set_seed(Config.SEED)

    print(f"    Debug Mode: {Config.DEBUG}")
    print(f"    Device: {Config.DEVICE}")

    # ---------------------------------------------------------
    # 2. Data Loading Demonstration
    # ---------------------------------------------------------
    print("\n[2] Loading Data...")

    # We force load_cached_data=False initially to demonstrate vocab building
    train_loader, val_loader, test_loader, vocab_size, num_contexts = get_dataloaders(
        load_cached_data=False
    )

    # Logic Verification: Check Loader Integrity
    assert len(train_loader) > 0, "Train loader should not be empty."
    assert len(val_loader) > 0, "Validation loader should not be empty."
    assert len(test_loader) > 0, "Test loader should not be empty."
    assert (
        vocab_size > 2
    ), "Vocabulary should contain more than just PAD and UNK tokens."

    # Logic Verification: Check Batch Structure
    batch = next(iter(train_loader))
    required_keys = ["anchor", "target", "context", "id", "score"]
    for key in required_keys:
        assert key in batch, f"Batch is missing required key: {key}"

    print(f"    Vocab Size: {vocab_size}")
    print(f"    Contexts: {num_contexts}")
    print(f"    Batch Shape (Anchor): {batch['anchor'].shape}")

    # ---------------------------------------------------------
    # 3. Model Instantiation & Forward Pass Verification
    # ---------------------------------------------------------
    print("\n[3] Verifying Model Architecture...")

    device = torch.device(Config.DEVICE)

    # Instantiate Model
    model = SiameseDAN(
        vocab_size=vocab_size,
        num_contexts=num_contexts,
        embedding_dim=Config.EMBEDDING_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        context_dim=Config.CONTEXT_EMBEDDING_DIM,
        dropout_rate=Config.DROPOUT,
    ).to(device)

    # Prepare inputs
    anchor = batch["anchor"].to(device)
    target = batch["target"].to(device)
    context = batch["context"].to(device)

    # Perform Forward Pass
    with torch.no_grad():
        output = model(anchor, target, context)

    # Logic Verification: Output Shape and Values
    expected_shape = (anchor.size(0),)
    assert (
        output.shape == expected_shape
    ), f"Expected output shape {expected_shape}, got {output.shape}"
    assert not torch.isnan(output).any(), "Model output contains NaNs."

    print("    Model forward pass successful.")

    # ---------------------------------------------------------
    # 4. Full Training Pipeline (Trainer)
    # ---------------------------------------------------------
    print("\n[4] Running Trainer (Train -> Eval -> Predict)...")

    trainer = Trainer()

    # Run training
    # Note: load_cached_data=True will use the vocab/context maps generated in step 2
    trainer.train(epochs=Config.EPOCHS, debug=Config.DEBUG, load_cached_data=True)

    # ---------------------------------------------------------
    # 5. Artifact Verification
    # ---------------------------------------------------------
    print("\n[5] Verifying Outputs...")

    # Check Model File
    if os.path.exists(Config.MODEL_SAVE_PATH):
        print(f"    Model file found at: {Config.MODEL_SAVE_PATH}")
    else:
        raise FileNotFoundError(f"Model file not found at {Config.MODEL_SAVE_PATH}")

    # Check Submission File
    if os.path.exists(Config.SUBMISSION_PATH):
        print(f"    Submission file found at: {Config.SUBMISSION_PATH}")
    else:
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    # Verify Submission Content
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)

    # Check Columns
    assert "id" in df_sub.columns, "Submission missing 'id' column."
    assert "score" in df_sub.columns, "Submission missing 'score' column."

    # Check Rows
    # In debug mode, we limit the input dataframe size.
    # The test loader will also be limited by DEBUG_SAMPLE_SIZE in the library logic.
    expected_len = min(Config.DEBUG_SAMPLE_SIZE, 3648)  # 3648 is full test size
    assert (
        len(df_sub) == expected_len
    ), f"Expected {expected_len} predictions, got {len(df_sub)}"

    # Check Score Range
    min_score = df_sub["score"].min()
    max_score = df_sub["score"].max()
    assert min_score >= 0.0, f"Found scores < 0: {min_score}"
    assert max_score <= 1.0, f"Found scores > 1: {max_score}"

    print("    Submission format verified successfully.")
    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    run_demonstration()
