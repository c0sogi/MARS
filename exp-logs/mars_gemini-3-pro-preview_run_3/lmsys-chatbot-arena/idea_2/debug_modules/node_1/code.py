import os
import sys
import torch
import numpy as np
import pandas as pd
import shutil

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, load_checkpoint
from library.data import RegexTokenizer, prepare_data
from library.model import ESIMHybridModel
from library.engine import run_training, generate_submission


def run_demo():
    print("Starting Library Usage Demonstration...")

    # 1. Setup and Configuration Overrides for Speed
    # We modify the Config class attributes directly to run a fast debug session.
    print("\n[1] Configuring environment for fast execution...")
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 50  # Use very small subset for demo
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.VOCAB_SIZE = 1000  # Smaller vocab for demo
    Config.WORKING_DIR = "./working/demo_run"  # Separate dir for this demo
    Config.SUBMISSION_DIR = "./submission"
    Config.SUBMISSION_FILE = os.path.join(Config.SUBMISSION_DIR, "demo_submission.csv")

    # Setup directories
    Config.setup()
    seed_everything(Config.SEED)
    print("Configuration updated. Debug mode enabled.")

    # 2. Tokenizer Verification
    print("\n[2] Verifying RegexTokenizer...")
    tokenizer = RegexTokenizer(vocab_size=100, max_len=10)
    dummy_texts = ["hello world", "hello chatbot", "chatbot arena"]
    tokenizer.fit(dummy_texts)

    # Check vocabulary
    assert "hello" in tokenizer.vocab, "Tokenizer failed to learn 'hello'"
    assert "chatbot" in tokenizer.vocab, "Tokenizer failed to learn 'chatbot'"

    # Check encoding
    ids, original_len = tokenizer.encode("hello world")
    assert len(ids) == 10, f"Expected padded length 10, got {len(ids)}"
    assert ids[0] == tokenizer.vocab["hello"], "Token ID mismatch for 'hello'"
    print("Tokenizer verification passed.")

    # 3. Data Pipeline Verification
    print("\n[3] Verifying Data Pipeline (prepare_data)...")
    # Force reload to process the debug subset
    train_loader, val_loader, test_loader, tokenizer = prepare_data(
        load_cached_data=False, batch_size=Config.BATCH_SIZE, debug=Config.DEBUG
    )

    # Fetch one batch to verify structure
    batch = next(iter(train_loader))

    # Verify Batch Keys
    required_keys = ["prompt_ids", "res_a_ids", "res_b_ids", "scalars", "target"]
    for key in required_keys:
        assert key in batch, f"Batch missing key: {key}"

    # Verify Shapes
    # prompt_ids: [Batch, Seq_Len]
    assert (
        batch["prompt_ids"].shape[0] == Config.BATCH_SIZE
    ), "Incorrect batch size in prompt_ids"
    # scalars: [Batch, 3]
    assert batch["scalars"].shape == (
        Config.BATCH_SIZE,
        3,
    ), f"Incorrect scalars shape: {batch['scalars'].shape}"
    # target: [Batch, 3] (3 classes)
    assert batch["target"].shape == (
        Config.BATCH_SIZE,
        3,
    ), f"Incorrect target shape: {batch['target'].shape}"

    print(f"Data Loader verification passed. Batch size: {Config.BATCH_SIZE}")

    # 4. Model Architecture Verification
    print("\n[4] Verifying ESIMHybridModel...")
    device = Config.DEVICE
    model = ESIMHybridModel().to(device)

    # Move batch to device
    p_ids = batch["prompt_ids"].to(device)
    a_ids = batch["res_a_ids"].to(device)
    b_ids = batch["res_b_ids"].to(device)
    sc = batch["scalars"].to(device)

    # Forward pass
    logits = model(p_ids, a_ids, b_ids, sc)

    # Check Output
    assert logits.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), f"Model output shape mismatch. Expected {(Config.BATCH_SIZE, Config.NUM_CLASSES)}, got {logits.shape}"

    # Check for NaNs
    assert not torch.isnan(logits).any(), "Model output contains NaNs"
    print("Model forward pass verification passed.")

    # 5. Training Loop Verification
    print("\n[5] Verifying Training Loop (run_training)...")
    # This runs the full training loop for 1 epoch on the debug subset
    best_model_path = run_training(
        epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE, debug=Config.DEBUG
    )

    assert os.path.exists(
        best_model_path
    ), f"Best model file not found at {best_model_path}"
    print(f"Training loop completed. Model saved to {best_model_path}")

    # 6. Inference Verification
    print("\n[6] Verifying Inference (generate_submission)...")
    generate_submission(best_model_path)

    assert os.path.exists(Config.SUBMISSION_FILE), "Submission file was not created"

    # Verify Submission Content
    sub_df = pd.read_csv(Config.SUBMISSION_FILE)

    # Check columns
    expected_cols = ["id", "winner_model_a", "winner_model_b", "winner_tie"]
    assert (
        list(sub_df.columns) == expected_cols
    ), f"Submission columns mismatch. Got {sub_df.columns}"

    # Check length (Should match debug subset size)
    assert (
        len(sub_df) == Config.DEBUG_SUBSET_SIZE
    ), f"Submission length mismatch. Expected {Config.DEBUG_SUBSET_SIZE}, got {len(sub_df)}"

    # Check probability constraints (rows should roughly sum to 1)
    # Allow small float error
    row_sums = sub_df[["winner_model_a", "winner_model_b", "winner_tie"]].sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-5), "Probabilities do not sum to 1"

    print("Inference verification passed.")
    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    # Ensure clean state for the demo directory if it exists
    if os.path.exists("./working/demo_run"):
        shutil.rmtree("./working/demo_run")

    try:
        run_demo()
    except Exception as e:
        print(f"\nFAILED: An error occurred during the demonstration: {e}")
        raise e
