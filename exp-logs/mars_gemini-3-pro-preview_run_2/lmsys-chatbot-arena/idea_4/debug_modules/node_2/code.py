import os
import sys
import torch
import pandas as pd
import numpy as np
import transformers
from transformers import AutoTokenizer

# Import library modules
from library.config import Config
from library.utils import seed_everything
from library.dataset import get_dataloaders
from library.model import SiameseDeberta
from library.engine import train_model, infer


def run_demo():
    # --- 1. Setup & Configuration ---
    print("\n--- 1. Setup & Configuration ---")

    # Suppress warnings for cleaner output
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    transformers.logging.set_verbosity_error()

    # Modify Config for a fast demonstration
    Config.DEBUG = True
    Config.DEBUG_SAMPLES = 50  # Small subset for speed
    Config.EPOCHS = 1
    Config.WORKING_DIR = "./working/demo_execution"
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "demo_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "demo_submission.csv")

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seeds
    seed_everything(Config.SEED)
    print(f"Configuration updated: DEBUG={Config.DEBUG}, EPOCHS={Config.EPOCHS}")
    print(f"Working directory: {Config.WORKING_DIR}")

    # --- 2. Data Pipeline Verification ---
    print("\n--- 2. Data Pipeline Verification ---")

    # Initialize Tokenizer
    print(f"Loading tokenizer: {Config.MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # Get DataLoaders (Force reload to apply DEBUG slicing)
    print("Generating DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        tokenizer, load_cached_data=False
    )

    # Verify Train Loader Batch
    print("Verifying training batch structure...")
    batch = next(iter(train_loader))

    # Check keys
    expected_keys = [
        "ids",
        "input_ids_a",
        "attention_mask_a",
        "input_ids_b",
        "attention_mask_b",
        "meta_features",
        "labels",
    ]
    for key in expected_keys:
        assert key in batch, f"Missing key in batch: {key}"

    # Check shapes
    batch_size = batch["input_ids_a"].size(0)
    seq_len = Config.MAX_LENGTH

    assert batch["input_ids_a"].shape == (
        batch_size,
        seq_len,
    ), "Incorrect input_ids_a shape"
    assert batch["input_ids_b"].shape == (
        batch_size,
        seq_len,
    ), "Incorrect input_ids_b shape"
    assert batch["meta_features"].shape == (
        batch_size,
        3,
    ), "Incorrect meta_features shape"
    assert batch["labels"].shape == (batch_size, 3), "Incorrect labels shape"

    print(f"Batch verification successful. Batch size: {batch_size}")

    # --- 3. Model Architecture Verification ---
    print("\n--- 3. Model Architecture Verification ---")

    device = Config.DEVICE
    print(f"Initializing model on {device}...")
    model = SiameseDeberta()
    model.to(device)

    # Move batch to device
    input_ids_a = batch["input_ids_a"].to(device)
    attention_mask_a = batch["attention_mask_a"].to(device)
    input_ids_b = batch["input_ids_b"].to(device)
    attention_mask_b = batch["attention_mask_b"].to(device)
    meta_features = batch["meta_features"].to(device)

    # Forward Pass
    print("Running forward pass...")
    with torch.no_grad():
        logits = model(
            input_ids_a, attention_mask_a, input_ids_b, attention_mask_b, meta_features
        )

    # Verify Output
    assert logits.shape == (
        batch_size,
        3,
    ), f"Expected logits shape ({batch_size}, 3), got {logits.shape}"
    print("Forward pass successful. Logits shape verified.")

    # --- 4. Training Loop Execution ---
    print("\n--- 4. Training Loop Execution ---")

    print("Starting training (1 epoch on debug subset)...")
    best_loss = train_model(train_loader, val_loader)

    assert os.path.exists(Config.MODEL_SAVE_PATH), "Model checkpoint was not created."
    print(f"Training complete. Best Log Loss: {best_loss:.4f}")
    print(f"Model saved to: {Config.MODEL_SAVE_PATH}")

    # --- 5. Inference and Submission ---
    print("\n--- 5. Inference and Submission ---")

    print("Running inference on test set...")
    ids, preds = infer(test_loader)

    # Verify Inference Shapes
    n_test_samples = len(test_loader.dataset)
    assert (
        len(ids) == n_test_samples
    ), f"ID count mismatch: {len(ids)} vs {n_test_samples}"
    assert preds.shape == (
        n_test_samples,
        3,
    ), f"Prediction shape mismatch: {preds.shape}"

    # Create Submission DataFrame
    print("Creating submission file...")
    submission_df = pd.DataFrame(
        {
            "id": ids.astype(int),
            "winner_model_a": preds[:, 0],
            "winner_model_b": preds[:, 1],
            "winner_tie": preds[:, 2],
        }
    )

    # Verify Probabilities sum to ~1
    sums = submission_df[["winner_model_a", "winner_model_b", "winner_tie"]].sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-5), "Probabilities do not sum to 1"

    # Save
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to: {Config.SUBMISSION_PATH}")

    # Display first few rows
    print("\nSample Submission:")
    print(submission_df.head())

    print("\n--- Demo Completed Successfully ---")


if __name__ == "__main__":
    run_demo()
