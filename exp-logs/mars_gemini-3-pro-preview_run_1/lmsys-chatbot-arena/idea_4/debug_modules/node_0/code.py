import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Add current directory to sys.path to ensure library imports work
sys.path.append(".")

# Import from the provided library files
from library.config import Config
from library.data import get_dataloaders
from library.model import SiameseDeberta
from library.train import run_training
from library.utils import seed_everything


def main():
    print("==================================================")
    print("   Chatbot Preference Prediction - Demo Script    ")
    print("==================================================")

    # --------------------------------------------------------------------------
    # 1. Configuration Setup for Fast Execution
    # --------------------------------------------------------------------------
    print("\n[1] Configuring environment for fast demonstration...")

    # Override Config values to run a very small, fast experiment
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 20  # Use only 20 samples for speed
    Config.BATCH_SIZE = 2  # Small batch size
    Config.EPOCHS = 1  # Only 1 epoch
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple script execution
    Config.FEAT_VERSION = "demo"  # Use a unique cache version for this run

    # Ensure the submission directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    seed_everything(Config.SEED)
    print("    Configuration updated: Debug Mode=True, Samples=20, Epochs=1")

    # --------------------------------------------------------------------------
    # 2. Data Loading & Verification
    # --------------------------------------------------------------------------
    print("\n[2] Demonstrating Data Loading...")

    # Load dataloaders in debug mode
    # We disable loading cached data to force feature computation logic to run
    train_loader, val_loader, test_loader = get_dataloaders(
        debug=True,
        load_cached_data=False,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
    )

    print(f"    Train Batches: {len(train_loader)}")
    print(f"    Test Batches:  {len(test_loader)}")

    # Verify a single batch
    batch = next(iter(train_loader))
    print(f"    Batch Keys: {list(batch.keys())}")

    # Assertions to verify data shapes
    expected_seq_len = Config.MAX_LEN
    input_ids_shape = batch["input_ids_a"].shape
    features_shape = batch["features"].shape
    target_shape = batch["target"].shape

    assert input_ids_shape == (
        Config.BATCH_SIZE,
        expected_seq_len,
    ), f"Input IDs shape mismatch: {input_ids_shape}"
    assert features_shape == (
        Config.BATCH_SIZE,
        8,
    ), f"Scalar features shape mismatch: {features_shape}"
    assert target_shape == (
        Config.BATCH_SIZE,
        3,
    ), f"Target shape mismatch: {target_shape}"

    print("    Data shapes verified successfully.")

    # --------------------------------------------------------------------------
    # 3. Model Instantiation & Forward Pass Verification
    # --------------------------------------------------------------------------
    print("\n[3] Demonstrating Model Architecture...")

    device = Config.DEVICE
    model = SiameseDeberta()
    model.to(device)
    model.eval()

    # Prepare inputs for the model
    inputs = {
        "input_ids_a": batch["input_ids_a"].to(device),
        "attention_mask_a": batch["attention_mask_a"].to(device),
        "input_ids_b": batch["input_ids_b"].to(device),
        "attention_mask_b": batch["attention_mask_b"].to(device),
        "features": batch["features"].to(device),
    }

    # Run forward pass
    with torch.no_grad():
        logits = model(**inputs)

    print(f"    Logits Shape: {logits.shape}")

    # Verify output shape
    assert logits.shape == (
        Config.BATCH_SIZE,
        3,
    ), f"Model output shape mismatch: {logits.shape}"

    print("    Forward pass verified successfully.")

    # --------------------------------------------------------------------------
    # 4. Training Loop Execution
    # --------------------------------------------------------------------------
    print("\n[4] Executing Training Loop (via library.train)...")

    # run_training handles the loop, optimization, and saving the best model
    run_training(debug=True, load_cached_data=False)

    # Verify the model was saved
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        raise FileNotFoundError(f"Model file not found at {Config.MODEL_SAVE_PATH}")

    print(f"    Training complete. Best model saved to {Config.MODEL_SAVE_PATH}")

    # --------------------------------------------------------------------------
    # 5. Inference & Submission Generation
    # --------------------------------------------------------------------------
    print("\n[5] Running Inference on Test Set...")

    # Load the best saved model
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.to(device)
    model.eval()

    all_probs = []

    with torch.no_grad():
        for batch in test_loader:
            inputs = {
                "input_ids_a": batch["input_ids_a"].to(device),
                "attention_mask_a": batch["attention_mask_a"].to(device),
                "input_ids_b": batch["input_ids_b"].to(device),
                "attention_mask_b": batch["attention_mask_b"].to(device),
                "features": batch["features"].to(device),
            }

            logits = model(**inputs)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)

    # Prepare Submission DataFrame
    # We need to get the IDs corresponding to the test set.
    # Since we used debug=True, the test set is the first N rows of test_metadata.csv
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)
    test_df_subset = test_df.head(Config.DEBUG_SAMPLE_SIZE).reset_index(drop=True)

    # Verify alignment
    assert len(all_probs) == len(
        test_df_subset
    ), f"Prediction count ({len(all_probs)}) matches ID count ({len(test_df_subset)})"

    submission = pd.DataFrame(
        {
            "id": test_df_subset["id"],
            "winner_model_a": all_probs[:, 0],
            "winner_model_b": all_probs[:, 1],
            "winner_tie": all_probs[:, 2],
        }
    )

    # Save submission
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"    Submission saved to {Config.SUBMISSION_PATH}")

    # --------------------------------------------------------------------------
    # 6. Final Validation of Submission File
    # --------------------------------------------------------------------------
    print("\n[6] Validating Submission File...")

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)

    # Check 1: Column Names
    required_cols = {"id", "winner_model_a", "winner_model_b", "winner_tie"}
    assert required_cols.issubset(
        df_sub.columns
    ), f"Missing columns. Found: {df_sub.columns}"

    # Check 2: Row Count
    assert (
        len(df_sub) == Config.DEBUG_SAMPLE_SIZE
    ), f"Row count mismatch. Expected {Config.DEBUG_SAMPLE_SIZE}, got {len(df_sub)}"

    # Check 3: Probability Sum
    # Sum the probability columns and ensure they equal ~1.0
    prob_cols = ["winner_model_a", "winner_model_b", "winner_tie"]
    row_sums = df_sub[prob_cols].sum(axis=1)

    # Allow for small floating point errors
    assert np.allclose(row_sums, 1.0, atol=1e-5), "Probabilities do not sum to 1.0"

    print("    Validation Successful! The script has executed correctly.")
    print("==================================================")


if __name__ == "__main__":
    main()
