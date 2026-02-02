import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import warnings

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, compute_log_loss
from library.data import get_dataloaders
from library.model import LongformerHybridClassifier
from library.train import Trainer


def run_demo():
    print("============================================================")
    print("   Chatbot Arena Prediction: Library Usage Demonstration    ")
    print("============================================================")

    # ------------------------------------------------------------------
    # 1. Configuration Setup for Rapid Execution
    # ------------------------------------------------------------------
    print("\n[1] Configuring environment for rapid demonstration...")

    # Override Config defaults to run a tiny, fast experiment
    Config.DEBUG = True
    Config.DEBUG_SIZE = 100  # Use only 100 rows for speed
    Config.EPOCHS = 1  # Run only 1 epoch
    Config.TRAIN_BATCH_SIZE = 4
    Config.VALID_BATCH_SIZE = 4
    Config.GRADIENT_ACCUMULATION_STEPS = 1
    Config.MAX_LENGTH = 128  # Reduce sequence length significantly for demo speed

    # Set a specific working directory for this demo to avoid conflicts
    Config.WORKING_DIR = "./working/demo_run"
    Config.CACHE_DIR = Config.WORKING_DIR
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Clean up previous demo runs if they exist
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    print(f"    Working Directory: {Config.WORKING_DIR}")
    print(f"    Debug Mode: {Config.DEBUG}")
    print(f"    Max Sequence Length: {Config.MAX_LENGTH}")

    # ------------------------------------------------------------------
    # 2. Utility Functions Verification
    # ------------------------------------------------------------------
    print("\n[2] Verifying Utility Functions...")

    # Seed everything for reproducibility
    seed_everything(Config.SEED)
    print("    Random seed set.")

    # Test Log Loss with dummy data
    # y_true: One-hot encoded (or probabilities)
    y_true_dummy = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    # y_pred: Predicted probabilities
    y_pred_dummy = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]])

    loss = compute_log_loss(y_true_dummy, y_pred_dummy)
    print(f"    Computed Log Loss (Dummy): {loss:.4f}")

    # Assertion to ensure logic is sound
    assert loss < 1.0, "Log loss calculation yielded unexpectedly high value."
    print("    Log Loss function verified.")

    # ------------------------------------------------------------------
    # 3. Data Pipeline Verification
    # ------------------------------------------------------------------
    print("\n[3] Verifying Data Pipeline...")

    # Load dataloaders (this will trigger caching in the new working dir)
    # We set load_cached_data=False to force processing of the debug subset
    print("    Initializing Dataloaders...")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    print(f"    Train Batches: {len(train_loader)}")
    print(f"    Val Batches: {len(val_loader)}")

    # Fetch a single batch to inspect structure
    batch = next(iter(train_loader))
    print("    Batch Keys:", list(batch.keys()))

    # Validate Batch Structure
    required_keys = [
        "input_ids",
        "attention_mask",
        "global_attention_mask",
        "scalar_features",
        "labels",
        "id",
    ]
    for key in required_keys:
        assert key in batch, f"Missing key {key} in batch."

    # Validate Shapes
    # input_ids: (batch_size, seq_len)
    assert batch["input_ids"].shape[0] == Config.TRAIN_BATCH_SIZE
    assert batch["input_ids"].shape[1] <= Config.MAX_LENGTH
    # scalar_features: (batch_size, 3)
    assert batch["scalar_features"].shape == (Config.TRAIN_BATCH_SIZE, 3)
    # labels: (batch_size, 3)
    assert batch["labels"].shape == (Config.TRAIN_BATCH_SIZE, 3)

    print("    Data shapes and content verified.")

    # ------------------------------------------------------------------
    # 4. Model Architecture Verification
    # ------------------------------------------------------------------
    print("\n[4] Verifying Model Architecture...")

    model = LongformerHybridClassifier()
    model.to(Config.DEVICE)
    model.eval()

    print("    Model instantiated successfully.")

    # Run a forward pass with the batch fetched earlier
    with torch.no_grad():
        input_ids = batch["input_ids"].to(Config.DEVICE)
        attention_mask = batch["attention_mask"].to(Config.DEVICE)
        global_attention_mask = batch["global_attention_mask"].to(Config.DEVICE)
        scalar_features = batch["scalar_features"].to(Config.DEVICE)

        # Use mixed precision as per config (though not strictly necessary for shape check)
        with torch.amp.autocast("cuda", enabled=Config.USE_FP16):
            logits = model(
                input_ids, attention_mask, global_attention_mask, scalar_features
            )

    print(f"    Output Logits Shape: {logits.shape}")

    # Assert output shape matches (batch_size, num_classes)
    assert logits.shape == (
        Config.TRAIN_BATCH_SIZE,
        3,
    ), f"Expected shape ({Config.TRAIN_BATCH_SIZE}, 3), got {logits.shape}"
    print("    Forward pass verified.")

    # ------------------------------------------------------------------
    # 5. Training Loop Verification
    # ------------------------------------------------------------------
    print("\n[5] Verifying Training Loop (Trainer)...")

    # Initialize Trainer
    # Note: Trainer re-initializes dataloaders internally.
    # Since we updated Config and cache exists from step 3, this will be fast.
    trainer = Trainer()

    # Run Fit (1 epoch, debug size)
    print("    Starting training (Fit)...")
    trainer.fit()

    # Check if model artifact exists
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        raise FileNotFoundError(
            f"Model file not found at {Config.MODEL_SAVE_PATH} after training."
        )

    print("    Training completed. Model saved.")

    # ------------------------------------------------------------------
    # 6. Submission Generation Verification
    # ------------------------------------------------------------------
    print("\n[6] Verifying Submission Generation...")

    trainer.generate_submission()

    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}."
        )

    # Load submission and check validity
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"    Submission File Loaded. Shape: {sub_df.shape}")
    print(f"    Columns: {list(sub_df.columns)}")

    # Check rows count (should match test debug size)
    # DEBUG mode limits all splits, including test
    assert (
        len(sub_df) == Config.DEBUG_SIZE
    ), f"Expected {Config.DEBUG_SIZE} rows in submission, found {len(sub_df)}"

    # Check probability sum (approximate sum to 1.0)
    row_sums = sub_df[["winner_model_a", "winner_model_b", "winner_tie"]].sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-5), "Probabilities do not sum to 1."

    print("    Submission format verified.")

    print("\n============================================================")
    print("   Demonstration Completed Successfully")
    print("============================================================")


if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")
    run_demo()
