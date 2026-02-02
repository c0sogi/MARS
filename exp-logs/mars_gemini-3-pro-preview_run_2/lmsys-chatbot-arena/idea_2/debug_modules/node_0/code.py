import os
import torch
import pandas as pd
import numpy as np
import shutil

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, get_logger
from library.dataset import get_dataloaders
from library.model import SiameseDualEncoder
from library.engine import train, predict

# Initialize logger
logger = get_logger("demo")


def run_demo():
    print("Starting demonstration of the Siamese Dual-Encoder pipeline...")

    # 1. Setup and Configuration overrides for Speed
    # We override Config attributes to run a fast "micro" experiment.
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 50  # Use only 50 samples for demonstration
    Config.EPOCHS = 1  # Train for only 1 epoch
    Config.TRAIN_BATCH_SIZE = 4  # Small batch size
    Config.VALID_BATCH_SIZE = 4
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "demo_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "demo_submission.csv")

    # Ensure reproducibility
    seed_everything(Config.SEED)

    # Ensure working directory exists
    if os.path.exists(Config.WORKING_DIR):
        # Clean up previous demo runs to ensure assertions are valid
        if os.path.exists(Config.MODEL_SAVE_PATH):
            os.remove(Config.MODEL_SAVE_PATH)
        if os.path.exists(Config.SUBMISSION_PATH):
            os.remove(Config.SUBMISSION_PATH)
    else:
        os.makedirs(Config.WORKING_DIR)

    print("\n--- Step 1: Data Loading Verification ---")
    # Load dataloaders using the factory function
    train_loader, val_loader, test_loader = get_dataloaders(
        train_batch_size=Config.TRAIN_BATCH_SIZE,
        valid_batch_size=Config.VALID_BATCH_SIZE,
        debug=Config.DEBUG,
        load_cached_data=False,  # Force reload to ensure debug subsetting works on fresh data logic if needed
    )

    # Verify Train Loader
    batch = next(iter(train_loader))
    print(f"Train batch keys: {batch.keys()}")

    # Assertions for data structure
    assert "input_ids_a" in batch, "Missing input_ids_a in batch"
    assert "input_ids_b" in batch, "Missing input_ids_b in batch"
    assert "labels" in batch, "Missing labels in training batch"

    # Check shapes
    # Shape should be [Batch_Size, Max_Length]
    expected_shape = (Config.TRAIN_BATCH_SIZE, Config.MAX_LENGTH)
    assert (
        batch["input_ids_a"].shape == expected_shape
    ), f"Incorrect shape for input_ids_a: {batch['input_ids_a'].shape} vs {expected_shape}"
    assert batch["labels"].shape == (
        Config.TRAIN_BATCH_SIZE,
    ), f"Incorrect shape for labels: {batch['labels'].shape}"

    print("Data loading verified successfully.")

    print("\n--- Step 2: Model Architecture Verification ---")
    # Instantiate model
    device = Config.DEVICE
    model = SiameseDualEncoder(
        model_name=Config.MODEL_NAME, num_classes=Config.NUM_CLASSES
    ).to(device)

    # Move batch to device
    input_ids_a = batch["input_ids_a"].to(device)
    attention_mask_a = batch["attention_mask_a"].to(device)
    input_ids_b = batch["input_ids_b"].to(device)
    attention_mask_b = batch["attention_mask_b"].to(device)

    # Perform a dummy forward pass
    model.eval()
    with torch.no_grad():
        logits = model(input_ids_a, attention_mask_a, input_ids_b, attention_mask_b)

    print(f"Logits shape: {logits.shape}")

    # Assert output shape [Batch_Size, Num_Classes]
    assert logits.shape == (
        Config.TRAIN_BATCH_SIZE,
        Config.NUM_CLASSES,
    ), f"Model output shape mismatch. Expected {(Config.TRAIN_BATCH_SIZE, Config.NUM_CLASSES)}, got {logits.shape}"

    print("Model architecture verified successfully.")

    print("\n--- Step 3: Training Loop Demonstration ---")
    # Run the training engine
    # We pass the overridden parameters explicitly to be safe
    trained_model = train(
        epochs=Config.EPOCHS,
        learning_rate=Config.LEARNING_RATE,
        train_batch_size=Config.TRAIN_BATCH_SIZE,
        valid_batch_size=Config.VALID_BATCH_SIZE,
        debug=Config.DEBUG,
        save_path=Config.MODEL_SAVE_PATH,
    )

    # Verify that the model checkpoint was created
    assert os.path.exists(
        Config.MODEL_SAVE_PATH
    ), f"Model checkpoint not found at {Config.MODEL_SAVE_PATH}"
    print(f"Training complete. Model saved to {Config.MODEL_SAVE_PATH}")

    print("\n--- Step 4: Inference Demonstration ---")
    # Run the prediction engine
    predict(
        test_batch_size=Config.VALID_BATCH_SIZE,
        debug=Config.DEBUG,
        model_path=Config.MODEL_SAVE_PATH,
        submission_path=Config.SUBMISSION_PATH,
    )

    # Verify submission file
    assert os.path.exists(
        Config.SUBMISSION_PATH
    ), f"Submission file not found at {Config.SUBMISSION_PATH}"

    # Verify submission content
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission shape: {df_sub.shape}")
    print(f"Submission columns: {df_sub.columns.tolist()}")

    expected_cols = ["id", "winner_model_a", "winner_model_b", "winner_tie"]
    assert all(
        col in df_sub.columns for col in expected_cols
    ), "Submission missing required columns"

    # In debug mode, we expect rows equal to Config.DEBUG_SUBSET_SIZE
    assert (
        len(df_sub) == Config.DEBUG_SUBSET_SIZE
    ), f"Submission row count mismatch. Expected {Config.DEBUG_SUBSET_SIZE}, got {len(df_sub)}"

    # Verify probabilities sum to roughly 1
    row_sums = df_sub[["winner_model_a", "winner_model_b", "winner_tie"]].sum(axis=1)
    # Allow small floating point error
    assert np.allclose(row_sums, 1.0, atol=1e-5), "Probabilities do not sum to 1.0"

    print("Inference verified successfully.")
    print("\nDemonstration complete. All systems functional.")


if __name__ == "__main__":
    run_demo()
