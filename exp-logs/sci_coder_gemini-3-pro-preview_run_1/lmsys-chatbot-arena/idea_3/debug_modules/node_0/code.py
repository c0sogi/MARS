import os
import sys
import torch
import pandas as pd
import numpy as np

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, ensure_directories
from library.features import FeatureEngineer
from library.dataset import get_dataloaders
from library.model import SiameseDebertaWithScalars
from library.train import run_training
from library.inference import predict


def main():
    # 1. Setup and Configuration Overrides for Speed
    print(">>> Setting up environment and configuration...")
    seed_everything(42)

    # Modify Config for a fast demonstration run
    # We reduce epochs and batch size to minimize runtime while testing logic
    Config.NUM_EPOCHS = 1
    Config.TRAIN_BATCH_SIZE = 4
    Config.VALID_BATCH_SIZE = 4

    # Ensure directories exist (Config.setup is called inside ensure_directories)
    ensure_directories()

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Model Save Path: {Config.MODEL_SAVE_PATH}")

    # 2. Verify Feature Engineering Logic
    print("\n>>> Verifying FeatureEngineer...")
    fe = FeatureEngineer()

    # Create a dummy dataframe to test feature calculation logic
    # Case 1: A="hi" (2 chars), B="hello" (5 chars) -> diff = -3
    # Case 2: A="hello world" (11 chars), B="hi" (2 chars) -> diff = 9
    dummy_df = pd.DataFrame(
        {"response_a": ["hi", "hello world"], "response_b": ["hello", "hi"]}
    )

    # Extract features
    features = fe.extract_features(dummy_df)

    # Check shape: (2 samples, 5 features defined in Config)
    if features.shape != (2, 5):
        raise AssertionError(
            f"Feature matrix shape mismatch. Expected (2, 5), got {features.shape}"
        )

    # Check specific feature values (char_len_diff is at index 0 in Config.SCALAR_FEATURE_LIST)
    # Row 0: 2 - 5 = -3
    if not np.isclose(features[0, 0], -3.0):
        raise AssertionError(
            f"Feature calculation error. Expected -3.0, got {features[0, 0]}"
        )

    # Row 1: 11 - 2 = 9
    if not np.isclose(features[1, 0], 9.0):
        raise AssertionError(
            f"Feature calculation error. Expected 9.0, got {features[1, 0]}"
        )

    print("FeatureEngineer logic verified successfully.")

    # 3. Verify Dataset and DataLoader
    print("\n>>> Verifying DataLoaders (Debug Mode)...")
    # debug=True loads only 50 rows from the metadata
    train_loader, val_loader, test_loader = get_dataloaders(debug=True)

    # Fetch one batch from the training loader
    try:
        batch = next(iter(train_loader))
    except StopIteration:
        raise AssertionError("Train loader is empty!")

    # Check for presence of required keys
    required_keys = [
        "input_ids_a",
        "attention_mask_a",
        "input_ids_b",
        "attention_mask_b",
        "scalar_features",
        "labels",
    ]
    for key in required_keys:
        if key not in batch:
            raise AssertionError(f"Missing key in batch: {key}")

    # Check tensor shapes
    batch_size = batch["input_ids_a"].shape[0]
    if batch_size != Config.TRAIN_BATCH_SIZE:
        raise AssertionError(
            f"Batch size mismatch. Expected {Config.TRAIN_BATCH_SIZE}, got {batch_size}"
        )

    if batch["scalar_features"].shape[1] != Config.NUM_SCALAR_FEATURES:
        raise AssertionError(
            f"Scalar feature dimension mismatch. Expected {Config.NUM_SCALAR_FEATURES}, got {batch['scalar_features'].shape[1]}"
        )

    print("DataLoaders verified successfully.")

    # 4. Verify Model Architecture
    print("\n>>> Verifying Model Architecture...")
    model = SiameseDebertaWithScalars()

    # Run a forward pass on CPU with the fetched batch to verify layer connections
    model.eval()
    with torch.no_grad():
        logits = model(
            input_ids_a=batch["input_ids_a"],
            attention_mask_a=batch["attention_mask_a"],
            input_ids_b=batch["input_ids_b"],
            attention_mask_b=batch["attention_mask_b"],
            scalar_features=batch["scalar_features"],
        )

    # Check output shape: (batch_size, 3 classes)
    if logits.shape != (batch_size, 3):
        raise AssertionError(
            f"Model output shape mismatch. Expected ({batch_size}, 3), got {logits.shape}"
        )

    print("Model architecture verified successfully.")

    # 5. Verify Training Loop
    print("\n>>> Verifying Training Loop (Debug Mode)...")
    # This will run for 1 epoch on 50 samples (approx 12 batches of size 4)
    # This function handles device placement internally
    run_training(debug=True)

    # Check if the model artifact was saved
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        raise AssertionError(f"Model file was not saved at {Config.MODEL_SAVE_PATH}")

    print("Training loop completed and model saved.")

    # 6. Verify Inference Loop
    print("\n>>> Verifying Inference Loop (Debug Mode)...")
    # This will generate predictions for the debug test set (50 samples)
    predict(debug=True)

    # Check if the submission file was saved
    if not os.path.exists(Config.SUBMISSION_SAVE_PATH):
        raise AssertionError(
            f"Submission file was not saved at {Config.SUBMISSION_SAVE_PATH}"
        )

    # Validate submission content and format
    submission = pd.read_csv(Config.SUBMISSION_SAVE_PATH)

    # Debug test set has 50 rows
    if len(submission) != 50:
        raise AssertionError(
            f"Submission row count mismatch. Expected 50, got {len(submission)}"
        )

    expected_cols = ["id", "winner_model_a", "winner_model_b", "winner_tie"]
    if not all(col in submission.columns for col in expected_cols):
        raise AssertionError(
            f"Submission columns mismatch. Expected {expected_cols}, got {submission.columns.tolist()}"
        )

    print("Inference loop completed and submission generated.")
    print("\n>>> All demonstrations and verifications passed successfully.")


if __name__ == "__main__":
    main()
