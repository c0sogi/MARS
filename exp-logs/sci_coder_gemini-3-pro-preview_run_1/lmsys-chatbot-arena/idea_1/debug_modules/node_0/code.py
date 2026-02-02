import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, get_device
from library.features import FeaturePipeline
from library.dataset import ArenaDataset
from library.model import ClassifierMLP
from library.trainer import run_training_task, ModelTrainer


def main():
    print("Starting Chatbot Arena Prediction Task Demonstration...")

    # 1. Setup and Configuration Override for Speed
    # We modify the Config class attributes to ensure the demo runs quickly.
    print("\n[1] Configuring environment for rapid demonstration...")
    seed_everything(Config.SEED)

    # Override Config for speed
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 8
    Config.HIDDEN_DIM = 64  # Reduce model size for demo
    Config.EARLY_STOPPING_PATIENCE = 1

    # Ensure working directories exist (redundant as Config does it, but good for safety)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    print(f"Device: {get_device()}")
    print("Configuration updated for speed (Epochs=2, Batch=8, Hidden=64).")

    # 2. Feature Pipeline Demonstration
    print("\n[2] Demonstrating FeaturePipeline...")
    pipeline = FeaturePipeline()

    # Process a small debug subset (20 samples)
    # We disable loading from cache to force the pipeline to run the logic
    debug_size = 20
    X_train, y_train, X_val, y_val, X_test, test_ids = pipeline.process_data(
        load_cached_data=False, debug_sample_size=debug_size
    )

    # Assertions to verify pipeline output
    print("Verifying FeaturePipeline outputs...")

    # Check Training Data
    assert isinstance(X_train, np.ndarray), "X_train must be a numpy array"
    assert X_train.shape == (
        debug_size,
        Config.INPUT_DIM,
    ), f"Expected X_train shape ({debug_size}, {Config.INPUT_DIM}), got {X_train.shape}"
    assert y_train.shape == (
        debug_size,
        3,
    ), f"Expected y_train shape ({debug_size}, 3), got {y_train.shape}"

    # Check Validation Data
    assert X_val.shape == (
        debug_size,
        Config.INPUT_DIM,
    ), f"Expected X_val shape ({debug_size}, {Config.INPUT_DIM}), got {X_val.shape}"

    # Check Test Data
    assert X_test.shape == (
        debug_size,
        Config.INPUT_DIM,
    ), f"Expected X_test shape ({debug_size}, {Config.INPUT_DIM}), got {X_test.shape}"
    assert len(test_ids) == debug_size, "Test IDs count mismatch"

    print("FeaturePipeline verification passed.")

    # 3. Dataset Demonstration
    print("\n[3] Demonstrating ArenaDataset...")
    train_dataset = ArenaDataset(X_train, y_train)
    test_dataset = ArenaDataset(X_test)  # No targets for test

    # Verify length
    assert len(train_dataset) == debug_size
    assert len(test_dataset) == debug_size

    # Verify getitem for Train (Features + Target)
    feat, target = train_dataset[0]
    assert torch.is_tensor(feat), "Feature must be a tensor"
    assert torch.is_tensor(target), "Target must be a tensor"
    assert feat.shape == (
        Config.INPUT_DIM,
    ), f"Feature tensor shape mismatch: {feat.shape}"
    assert target.shape == (3,), f"Target tensor shape mismatch: {target.shape}"

    # Verify getitem for Test (Features only)
    feat_test = test_dataset[0]
    assert torch.is_tensor(feat_test), "Test feature must be a tensor"
    # Note: ArenaDataset implementation returns (features, targets) if targets are not None.
    # If targets are None (default), it returns just features.
    # Let's verify the behavior based on initialization.
    assert feat_test.shape == (Config.INPUT_DIM,), "Test feature shape mismatch"

    print("ArenaDataset verification passed.")

    # 4. Model Demonstration
    print("\n[4] Demonstrating ClassifierMLP...")
    model = ClassifierMLP(
        input_dim=Config.INPUT_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        output_dim=Config.OUTPUT_DIM,
        dropout_rate=0.1,
    )

    # Move to device
    device = get_device()
    model.to(device)

    # Create a dummy batch
    dummy_input = torch.randn(4, Config.INPUT_DIM).to(device)

    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)

    # Verify output shape (Batch Size, Num Classes)
    assert output.shape == (
        4,
        3,
    ), f"Model output shape mismatch. Expected (4, 3), got {output.shape}"
    print("ClassifierMLP forward pass verification passed.")

    # 5. Full Training Task Integration
    print("\n[5] Running Full Training Task (Integration Test)...")

    # We use run_training_task from library.trainer
    # This function encapsulates: Pipeline -> Dataset -> DataLoader -> Model -> Train -> Predict -> Submit
    # We use a slightly larger debug size (50) to ensure batching logic works (batch_size=8)
    integration_debug_size = 50

    run_training_task(
        debug_sample_size=integration_debug_size,
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        load_cached_data=False,  # Force re-computation to test full flow
    )

    # 6. Verify Submission
    print("\n[6] Verifying Submission File...")
    submission_path = Config.SUBMISSION_PATH

    if not os.path.exists(submission_path):
        raise FileNotFoundError(f"Submission file was not created at {submission_path}")

    df_sub = pd.read_csv(submission_path)

    # Check shape
    assert (
        len(df_sub) == integration_debug_size
    ), f"Submission rows mismatch. Expected {integration_debug_size}, got {len(df_sub)}"

    # Check columns
    expected_cols = ["id", "winner_model_a", "winner_model_b", "winner_tie"]
    assert all(
        col in df_sub.columns for col in expected_cols
    ), f"Submission columns mismatch. Expected {expected_cols}, got {list(df_sub.columns)}"

    # Check values (probabilities should sum to ~1, but logits -> softmax might have slight float precision diffs)
    # We just check if values are numeric and within range [0, 1]
    probs = df_sub[["winner_model_a", "winner_model_b", "winner_tie"]].values
    assert (probs >= 0).all() and (
        probs <= 1.0001
    ).all(), "Probabilities out of range [0, 1]"

    print(f"Submission file verified successfully at {submission_path}")
    print("\nAll demonstrations and verifications completed successfully.")


if __name__ == "__main__":
    main()
