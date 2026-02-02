import os
import numpy as np
import pandas as pd
import torch
import random
from library.config import Config
from library.data_loader import get_data_loaders
from library.model import D2N
from library.train import run_training
from library.predict import generate_submission


def set_seeds(seed=42):
    """Sets fixed seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def verify_data_loaders():
    """Verifies that data loaders return batches with correct shapes."""
    print("\n[1/4] Verifying Data Loaders...")

    # Use a small batch size and sample count for speed
    batch_size = 8
    max_samples = 50

    # Force reload to verify processing logic (load_cached_data=False)
    train_loader, val_loader, test_loader, test_ids = get_data_loaders(
        batch_size=batch_size, load_cached_data=False, max_samples=max_samples
    )

    # Check Train Loader
    X_batch, y_batch = next(iter(train_loader))

    # Expected Input Dim: (32 * 32 * 2) + 1 (inc_angle) = 2049
    expected_dim = Config.INPUT_DIM

    print(f"  Train Batch X shape: {X_batch.shape}")
    print(f"  Train Batch y shape: {y_batch.shape}")

    if X_batch.shape[1] != expected_dim:
        raise AssertionError(
            f"Expected input dimension {expected_dim}, got {X_batch.shape[1]}"
        )

    if len(X_batch) != batch_size and len(X_batch) != max_samples:
        # Note: Last batch might be smaller, but here we check the first batch
        # If max_samples < batch_size, len is max_samples
        pass

    # Check Test Loader (should not have labels)
    X_test_batch = next(iter(test_loader))
    # DataLoader returns a list/tuple, so X_test_batch is a tuple (tensor,)
    if isinstance(X_test_batch, (list, tuple)):
        X_test_tensor = X_test_batch[0]
    else:
        X_test_tensor = X_test_batch

    print(f"  Test Batch X shape: {X_test_tensor.shape}")
    if X_test_tensor.shape[1] != expected_dim:
        raise AssertionError(
            f"Expected test input dimension {expected_dim}, got {X_test_tensor.shape[1]}"
        )

    print("  Data Loader verification successful.")
    return train_loader


def verify_model_architecture(train_loader):
    """Verifies model instantiation and forward pass."""
    print("\n[2/4] Verifying Model Architecture...")

    model = D2N(
        input_dim=Config.INPUT_DIM,
        hidden_units=Config.HIDDEN_UNITS,
        dropout_rate=Config.DROPOUT_RATE,
    )

    # Get a batch from the loader
    X_batch, _ = next(iter(train_loader))

    # Move to appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    X_batch = X_batch.to(device)

    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(X_batch)

    print(f"  Model Output shape: {output.shape}")

    # Check shape: (batch_size, 1)
    if output.shape != (X_batch.shape[0], 1):
        raise AssertionError(
            f"Expected output shape {(X_batch.shape[0], 1)}, got {output.shape}"
        )

    # Check value range (Sigmoid should be between 0 and 1)
    if output.min() < 0 or output.max() > 1:
        raise AssertionError("Model outputs are not within [0, 1] range.")

    print("  Model architecture verification successful.")


def verify_training_pipeline():
    """Verifies the training loop and checkpoint generation."""
    print("\n[3/4] Verifying Training Pipeline...")

    # Run training for a minimal number of epochs on a subset
    run_training(
        epochs=2,
        batch_size=8,
        learning_rate=0.001,
        patience=2,
        max_samples=50,
        load_cached_data=True,  # Use data cached from step 1
    )

    # Check if model checkpoint was created
    if not os.path.exists(Config.MODEL_CHECKPOINT):
        raise FileNotFoundError(
            f"Model checkpoint not found at {Config.MODEL_CHECKPOINT}"
        )

    print(f"  Model checkpoint found at {Config.MODEL_CHECKPOINT}")
    print("  Training pipeline verification successful.")


def verify_prediction_pipeline():
    """Verifies submission generation."""
    print("\n[4/4] Verifying Prediction Pipeline...")

    # Generate submission using the model trained in the previous step
    generate_submission(load_cached_data=True, batch_size=8, max_samples=50)

    # Check if submission file exists
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    # Validate submission format
    df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"  Submission shape: {df.shape}")
    print(f"  Columns: {df.columns.tolist()}")

    expected_cols = ["id", "is_iceberg"]
    if not all(col in df.columns for col in expected_cols):
        raise AssertionError(
            f"Submission missing required columns. Found: {df.columns}"
        )

    # Check probability values
    if df["is_iceberg"].min() < 0 or df["is_iceberg"].max() > 1:
        raise AssertionError("Submission probabilities out of range [0, 1].")

    print("  Prediction pipeline verification successful.")


if __name__ == "__main__":
    # Setup
    set_seeds(42)
    Config.setup()

    print("Starting Integration Tests...")
    print("=============================")

    # Execute steps
    loader = verify_data_loaders()
    verify_model_architecture(loader)
    verify_training_pipeline()
    verify_prediction_pipeline()

    print("\n=============================")
    print("All integration tests passed successfully.")
