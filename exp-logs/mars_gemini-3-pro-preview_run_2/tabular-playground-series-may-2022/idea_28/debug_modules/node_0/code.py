import os
import torch
import numpy as np
import pandas as pd
import sys

# Import from the provided library files
from library.dataset import get_dataloaders
from library.architecture import SwishGatedResFunnel
from library.trainer import Trainer

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
SEED = 42
BATCH_SIZE = 4096  # Large batch size for speed on A100
EPOCHS = 1  # Minimal epochs for demonstration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=42):
    """Sets fixed seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def verify_data_pipeline():
    """
    Verifies that the data loader yields batches with correct shapes and types.
    """
    print("\n=== Verifying Data Pipeline ===")

    # Load dataloaders
    # Note: This will trigger data processing/caching if not already present in ./working/idea_28
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        batch_size=BATCH_SIZE, load_cached_data=True
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")
    print(f"Test batches:  {len(test_loader)}")

    # Fetch one batch from training
    batch = next(iter(train_loader))

    num_x = batch["numerical"]
    cat_x = batch["categorical"]
    target = batch["target"]

    # Assertions
    # Numerical features: f_00 to f_30 excluding f_27 -> 30 features
    assert num_x.dim() == 2, f"Expected 2D numerical tensor, got {num_x.dim()}"
    assert num_x.shape[1] == 30, f"Expected 30 numerical features, got {num_x.shape[1]}"

    # Categorical features: f_27 is a sequence of 10 characters
    assert cat_x.dim() == 2, f"Expected 2D categorical tensor, got {cat_x.dim()}"
    assert cat_x.shape[1] == 10, f"Expected sequence length 10, got {cat_x.shape[1]}"

    # Target: Binary classification
    assert target.dim() == 2, f"Expected 2D target tensor, got {target.dim()}"
    assert target.shape[1] == 1, f"Expected target dim 1, got {target.shape[1]}"

    print("Data Pipeline Verification Passed.")
    return train_loader, val_loader


def verify_model_architecture():
    """
    Verifies that the model accepts input tensors and produces valid output.
    """
    print("\n=== Verifying Model Architecture ===")

    model = SwishGatedResFunnel().to(DEVICE)
    model.eval()

    # Create dummy inputs
    dummy_num = torch.randn(BATCH_SIZE, 30).to(DEVICE)
    # Categorical inputs are indices 0-25
    dummy_cat = torch.randint(0, 26, (BATCH_SIZE, 10)).to(DEVICE)

    with torch.no_grad():
        output = model(dummy_num, dummy_cat)

    # Check output shape
    assert output.shape == (
        BATCH_SIZE,
        1,
    ), f"Expected output shape {(BATCH_SIZE, 1)}, got {output.shape}"

    # Check output range (Sigmoid)
    assert (
        output.min() >= 0.0 and output.max() <= 1.0
    ), "Output values out of range [0, 1]"

    print("Model Architecture Verification Passed.")


def run_training_demonstration():
    """
    Runs the Trainer to demonstrate training loop and inference.
    """
    print("\n=== Running Training Demonstration ===")

    # Initialize Trainer
    trainer = Trainer(device=DEVICE)

    # Run Fit
    # We use 1 epoch to keep the runtime short for demonstration
    print(f"Starting training for {EPOCHS} epoch(s)...")
    trainer.fit(epochs=EPOCHS, batch_size=BATCH_SIZE, patience=1)

    # Check if model was saved
    if not os.path.exists(trainer.best_model_path):
        raise FileNotFoundError(
            f"Model checkpoint not found at {trainer.best_model_path}"
        )

    # Run Predict
    print("Running inference...")
    trainer.predict(batch_size=BATCH_SIZE)

    # Verify Submission
    submission_path = "./submission/submission.csv"
    if not os.path.exists(submission_path):
        raise FileNotFoundError(f"Submission file not found at {submission_path}")

    df_sub = pd.read_csv(submission_path)

    # Check submission format
    assert (
        "id" in df_sub.columns and "target" in df_sub.columns
    ), "Submission missing required columns"
    assert len(df_sub) == 100000, f"Expected 100,000 predictions, got {len(df_sub)}"

    # Check value validity
    assert (
        df_sub["target"].min() >= 0 and df_sub["target"].max() <= 1
    ), "Predictions out of probability range"

    print(f"Training Demonstration Passed. Submission saved to {submission_path}")
    print(df_sub.head())


if __name__ == "__main__":
    set_seed(SEED)

    print(f"Running on device: {DEVICE}")

    # 1. Verify Data Loading
    verify_data_pipeline()

    # 2. Verify Model Forward Pass
    verify_model_architecture()

    # 3. Run Training and Inference
    run_training_demonstration()

    print("\nAll demonstrations completed successfully.")
