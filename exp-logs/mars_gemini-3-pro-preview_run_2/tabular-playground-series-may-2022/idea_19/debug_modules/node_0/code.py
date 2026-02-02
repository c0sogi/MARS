import os
import sys
import torch
import torch.optim as optim
import pandas as pd
import numpy as np

# Import library components
from library.config import Config
from library.utils import seed_everything
from library.dataset import get_dataloaders
from library.model import NormFusionResFunnel
from library.engine import train_model, predict


def run_demo():
    # --------------------------------------------------------------------------
    # 1. Setup and Configuration
    # --------------------------------------------------------------------------
    print("Step 1: Setup and Configuration")

    # Ensure necessary directories exist
    Config.setup()

    # Set random seeds for reproducibility
    seed_everything(Config.SEED)

    # Define demo-specific parameters to ensure quick execution
    DEMO_EPOCHS = 1

    # Check device
    print(f"Device: {Config.DEVICE}")

    # --------------------------------------------------------------------------
    # 2. Data Loading
    # --------------------------------------------------------------------------
    print("\nStep 2: Loading Data")

    # get_dataloaders handles preprocessing and caching internally.
    # It returns PyTorch DataLoaders for train, validation, and test sets.
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # Verify DataLoaders
    print("Verifying data loaders...")

    # Fetch a single batch to verify structure and shapes
    sample_batch = next(iter(train_loader))

    # Check keys
    assert "continuous" in sample_batch, "Batch missing 'continuous' key"
    assert "sequence" in sample_batch, "Batch missing 'sequence' key"
    assert "target" in sample_batch, "Batch missing 'target' key"

    # Check shapes
    # Continuous: (Batch, 30)
    assert (
        sample_batch["continuous"].shape[1] == Config.NUM_CONT_FEATURES
    ), f"Incorrect continuous feature dim: {sample_batch['continuous'].shape[1]}"

    # Sequence: (Batch, 10)
    assert (
        sample_batch["sequence"].shape[1] == Config.SEQ_LEN
    ), f"Incorrect sequence length: {sample_batch['sequence'].shape[1]}"

    print("Data integrity verified.")

    # --------------------------------------------------------------------------
    # 3. Model Initialization
    # --------------------------------------------------------------------------
    print("\nStep 3: Model Initialization")

    model = NormFusionResFunnel().to(Config.DEVICE)

    # Setup Optimizer and Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=Config.SCHEDULER_STEP_SIZE, gamma=Config.SCHEDULER_GAMMA
    )

    print("Model instantiated successfully.")

    # --------------------------------------------------------------------------
    # 4. Training Loop
    # --------------------------------------------------------------------------
    print(f"\nStep 4: Training for {DEMO_EPOCHS} epoch(s)")

    # train_model handles the training loop, validation, and checkpointing
    best_auc = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=Config.DEVICE,
        num_epochs=DEMO_EPOCHS,
        patience=Config.EARLY_STOPPING_PATIENCE,
    )

    print(f"Training finished. Best Validation AUC: {best_auc:.4f}")

    # --------------------------------------------------------------------------
    # 5. Inference and Submission
    # --------------------------------------------------------------------------
    print("\nStep 5: Inference")

    # Load the best model saved during training
    if os.path.exists(Config.MODEL_PATH):
        checkpoint = torch.load(Config.MODEL_PATH, map_location=Config.DEVICE)
        model.load_state_dict(checkpoint)
        print(f"Loaded best model from {Config.MODEL_PATH}")
    else:
        raise FileNotFoundError(f"Model checkpoint not found at {Config.MODEL_PATH}")

    # Generate predictions
    df_submission = predict(model, test_loader, Config.DEVICE)

    # --------------------------------------------------------------------------
    # 6. Final Verification
    # --------------------------------------------------------------------------
    print("\nStep 6: Final Verification")

    # Check if file exists
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    # Check dataframe shape (Test set is 100,000 rows)
    expected_rows = 100000
    assert (
        len(df_submission) == expected_rows
    ), f"Submission has {len(df_submission)} rows, expected {expected_rows}."

    # Check probability bounds
    probs = df_submission["target"].values
    assert probs.min() >= 0.0, "Probabilities contain values < 0"
    assert probs.max() <= 1.0, "Probabilities contain values > 1"

    print("Submission verified successfully.")
    print("Demo completed.")


if __name__ == "__main__":
    run_demo()
