import os
import shutil
import numpy as np
import pandas as pd
import torch
import torch.optim as optim

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, MetricTracker
from library.loss import MaskedMCRMSELoss
from library.data import get_loaders
from library.model import DenseStackingHybridNet
from library.train import train_epoch, validate, generate_submission


def run_demo():
    # =========================================================================
    # 1. Configuration & Setup
    # =========================================================================
    print("1. Setting up configuration for demo...")

    # Override Config for speed and isolation
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "data_cache")
    Config.BATCH_SIZE = 8
    Config.EPOCHS = 2
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Setup directories
    Config.setup()

    # Set seeds for reproducibility
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"   Working Directory: {Config.WORKING_DIR}")
    print(f"   Device: {device}")

    # =========================================================================
    # 2. Data Loading (Debug Mode)
    # =========================================================================
    print("\n2. Loading data in debug mode (subset)...")

    # get_loaders(debug=True) loads only 32 samples
    train_loader, val_loader, test_loader, test_ids = get_loaders(debug=True)

    # Verify Data Shapes
    sample_inputs, sample_neighbors, sample_targets = next(iter(train_loader))

    print(f"   Batch Inputs Shape: {sample_inputs.shape}")  # (B, 107, 19)
    print(f"   Batch Neighbors Shape: {sample_neighbors.shape}")  # (B, 107, 3)
    print(f"   Batch Targets Shape: {sample_targets.shape}")  # (B, 107, 5)

    assert sample_inputs.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        Config.INPUT_DIM,
    ), "Input shape mismatch"
    assert sample_neighbors.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        3,
    ), "Neighbor indices shape mismatch"
    assert sample_targets.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        5,
    ), "Target shape mismatch"

    # =========================================================================
    # 3. Model Initialization & Forward Pass Verification
    # =========================================================================
    print("\n3. Initializing Model and verifying forward pass...")

    model = DenseStackingHybridNet().to(device)

    # Move sample batch to device
    sample_inputs = sample_inputs.to(device)
    sample_neighbors = sample_neighbors.to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(sample_inputs, sample_neighbors)

    print(f"   Model Output Shape: {outputs.shape}")

    assert outputs.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        5,
    ), "Model output shape mismatch"
    assert not torch.isnan(outputs).any(), "Model produced NaN values"

    # =========================================================================
    # 4. Loss Function Verification
    # =========================================================================
    print("\n4. Verifying MaskedMCRMSELoss...")

    criterion = MaskedMCRMSELoss().to(device)
    sample_targets = sample_targets.to(device)

    loss = criterion(outputs, sample_targets)
    print(f"   Calculated Loss: {loss.item():.6f}")

    assert loss.dim() == 0, "Loss should be a scalar"
    assert loss.item() >= 0, "Loss should be non-negative"

    # =========================================================================
    # 5. MetricTracker Logic Verification
    # =========================================================================
    print("\n5. Verifying MetricTracker logic...")

    tracker = MetricTracker()

    # Create synthetic data to test math
    # Scored columns are indices [0, 1, 3] (reactivity, deg_Mg_pH10, deg_Mg_50C)
    # Let's make targets all 0.0 and preds all 1.0 for scored columns
    # Squared Error = (0-1)^2 = 1.0
    # RMSE = sqrt(1.0) = 1.0
    # MCRMSE = Mean(1.0, 1.0, 1.0) = 1.0

    y_true_syn = np.zeros((10, 5))
    y_pred_syn = np.zeros((10, 5))

    # Set predictions for scored columns to 1.0
    scored_indices = [0, 1, 3]
    y_pred_syn[:, scored_indices] = 1.0

    # Update tracker
    tracker.update(y_true_syn, y_pred_syn)
    result = tracker.compute()

    print(f"   Synthetic MCRMSE (Expected 1.0): {result:.6f}")
    assert np.isclose(result, 1.0), f"MetricTracker failed. Expected 1.0, got {result}"

    # Reset for training
    tracker.reset()

    # =========================================================================
    # 6. Training Loop Simulation
    # =========================================================================
    print("\n6. Running Training Loop (2 Epochs)...")

    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_mcrmse = validate(model, val_loader, criterion, tracker, device)

        print(
            f"   Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val MCRMSE={val_mcrmse:.4f}"
        )

        # Basic sanity check: Loss should be finite
        assert np.isfinite(train_loss), "Training loss is infinite or NaN"
        assert np.isfinite(val_loss), "Validation loss is infinite or NaN"

    # Save dummy model for submission generation
    model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"   Saved model to {model_path}")

    # =========================================================================
    # 7. Submission Generation
    # =========================================================================
    print("\n7. Generating Submission...")

    submission_path = os.path.join(Config.WORKING_DIR, "submission.csv")
    generate_submission(model, test_loader, test_ids, device, submission_path)

    # Verify submission file
    assert os.path.exists(submission_path), "Submission file was not created"

    df_sub = pd.read_csv(submission_path)
    print(f"   Submission Shape: {df_sub.shape}")
    print(f"   First few rows:\n{df_sub.head()}")

    # Expected rows: Number of test samples (32 in debug) * Sequence Length (107)
    # Note: test_loader in debug mode loads 32 samples.
    expected_rows = 32 * Config.SEQ_LEN
    assert (
        len(df_sub) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(df_sub)}"

    # Check columns
    expected_cols = ["id_seqpos"] + Config.TARGET_COLS
    assert list(df_sub.columns) == expected_cols, "Submission columns mismatch"

    print("\nAll demonstration steps completed successfully.")


if __name__ == "__main__":
    run_demo()
