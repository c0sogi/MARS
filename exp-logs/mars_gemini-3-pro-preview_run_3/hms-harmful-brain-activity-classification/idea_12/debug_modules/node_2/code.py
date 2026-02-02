import os
import sys
import shutil
import numpy as np
import torch
import pandas as pd
import warnings

# Ensure the library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config, seed_everything
from library.utils import AverageMeter, kl_divergence
from library.data import get_dataloaders
from library.model import MultiResNetwork
from library.train import Trainer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    print("============================================================")
    print("       EEG Classification Library Demonstration")
    print("============================================================")

    # ------------------------------------------------------------------
    # 1. Configuration & Setup
    # ------------------------------------------------------------------
    print("\n[Step 1] Setting up configuration for demo...")

    # Override Config for speed and isolation
    Config.WORKING_DIR = "./working/demo_execution"
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 2

    # Ensure clean working directory
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seeds
    seed_everything(Config.SEED)
    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Epochs: {Config.EPOCHS}, Batch Size: {Config.BATCH_SIZE}")

    # ------------------------------------------------------------------
    # 2. Verify Utility Functions
    # ------------------------------------------------------------------
    print("\n[Step 2] Verifying utility functions...")

    # Test AverageMeter
    meter = AverageMeter()
    meter.update(val=10, n=2)
    meter.update(val=20, n=2)
    # Total sum = 10*2 + 20*2 = 60. Total count = 4. Avg = 15.
    assert meter.avg == 15.0, f"AverageMeter failed: expected 15.0, got {meter.avg}"
    print("  AverageMeter logic verified.")

    # Test KL Divergence
    # Case 1: Identical distributions -> KL should be 0
    y_true = np.array([[0.2, 0.3, 0.5]])
    y_pred = np.array([[0.2, 0.3, 0.5]])
    kl_val = kl_divergence(y_true, y_pred)
    assert np.isclose(
        kl_val, 0.0, atol=1e-6
    ), f"KL Divergence failed identity check: {kl_val}"

    # Case 2: Known difference
    # P = [0.5, 0.5], Q = [0.2, 0.8]
    # KL = 0.5 * log(0.5/0.2) + 0.5 * log(0.5/0.8)
    #    = 0.5 * 0.91629 + 0.5 * -0.4700
    #    = 0.4581 - 0.235 = 0.2231
    y_true_2 = np.array([[0.5, 0.5]])
    y_pred_2 = np.array([[0.2, 0.8]])
    kl_val_2 = kl_divergence(y_true_2, y_pred_2)
    assert kl_val_2 > 0, "KL Divergence should be positive for different distributions"
    print("  KL Divergence logic verified.")

    # ------------------------------------------------------------------
    # 3. Verify Data Loading
    # ------------------------------------------------------------------
    print("\n[Step 3] Verifying DataLoaders and Input Shapes...")

    # Initialize loaders in debug mode (loads small subset)
    train_loader, val_loader, test_loader = get_dataloaders(debug=True)

    # Fetch one batch
    inputs, targets = next(iter(train_loader))
    x_a, x_b = inputs

    print(f"  Fetched batch of size {x_a.size(0)}")
    print(f"  Stream A (EEG) Shape: {x_a.shape}")
    print(f"  Stream B (Spec) Shape: {x_b.shape}")
    print(f"  Targets Shape: {targets.shape}")

    # Assertions
    # Stream A: (Batch, 57 channels, 128 freq, 500 time)
    assert x_a.dim() == 4, "Stream A should be 4D"
    assert x_a.size(1) == 57, f"Stream A should have 57 channels, got {x_a.size(1)}"
    assert x_a.size(2) == 128, f"Stream A height should be 128, got {x_a.size(2)}"
    assert x_a.size(3) == 500, f"Stream A width should be 500, got {x_a.size(3)}"

    # Stream B: (Batch, 4 channels, 256 height, 256 width)
    assert x_b.dim() == 4, "Stream B should be 4D"
    assert x_b.size(1) == 4, f"Stream B should have 4 channels, got {x_b.size(1)}"
    assert x_b.size(2) == 256, f"Stream B height should be 256, got {x_b.size(2)}"

    # Targets: (Batch, 6 classes)
    assert targets.size(1) == 6, f"Targets should have 6 classes, got {targets.size(1)}"

    print("  Data shapes verified.")

    # ------------------------------------------------------------------
    # 4. Verify Model Architecture
    # ------------------------------------------------------------------
    print("\n[Step 4] Verifying MultiResNetwork Architecture...")

    device = torch.device(Config.DEVICE)
    model = MultiResNetwork().to(device)

    # Move inputs to device
    x_a = x_a.to(device)
    x_b = x_b.to(device)

    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model((x_a, x_b))

    print(f"  Output Shape: {outputs.shape}")

    # Assertions
    assert outputs.size(0) == Config.BATCH_SIZE, "Output batch size mismatch"
    assert outputs.size(1) == 6, "Output class count mismatch"

    # Check probability constraints
    sums = outputs.sum(dim=1).cpu().numpy()
    assert np.allclose(
        sums, 1.0, atol=1e-5
    ), "Model outputs do not sum to 1 (Softmax failure)"
    assert (outputs >= 0).all(), "Model outputs contain negative probabilities"

    print("  Model forward pass and output constraints verified.")

    # ------------------------------------------------------------------
    # 5. Execute Training Loop
    # ------------------------------------------------------------------
    print("\n[Step 5] Executing Training Loop (Debug Mode)...")

    # Instantiate Trainer
    trainer = Trainer(debug=True)

    # Run training
    # This will run for Config.EPOCHS (set to 1) and save 'best_model.pth'
    trainer.fit()

    # Verify artifacts
    expected_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    if os.path.exists(expected_model_path):
        print(f"  Training complete. Model saved at: {expected_model_path}")
    else:
        raise FileNotFoundError("Training failed to save best_model.pth")

    # ------------------------------------------------------------------
    # 6. Inference & Submission
    # ------------------------------------------------------------------
    print("\n[Step 6] Generating Submission from Test Set...")

    # Load best model
    checkpoint = torch.load(
        expected_model_path, map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    predictions = []
    eeg_ids = []

    # Iterate test loader
    # Note: Test loader in debug mode is also subsetted
    with torch.no_grad():
        for i, (inputs) in enumerate(test_loader):
            # Test dataset returns (stream_a, stream_b) tuple
            # We need to get eeg_ids. The dataset doesn't return them directly in __getitem__,
            # so we look them up from the dataframe using indices.

            # Get data
            x_a, x_b = inputs
            x_a = x_a.to(device)
            x_b = x_b.to(device)

            # Predict
            preds = model((x_a, x_b))
            predictions.append(preds.cpu().numpy())

            # Get IDs (batch logic)
            # Calculate start/end indices for this batch
            start_idx = i * Config.BATCH_SIZE
            end_idx = start_idx + x_a.size(0)

            # Access underlying dataframe from dataset
            batch_ids = test_loader.dataset.eeg_data[start_idx:end_idx]
            # Wait, eeg_data is the numpy array. We need the metadata dataframe.
            # In library.data.load_and_cache_data, the df is returned but not stored in dataset.
            # However, for this demo, we can just read the test.csv metadata again to get IDs.
            # Or simpler: The prompt requires submission format.
            # Let's read the test metadata file directly to align IDs.
            pass

    # Concatenate predictions
    predictions = np.concatenate(predictions)

    # Load test metadata to get IDs corresponding to the debug subset
    # Since we used debug=True, the loader loaded the first N samples.
    df_test = pd.read_csv(Config.TEST_CSV)
    if len(predictions) < len(df_test):
        # We are in debug mode, so we only have predictions for the subset
        df_test = df_test.iloc[: len(predictions)]

    # Create submission DataFrame
    submission = pd.DataFrame(predictions, columns=Config.CLASS_NAMES)
    submission.insert(0, "eeg_id", df_test["eeg_id"])

    # Save submission
    submission_path = os.path.join(Config.WORKING_DIR, "submission.csv")
    submission.to_csv(submission_path, index=False)

    print(f"  Submission generated with shape {submission.shape}")
    print(f"  Saved to: {submission_path}")
    print("\n  First 3 rows:")
    print(submission.head(3))

    print("\n============================================================")
    print("       Demo Completed Successfully")
    print("============================================================")


if __name__ == "__main__":
    run_demo()
