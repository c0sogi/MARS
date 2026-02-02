import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import warnings

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

# Ensure the current directory is in the path to import library modules
sys.path.append(os.getcwd())

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything
from library.dataset import (
    EEGSpecDataset,
    get_train_dataloader,
    get_val_dataloader,
    get_test_dataloader,
)
from library.models import PyramidFusionNet
from library.trainer import Trainer


def run_demo():
    print("--- Starting Library Usage Demo ---")

    # 1. Setup and Configuration Override
    # We modify the Config class attributes directly to create a "Lite" version for demonstration.
    print("\n[1] Configuring environment...")

    # Set a specific working directory for this demo
    Config.WORKING_DIR = "./working/demo_run"
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.SUBMISSION_DIR = Config.WORKING_DIR

    # Reduce data size for speed
    Config.TRAIN_SUBSAMPLE_SIZE = 50  # Only use 50 samples for training
    Config.BATCH_SIZE = 8
    Config.EPOCHS = 2
    Config.NUM_WORKERS = 0  # Use main thread to avoid overhead in demo

    # Ensure directories exist
    Config.setup()

    # Set seeds for reproducibility
    seed_everything(Config.SEED)

    device = Config.DEVICE
    print(f"    Device: {device}")
    print(f"    Working Directory: {Config.WORKING_DIR}")

    # 2. Data Loading and Validation
    print("\n[2] Testing Data Loading...")

    # Load metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)

    # For demo speed, also subsample the validation dataframe loaded into memory
    # (The Dataset class handles train subsampling via Config, but we do val manually here for speed)
    val_df = val_df.head(20).reset_index(drop=True)

    print(f"    Loaded Metadata: Train={len(train_df)}, Val (Subsampled)={len(val_df)}")

    # Instantiate DataLoaders
    train_loader = get_train_dataloader(train_df)
    val_loader = get_val_dataloader(val_df)

    # Verify Batch Shapes
    # Fetch one batch to ensure pipeline works
    eeg_batch, spec_batch, target_batch = next(iter(train_loader))

    print("    Batch Shapes Verification:")
    print(f"    EEG: {eeg_batch.shape} (Expected: [{Config.BATCH_SIZE}, 20, 5000])")
    print(
        f"    Spec: {spec_batch.shape} (Expected: [{Config.BATCH_SIZE}, 5, 512, 512])"
    )
    print(f"    Targets: {target_batch.shape} (Expected: [{Config.BATCH_SIZE}, 6])")

    # Assertions
    assert eeg_batch.shape == (Config.BATCH_SIZE, 20, 5000), "EEG shape mismatch"
    assert spec_batch.shape == (
        Config.BATCH_SIZE,
        5,
        512,
        512,
    ), "Spectrogram shape mismatch"
    assert target_batch.shape == (Config.BATCH_SIZE, 6), "Target shape mismatch"
    assert torch.isfinite(eeg_batch).all(), "NaNs/Infs found in EEG data"

    # 3. Model Initialization
    print("\n[3] Initializing Model...")
    model = PyramidFusionNet()
    model.to(device)

    # Forward pass check
    with torch.no_grad():
        dummy_out = model(eeg_batch.to(device), spec_batch.to(device))

    print(f"    Model Output Shape: {dummy_out.shape}")
    assert dummy_out.shape == (Config.BATCH_SIZE, 6), "Model output shape mismatch"

    # Check Softmax constraint (sums approx 1.0)
    sums = dummy_out.sum(dim=1).cpu().numpy()
    assert np.allclose(
        sums, 1.0, atol=1e-5
    ), "Model outputs do not sum to 1 (Softmax failure)"
    print("    Model initialized and forward pass verified.")

    # 4. Training Loop
    print("\n[4] Running Training Loop (Demo)...")

    # Setup Optimizer and Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )

    # Simple OneCycleLR for the demo duration
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LR,
        epochs=Config.EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        patience=Config.PATIENCE,
    )

    # Run training
    trainer.fit(epochs=Config.EPOCHS)

    # Verify checkpoint creation
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        print(f"    Success: Checkpoint found at {best_model_path}")
    else:
        raise FileNotFoundError("Training finished but best_model.pth was not created.")

    # 5. Inference and Submission
    print("\n[5] Running Inference on Test Set...")

    # Load Test Metadata
    test_df = pd.read_csv(Config.TEST_CSV)
    # Subsample test for demo speed
    test_df = test_df.head(20).reset_index(drop=True)

    test_loader = get_test_dataloader(test_df)

    # Load Best Model
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    predictions = []

    with torch.no_grad():
        for eeg, spec in test_loader:
            eeg = eeg.to(device)
            spec = spec.to(device)

            outputs = model(eeg, spec)
            predictions.append(outputs.cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)

    # Create Submission DataFrame
    sub_df = pd.DataFrame(predictions, columns=Config.TARGET_COLS)
    sub_df["eeg_id"] = test_df["eeg_id"]

    # Reorder columns to match submission format: eeg_id, then votes
    cols = ["eeg_id"] + Config.TARGET_COLS
    sub_df = sub_df[cols]

    # Save
    submission_path = os.path.join(Config.WORKING_DIR, "submission.csv")
    sub_df.to_csv(submission_path, index=False)

    print(f"    Predictions shape: {predictions.shape}")
    print(f"    Submission saved to: {submission_path}")

    # Final Validation of Submission File
    saved_df = pd.read_csv(submission_path)
    assert len(saved_df) == len(test_df), "Submission row count mismatch"
    assert list(saved_df.columns) == cols, "Submission column mismatch"

    # Check probability sums in submission
    row_sums = saved_df[Config.TARGET_COLS].sum(axis=1)
    assert np.allclose(
        row_sums, 1.0, atol=1e-4
    ), "Submission probabilities do not sum to 1"

    print("\n--- Demo Completed Successfully ---")


if __name__ == "__main__":
    run_demo()
