import os
import torch
import pandas as pd
import numpy as np
import shutil
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

# Import from the provided library
from library.config import Config
from library.utils import seed_everything
from library.data import get_dataloaders
from library.models import DualStreamModel
from library.engine import fit


# --- 1. Configuration Setup ---
class DemoConfig(Config):
    """
    Configuration override for a quick demonstration run.
    """

    # Directory settings
    OUTPUT_DIR = "./working/demo_execution"

    # Speed optimizations
    EPOCHS = 2
    DEBUG_SAMPLE_SIZE = 64  # Use only 64 samples to ensure quick execution
    BATCH_SIZE = 16  # Smaller batch size for the demo
    USE_CACHE = False  # Disable cache to verify raw data processing logic

    # Model settings (keep lightweight for demo if needed, but using default here)
    BACKBONE = "efficientnet_b0"


def run_demo():
    print("Initializing Demo Execution...")

    # Ensure output directory exists and is clean
    if os.path.exists(DemoConfig.OUTPUT_DIR):
        shutil.rmtree(DemoConfig.OUTPUT_DIR)
    os.makedirs(DemoConfig.OUTPUT_DIR, exist_ok=True)

    # Set seeds for reproducibility
    seed_everything(DemoConfig.SEED)

    # --- 2. Data Loading & Verification ---
    print("\n[Step 1] Loading Data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        train_csv_path=DemoConfig.TRAIN_CSV,
        val_csv_path=DemoConfig.VAL_CSV,
        test_csv_path=DemoConfig.TEST_CSV,
        config=DemoConfig,
    )

    # Verify Train Loader Batch
    print("Verifying data shapes...")
    eeg_batch, spec_batch, target_batch = next(iter(train_loader))

    # Assertions to verify data integrity
    # EEG: (Batch, Channels=20, Time=5000)
    assert eeg_batch.shape == (
        DemoConfig.BATCH_SIZE,
        20,
        5000,
    ), f"Incorrect EEG shape: {eeg_batch.shape}"

    # Spec: (Batch, Channels=5, Height=512, Width=512)
    # Channels = 4 (LL, RL, LP, RP) + 1 (Coord)
    assert spec_batch.shape == (
        DemoConfig.BATCH_SIZE,
        5,
        512,
        512,
    ), f"Incorrect Spectrogram shape: {spec_batch.shape}"

    # Targets: (Batch, Classes=6)
    assert target_batch.shape == (
        DemoConfig.BATCH_SIZE,
        6,
    ), f"Incorrect Target shape: {target_batch.shape}"

    # Verify targets sum to ~1
    sums = target_batch.sum(dim=1)
    assert torch.allclose(
        sums, torch.ones_like(sums), atol=1e-5
    ), "Targets do not sum to 1.0"

    print("Data shapes verified successfully.")

    # --- 3. Model Initialization & Forward Pass ---
    print("\n[Step 2] Initializing Model...")
    device = DemoConfig.DEVICE
    model = DualStreamModel(config=DemoConfig, pretrained=True).to(device)

    # Verify Forward Pass
    print("Verifying forward pass...")
    eeg_batch = eeg_batch.to(device)
    spec_batch = spec_batch.to(device)

    with torch.no_grad():
        logits = model(eeg_batch, spec_batch)

    assert logits.shape == (
        DemoConfig.BATCH_SIZE,
        6,
    ), f"Incorrect Output shape: {logits.shape}"
    assert not torch.isnan(logits).any(), "Model output contains NaNs"

    print("Forward pass successful.")

    # --- 4. Training Loop Execution ---
    print("\n[Step 3] Starting Training Loop...")

    optimizer = AdamW(
        model.parameters(),
        lr=DemoConfig.LEARNING_RATE,
        weight_decay=DemoConfig.WEIGHT_DECAY,
    )

    # Calculate total steps for OneCycleLR
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * DemoConfig.EPOCHS

    scheduler = OneCycleLR(
        optimizer,
        max_lr=DemoConfig.MAX_LR,
        total_steps=total_steps,
        pct_start=DemoConfig.PCT_START,
        div_factor=DemoConfig.DIV_FACTOR,
        final_div_factor=DemoConfig.FINAL_DIV_FACTOR,
    )

    # Run the fit engine
    fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=DemoConfig,
    )

    # --- 5. Inference & Submission ---
    print("\n[Step 4] Running Inference on Test Set...")

    # Load best model
    best_model_path = os.path.join(DemoConfig.OUTPUT_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"Loaded best model from {best_model_path}")
    else:
        print("Warning: Best model not found, using current weights.")

    model.eval()
    preds = []

    # Iterate test loader
    # Note: Test loader returns dummy targets, we ignore them
    with torch.no_grad():
        for eeg, spec, _ in test_loader:
            eeg = eeg.to(device)
            spec = spec.to(device)

            logits = model(eeg, spec)
            probs = torch.softmax(logits, dim=1)
            preds.append(probs.cpu().numpy())

    preds = np.concatenate(preds)

    # Verify predictions match test set size (clipped by debug size if applied,
    # but usually test set is full. Here we verify against loader size)
    expected_rows = len(test_loader.dataset)
    assert (
        len(preds) == expected_rows
    ), f"Prediction count {len(preds)} mismatch with test set size {expected_rows}"

    # Create submission dataframe
    # We need eeg_id from the test dataset metadata
    test_df = test_loader.dataset.df
    submission = pd.DataFrame(
        preds,
        columns=[
            "seizure_vote",
            "lpd_vote",
            "gpd_vote",
            "lrda_vote",
            "grda_vote",
            "other_vote",
        ],
    )
    submission["eeg_id"] = test_df["eeg_id"].values

    # Reorder columns to put eeg_id first
    cols = ["eeg_id"] + [c for c in submission.columns if c != "eeg_id"]
    submission = submission[cols]

    # Save submission
    sub_path = os.path.join(DemoConfig.OUTPUT_DIR, "submission.csv")
    submission.to_csv(sub_path, index=False)
    print(f"Submission saved to {sub_path}")
    print(submission.head())

    print("\nDemo Execution Completed Successfully.")


if __name__ == "__main__":
    run_demo()
