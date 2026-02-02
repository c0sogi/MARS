import os
import sys
import shutil
import torch
import pandas as pd
import numpy as np
import warnings

# Import from the provided library
from library.config import Config
from library.dataset import EEGDataset
from library.model import TimeRelativeTransformer
from library.engine import train_loop, generate_submission
from library.utils import seed_everything

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    print("----------------------------------------------------------------")
    print("Starting Demo Execution of HMS Brain Activity Classification")
    print("----------------------------------------------------------------")

    # ----------------------------------------------------------------
    # 1. Configuration & Setup
    # ----------------------------------------------------------------
    # Override Config for a fast demonstration run
    print("[1/6] Configuring environment...")

    # Set paths to a dedicated demo directory in ./working
    Config.OUTPUT_DIR = "./working/demo_execution"
    Config.CACHE_DIR = os.path.join(Config.OUTPUT_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(Config.OUTPUT_DIR, "checkpoints")
    Config.SUBMISSION_PATH = os.path.join(Config.OUTPUT_DIR, "submission.csv")

    # Hyperparameters for speed
    Config.BATCH_SIZE = 4
    Config.EPOCHS = 1
    Config.NUM_WORKERS = (
        0  # Use 0 workers to avoid potential multiprocessing overhead in demo
    )

    # Ensure directories exist
    if os.path.exists(Config.OUTPUT_DIR):
        shutil.rmtree(Config.OUTPUT_DIR)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    # Set Random Seed
    seed_everything(Config.SEED)

    device = Config.DEVICE
    print(f"      Device: {device}")
    print(f"      Output Directory: {Config.OUTPUT_DIR}")

    # ----------------------------------------------------------------
    # 2. Data Loading & Verification
    # ----------------------------------------------------------------
    print("\n[2/6] Initializing Datasets (Subsampled)...")

    # Use a small sample size (e.g., 20 samples) to verify pipeline without processing all data
    sample_size = 20

    train_ds = EEGDataset(mode="train", load_cached_data=False, sample_size=sample_size)
    val_ds = EEGDataset(mode="val", load_cached_data=False, sample_size=sample_size)
    test_ds = EEGDataset(mode="test", load_cached_data=False, sample_size=sample_size)

    print(
        f"      Train size: {len(train_ds)}, Val size: {len(val_ds)}, Test size: {len(test_ds)}"
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=Config.BATCH_SIZE, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=Config.BATCH_SIZE, shuffle=False
    )

    # Verify Batch Shapes
    print("      Verifying batch shapes...")
    batch = next(iter(train_loader))
    eeg, spec, target = batch["eeg"], batch["spec"], batch["target"]

    # Expected: EEG=(B, 20, 5000), Spec=(B, 1, 512, 512), Target=(B, 6)
    assert eeg.shape == (Config.BATCH_SIZE, 20, 5000), f"EEG Shape Error: {eeg.shape}"
    assert spec.shape == (
        Config.BATCH_SIZE,
        1,
        512,
        512,
    ), f"Spec Shape Error: {spec.shape}"
    assert target.shape == (Config.BATCH_SIZE, 6), f"Target Shape Error: {target.shape}"
    print("      Batch shapes verified successfully.")

    # ----------------------------------------------------------------
    # 3. Model Initialization & Verification
    # ----------------------------------------------------------------
    print("\n[3/6] Initializing Model...")

    model = TimeRelativeTransformer(Config).to(device)

    # Verify Forward Pass
    print("      Verifying forward pass...")
    with torch.no_grad():
        dummy_eeg = eeg.to(device)
        dummy_spec = spec.to(device)
        output = model(dummy_eeg, dummy_spec)

    assert output.shape == (Config.BATCH_SIZE, 6), f"Output Shape Error: {output.shape}"

    # Verify Softmax (Sum to 1)
    sums = output.sum(dim=1).cpu().numpy()
    assert np.allclose(sums, 1.0, atol=1e-5), f"Probability Sum Error: {sums}"
    print("      Model forward pass verified successfully.")

    # ----------------------------------------------------------------
    # 4. Training Loop
    # ----------------------------------------------------------------
    print("\n[4/6] Starting Training Loop (1 Epoch)...")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.MAX_LR,
        epochs=Config.EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=Config.PCT_START,
        div_factor=Config.DIV_FACTOR,
        final_div_factor=Config.FINAL_DIV_FACTOR,
    )

    best_score = train_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=Config.EPOCHS,
        checkpoint_dir=Config.CHECKPOINT_DIR,
        patience=1,
    )

    print(f"      Training finished. Best Validation Score: {best_score:.4f}")

    # Verify Checkpoint
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    assert os.path.exists(checkpoint_path), "Checkpoint file was not created."
    print("      Checkpoint verified.")

    # ----------------------------------------------------------------
    # 5. Submission Generation
    # ----------------------------------------------------------------
    print("\n[5/6] Generating Submission...")

    # Load Best Model
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])

    generate_submission(model, test_loader, device, Config.SUBMISSION_PATH)

    # Verify Submission File
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"      Submission shape: {sub_df.shape}")

    # Check columns
    expected_cols = ["eeg_id"] + Config.CLASS_NAMES
    assert (
        list(sub_df.columns) == expected_cols
    ), f"Submission columns mismatch. Got {list(sub_df.columns)}"

    # Check row count (should match test_ds length)
    assert len(sub_df) == len(
        test_ds
    ), f"Submission row count mismatch. Expected {len(test_ds)}, got {len(sub_df)}"

    # Check probability sums
    row_sums = sub_df[Config.CLASS_NAMES].sum(axis=1)
    assert np.allclose(
        row_sums, 1.0, atol=1e-4
    ), "Submission probabilities do not sum to 1."

    print("      Submission file verified successfully.")

    # ----------------------------------------------------------------
    # 6. Conclusion
    # ----------------------------------------------------------------
    print("\n[6/6] Demo Execution Complete.")
    print(f"      Results saved to: {Config.OUTPUT_DIR}")


if __name__ == "__main__":
    run_demo()
