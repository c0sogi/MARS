import os
import sys
import shutil
import torch
import pandas as pd
import numpy as np
import warnings

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, MCRMSELoss
from library.data import get_dataloaders
from library.model import RNAModel
from library.train import train_one_epoch, validate, generate_submission

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def run_demo():
    print("==== RNA Degradation Prediction: Library Demo ====")

    # 1. Setup Configuration for Demo
    # We create a subclass to override settings for a quick run
    class DemoConfig(Config):
        # Use a separate directory to avoid cache conflicts with full runs
        working_dir = "./working/demo_run"

        # Debug mode forces the data loader to use only the first 100 samples
        debug = True

        # minimal training parameters
        epochs = 1
        batch_size = 4
        num_workers = 0  # Avoid multiprocessing overhead for small demo

        # Paths
        model_save_path = os.path.join(working_dir, "demo_model.pth")
        submission_path = os.path.join(working_dir, "demo_submission.csv")

    config = DemoConfig()

    # Ensure demo directory exists
    if os.path.exists(config.working_dir):
        shutil.rmtree(config.working_dir)
    os.makedirs(config.working_dir, exist_ok=True)

    # Set seeds for reproducibility
    seed_everything(config.seed)
    print(f"Configuration: Debug={config.debug}, Device={config.device}")

    # 2. Data Loading
    print("\n[Step 1] Loading Data...")
    # This will trigger preprocessing for the first 100 samples and cache them
    train_loader, val_loader, test_loader = get_dataloaders(config)

    # Verify Train Loader
    print("Verifying Train Loader batch structure...")
    train_batch = next(iter(train_loader))

    inputs = train_batch["inputs"]
    pair_index = train_batch["pair_index"]
    targets = train_batch["targets"]
    mask = train_batch["mask"]
    ids = train_batch["id"]

    # Assertions for shapes
    # Inputs: (Batch, Seq_Len, Channels=14)
    assert inputs.shape == (
        config.batch_size,
        config.seq_len,
        config.input_channels,
    ), f"Input shape mismatch: {inputs.shape}"
    # Pair Index: (Batch, Seq_Len)
    assert pair_index.shape == (
        config.batch_size,
        config.seq_len,
    ), f"Pair index shape mismatch: {pair_index.shape}"
    # Targets: (Batch, Seq_Len, Targets=5)
    assert targets.shape == (
        config.batch_size,
        config.seq_len,
        config.num_targets,
    ), f"Target shape mismatch: {targets.shape}"
    # Mask: (Batch, Seq_Len)
    assert mask.shape == (
        config.batch_size,
        config.seq_len,
    ), f"Mask shape mismatch: {mask.shape}"

    print("  -> Train batch shapes verified successfully.")

    # 3. Model Initialization
    print("\n[Step 2] Initializing Model...")
    model = RNAModel(config).to(config.device)

    # Move batch to device
    inputs = inputs.to(config.device)
    pair_index = pair_index.to(config.device)
    targets = targets.to(config.device)
    mask = mask.to(config.device)

    # 4. Forward Pass & Loss Calculation
    print("\n[Step 3] Running Forward Pass and Loss Calculation...")

    # Forward
    preds = model(inputs, pair_index)

    # Check output shape: (Batch, Seq_Len, 5)
    assert preds.shape == (
        config.batch_size,
        config.seq_len,
        config.num_targets,
    ), f"Prediction shape mismatch: {preds.shape}"

    # Loss
    criterion = MCRMSELoss()

    # Apply mask for loss calculation (simulating training step logic)
    active_mask = mask.bool()
    active_preds = preds[active_mask]
    active_targets = targets[active_mask]

    loss = criterion(active_preds, active_targets)

    print(f"  -> Forward pass successful.")
    print(f"  -> Calculated MCRMSE Loss: {loss.item():.6f}")

    assert not torch.isnan(loss), "Loss is NaN!"
    assert loss.item() >= 0, "Loss is negative!"

    # 5. Training Loop Simulation
    print("\n[Step 4] Simulating Training Loop (1 Epoch)...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    # Train one epoch
    avg_train_loss = train_one_epoch(
        model, train_loader, criterion, optimizer, config.device, config
    )
    print(f"  -> Epoch 1 Train Loss: {avg_train_loss:.6f}")

    # Validate
    print("[Step 5] Running Validation...")
    val_score = validate(model, val_loader, config.device)
    print(f"  -> Validation MCRMSE: {val_score:.6f}")

    # Save model (required for inference step in generate_submission)
    torch.save(model.state_dict(), config.model_save_path)
    print(f"  -> Model saved to {config.model_save_path}")

    # 6. Inference and Submission
    print("\n[Step 6] Generating Submission...")

    # Reload model to ensure saving/loading works
    model.load_state_dict(
        torch.load(config.model_save_path, map_location=config.device)
    )

    # Generate submission dataframe
    df_sub = generate_submission(model, test_loader, config.device, config)

    # Verify submission format
    # Total rows should be Num_Test_Samples * Seq_Len
    # In debug mode, test set is processed fully (240 samples), but let's check the loader size
    # The get_dataloaders function only slices train/val in debug mode, not test.
    # However, for this demo, we want to be sure.
    n_test_samples = len(test_loader.dataset)
    expected_rows = n_test_samples * config.seq_len

    assert (
        len(df_sub) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(df_sub)}"

    expected_cols = ["id_seqpos"] + config.target_cols
    assert (
        list(df_sub.columns) == expected_cols
    ), f"Submission columns mismatch. Expected {expected_cols}, got {list(df_sub.columns)}"

    print(f"  -> Submission DataFrame shape: {df_sub.shape}")
    print(f"  -> First 5 rows:\n{df_sub.head()}")

    # Save to CSV
    df_sub.to_csv(config.submission_path, index=False)
    print(f"  -> Submission saved to {config.submission_path}")

    print("\n==== Demo Completed Successfully ====")


if __name__ == "__main__":
    run_demo()
