import os
import sys
import torch
import pandas as pd
import numpy as np

# Ensure the library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything
from library.dataset import get_dataloader
from library.model import UncertaintyAwareBiGRU
from library.loss import UncertaintyAwareMSELoss
from library.engine import train_one_epoch, validate, generate_submission


def main():
    # --------------------------------------------------------------------------
    # 1. Setup & Configuration
    # --------------------------------------------------------------------------
    print(">>> Setting up configuration and seeding...")
    seed_everything(42)

    # Instantiate config and override for a quick demo run
    config = Config()
    config.epochs = 1  # Run only 1 epoch for speed
    config.batch_size = 32
    config.cache_dir = "./working/demo_run"
    config.submission_path = "./working/demo_run/submission.csv"

    # Ensure working directory exists
    os.makedirs(config.cache_dir, exist_ok=True)

    print(f"Device: {config.device}")
    print(f"Cache Directory: {config.cache_dir}")

    # --------------------------------------------------------------------------
    # 2. Data Loading & Verification
    # --------------------------------------------------------------------------
    print("\n>>> Loading DataLoaders...")
    # Load training data
    train_loader = get_dataloader(mode="train", config=config, shuffle=True)
    val_loader = get_dataloader(mode="val", config=config, shuffle=False)

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Verify Batch Structure
    print("Verifying batch structure...")
    batch = next(iter(train_loader))

    # Check keys
    expected_keys = ["sequence", "loop", "pair_dist", "target", "error"]
    for key in expected_keys:
        assert key in batch, f"Missing key {key} in batch"

    # Check shapes
    # Sequence: [Batch, SeqLen]
    assert batch["sequence"].shape == (
        config.batch_size,
        config.seq_len,
    ), f"Incorrect sequence shape: {batch['sequence'].shape}"
    # Target: [Batch, SeqLen, 3]
    assert batch["target"].shape == (
        config.batch_size,
        config.seq_len,
        3,
    ), f"Incorrect target shape: {batch['target'].shape}"

    print("Batch structure verification passed.")

    # --------------------------------------------------------------------------
    # 3. Model Initialization & Forward Pass
    # --------------------------------------------------------------------------
    print("\n>>> Initializing Model...")
    model = UncertaintyAwareBiGRU(config).to(config.device)

    print("Running forward pass on verification batch...")
    seq = batch["sequence"].to(config.device)
    loop = batch["loop"].to(config.device)
    dist = batch["pair_dist"].to(config.device)

    # Forward pass
    pred_val, pred_err = model(seq, loop, dist)

    # Verify Output Shapes
    # Output: [Batch, SeqLen, 3]
    assert pred_val.shape == (
        config.batch_size,
        config.seq_len,
        3,
    ), f"Incorrect prediction shape: {pred_val.shape}"
    assert pred_err.shape == (
        config.batch_size,
        config.seq_len,
        3,
    ), f"Incorrect uncertainty shape: {pred_err.shape}"

    print("Model forward pass verification passed.")

    # --------------------------------------------------------------------------
    # 4. Loss Calculation
    # --------------------------------------------------------------------------
    print("\n>>> Calculating Loss...")
    loss_fn = UncertaintyAwareMSELoss(lambda_uncertainty=1.0)

    target_val = batch["target"].to(config.device)
    target_err = batch["error"].to(config.device)

    # Create mask for valid positions (first 68)
    mask = torch.zeros(config.seq_len, device=config.device)
    mask[: config.pred_len] = 1.0
    batch_mask = mask.unsqueeze(0).expand(config.batch_size, -1)

    # Calculate loss
    loss = loss_fn(pred_val, pred_err, target_val, target_err, batch_mask)

    print(f"Calculated Loss: {loss.item():.6f}")

    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() >= 0, "Loss is negative"

    print("Loss calculation verification passed.")

    # --------------------------------------------------------------------------
    # 5. Training & Validation Loop
    # --------------------------------------------------------------------------
    print("\n>>> Starting Training (1 Epoch)...")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    # Train for one epoch
    train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, config)
    print(f"Epoch 1 Train Loss: {train_loss:.6f}")

    # Validate
    print("Validating...")
    val_mcrmse = validate(model, val_loader, config)
    print(f"Validation MCRMSE: {val_mcrmse:.6f}")

    assert isinstance(
        val_mcrmse, (float, np.floating)
    ), "Validation score is not a float"

    # Save model weights for submission generation step
    model_path = os.path.join(config.cache_dir, "best_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # --------------------------------------------------------------------------
    # 6. Submission Generation
    # --------------------------------------------------------------------------
    print("\n>>> Generating Submission...")

    # Reload model to ensure state consistency (simulating inference phase)
    model.load_state_dict(torch.load(model_path, map_location=config.device))

    # Generate submission
    generate_submission(model, config)

    # Verify Submission File
    assert os.path.exists(config.submission_path), "Submission file was not created"

    sub_df = pd.read_csv(config.submission_path)
    print(f"Submission file loaded. Shape: {sub_df.shape}")

    # Check Columns
    expected_cols = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]
    assert (
        list(sub_df.columns) == expected_cols
    ), f"Columns mismatch. Found: {sub_df.columns}"

    # Check Content
    # Test set has 240 samples, sequence length 107. Total rows = 240 * 107 = 25680
    assert len(sub_df) == 25680, f"Expected 25680 rows, found {len(sub_df)}"

    print("Submission verification passed.")
    print("\n>>> Demo Completed Successfully.")


if __name__ == "__main__":
    main()
