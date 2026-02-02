import os
import shutil
import torch
import numpy as np
import pandas as pd
import torch.optim as optim

# Import library components
from library.config import Config, set_seed
from library.utils import mcrmse_metric
from library.dataset import get_dataloaders
from library.model import HybridResNetBiGRU
from library.loss import MaskedHuberLoss
from library.train import train_one_epoch, validate, generate_submission


def main():
    print("=== Starting RNA Degradation Library Demo ===")

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Speed and Testing
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment for debug run...")

    # Override Config parameters to run quickly on a small subset
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 32  # Small subset for demo
    Config.BATCH_SIZE = 8
    Config.EPOCHS = 1
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Set paths to a specific demo directory in working/
    demo_dir = os.path.join(Config.ROOT_DIR, "working", "demo_run")
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)

    Config.WORKING_DIR = demo_dir
    Config.CACHE_DIR = os.path.join(demo_dir, "cache")
    Config.MODEL_SAVE_PATH = os.path.join(demo_dir, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "submission.csv")

    # Re-run setup to create these directories
    Config.setup_workspace()

    # Set seed for reproducibility
    set_seed(Config.SEED)

    device = Config.DEVICE
    print(f"Running on device: {device}")

    # -------------------------------------------------------------------------
    # 2. Metric Verification
    # -------------------------------------------------------------------------
    print("\n[2] Verifying MCRMSE Metric...")

    # Create dummy ground truth and predictions
    # Shape: (Batch=2, Seq=3, Channels=2)
    y_true = np.array(
        [[[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]], [[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]]]
    )

    # Case 1: Perfect prediction
    y_pred_perfect = y_true.copy()
    score_perfect = mcrmse_metric(y_true, y_pred_perfect)
    assert (
        score_perfect == 0.0
    ), f"Expected 0.0 for perfect prediction, got {score_perfect}"
    print("   -> Perfect prediction check passed.")

    # Case 2: Constant error
    # Add 1.0 to channel 0, Add 0.0 to channel 1
    # MSE Ch0 = 1.0, RMSE Ch0 = 1.0
    # MSE Ch1 = 0.0, RMSE Ch1 = 0.0
    # MCRMSE = (1.0 + 0.0) / 2 = 0.5
    y_pred_offset = y_true.copy()
    y_pred_offset[:, :, 0] += 1.0

    score_offset = mcrmse_metric(y_true, y_pred_offset)
    assert np.isclose(score_offset, 0.5), f"Expected 0.5, got {score_offset}"
    print("   -> Offset prediction check passed.")

    # -------------------------------------------------------------------------
    # 3. Data Loading Verification
    # -------------------------------------------------------------------------
    print("\n[3] Loading DataLoaders...")

    # Force reload from source (parquet) to verify processing logic, then cache
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=False, batch_size=Config.BATCH_SIZE, debug=Config.DEBUG
    )

    # Fetch one batch
    batch = next(iter(train_loader))

    # Verify keys
    expected_keys = {
        "ids",
        "sequence",
        "structure",
        "predicted_loop_type",
        "targets",
        "mask",
    }
    assert (
        set(batch.keys()) == expected_keys
    ), f"Missing keys in batch. Found: {batch.keys()}"

    # Verify Shapes
    # Sequence length should be 107 (Config.SEQ_LEN)
    seq_shape = batch["sequence"].shape
    assert seq_shape == (
        Config.BATCH_SIZE,
        107,
    ), f"Expected sequence shape ({Config.BATCH_SIZE}, 107), got {seq_shape}"

    # Targets should be (Batch, 107, 5)
    target_shape = batch["targets"].shape
    assert target_shape == (
        Config.BATCH_SIZE,
        107,
        5,
    ), f"Expected target shape ({Config.BATCH_SIZE}, 107, 5), got {target_shape}"

    # Mask should be (Batch, 107)
    mask_shape = batch["mask"].shape
    assert mask_shape == (
        Config.BATCH_SIZE,
        107,
    ), f"Expected mask shape ({Config.BATCH_SIZE}, 107), got {mask_shape}"

    print(
        f"   -> Batch shapes verified. Sequence: {seq_shape}, Targets: {target_shape}"
    )

    # -------------------------------------------------------------------------
    # 4. Model Verification
    # -------------------------------------------------------------------------
    print("\n[4] Initializing Model and Running Forward Pass...")

    model = HybridResNetBiGRU().to(device)

    # Move inputs to device
    seq = batch["sequence"].to(device)
    struct = batch["structure"].to(device)
    loop = batch["predicted_loop_type"].to(device)

    # Forward pass
    preds = model(seq, struct, loop)

    # Verify output shape: (Batch, Seq_Len, Num_Targets)
    expected_out_shape = (Config.BATCH_SIZE, 107, 5)
    assert (
        preds.shape == expected_out_shape
    ), f"Model output shape mismatch. Expected {expected_out_shape}, got {preds.shape}"

    print(f"   -> Model forward pass successful. Output shape: {preds.shape}")

    # -------------------------------------------------------------------------
    # 5. Loss Function Verification
    # -------------------------------------------------------------------------
    print("\n[5] Verifying MaskedHuberLoss...")

    criterion = MaskedHuberLoss(delta=1.0)
    targets = batch["targets"].to(device)
    mask = batch["mask"].to(device)

    # Calculate loss
    loss = criterion(preds, targets, mask)

    # Verify loss is a scalar and has gradients
    assert loss.dim() == 0, "Loss should be a scalar"
    assert loss.requires_grad, "Loss should require gradients"
    assert loss.item() >= 0, "Loss should be non-negative"

    print(f"   -> Loss calculation successful. Value: {loss.item():.6f}")

    # -------------------------------------------------------------------------
    # 6. Training Loop Simulation
    # -------------------------------------------------------------------------
    print("\n[6] Simulating Training Epoch...")

    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    # Run one epoch of training
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    print(f"   -> Train Epoch Loss: {train_loss:.6f}")

    # Run validation
    val_loss, val_mcrmse = validate(model, val_loader, criterion, device)
    print(f"   -> Validation Loss: {val_loss:.6f} | MCRMSE: {val_mcrmse:.6f}")

    # Save this model as "best model" for the next step
    torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
    print(f"   -> Model saved to {Config.MODEL_SAVE_PATH}")

    # -------------------------------------------------------------------------
    # 7. Submission Generation Verification
    # -------------------------------------------------------------------------
    print("\n[7] Generating Submission for Test Set...")

    # Load the saved model
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))

    # Generate submission
    generate_submission(model, test_loader, device, Config.SUBMISSION_PATH)

    # Verify output file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created"

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"   -> Submission file loaded. Shape: {df_sub.shape}")

    # Check columns
    expected_cols = ["id_seqpos"] + Config.TARGET_COLS
    assert (
        list(df_sub.columns) == expected_cols
    ), f"Submission columns mismatch. Got {df_sub.columns}"

    # Check row count
    # We used a debug subset size of 32 for the test set
    # Each sample has 107 positions. Total rows = 32 * 107 = 3424
    expected_rows = Config.DEBUG_SUBSET_SIZE * Config.SEQ_LEN
    assert (
        len(df_sub) == expected_rows
    ), f"Expected {expected_rows} rows, got {len(df_sub)}"

    # Check IDs format (e.g., id_00b436dec_0)
    sample_id_seqpos = df_sub.iloc[0]["id_seqpos"]
    assert (
        "_0" in sample_id_seqpos or "_1" in sample_id_seqpos
    ), "id_seqpos format seems incorrect"

    print("   -> Submission format verified successfully.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
