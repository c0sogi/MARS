import os
import torch
import numpy as np
import pandas as pd
import torch.optim as optim

# Import provided library components
from library.config import Config
from library.utils import set_seed, MCRMSELoss
from library.data import load_data, get_dataloaders
from library.model import HighCapacityRNAnet
from library.engine import train_one_epoch, validate, generate_submission

if __name__ == "__main__":
    print("Starting High-Capacity RNA Degradation Prediction Demo...")

    # ==========================================
    # 1. Configuration for Fast Demonstration
    # ==========================================
    print("\n[1] Configuring environment for speed...")

    # Modify Config for a quick debug run
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 20  # Use only 20 samples per split
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.WORKING_DIR = "./working/demo_execution"

    # Update paths to avoid conflicts with real training runs
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    Config.TRAIN_CACHE = os.path.join(Config.WORKING_DIR, "train_cache.npy")
    Config.VAL_CACHE = os.path.join(Config.WORKING_DIR, "val_cache.npy")
    Config.TEST_CACHE = os.path.join(Config.WORKING_DIR, "test_cache.npy")
    Config.BEST_MODEL_PATH = os.path.join(Config.WORKING_DIR, "demo_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission_demo.csv")

    # Set seed for reproducibility
    set_seed(42)
    device = Config.DEVICE
    print(f"    Device: {device}")
    print(f"    Debug Mode: {Config.DEBUG}")
    print(f"    Subset Size: {Config.DEBUG_SUBSET_SIZE}")

    # ==========================================
    # 2. Data Loading & Verification
    # ==========================================
    print("\n[2] Loading and verifying data...")

    # Load dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=0,  # Use 0 workers for simple script execution
    )

    # Fetch one batch to verify structure
    batch = next(iter(train_loader))
    inputs = batch["inputs"]
    pair_indices = batch["pair_indices"]
    pair_masks = batch["pair_masks"]
    targets = batch["targets"]
    ids = batch["ids"]

    # Assertions
    # Input shape: (Batch, Seq_Len, Channels=14)
    assert inputs.shape == (
        Config.BATCH_SIZE,
        107,
        14,
    ), f"Expected input shape ({Config.BATCH_SIZE}, 107, 14), got {inputs.shape}"

    # Pair indices shape: (Batch, Seq_Len)
    assert pair_indices.shape == (
        Config.BATCH_SIZE,
        107,
    ), f"Expected pair_indices shape ({Config.BATCH_SIZE}, 107), got {pair_indices.shape}"

    # Target shape: (Batch, Seq_Len, Targets=5)
    assert targets.shape == (
        Config.BATCH_SIZE,
        107,
        5,
    ), f"Expected target shape ({Config.BATCH_SIZE}, 107, 5), got {targets.shape}"

    print("    Data shapes verified successfully.")
    print(f"    Batch inputs: {inputs.shape}")
    print(f"    Batch targets: {targets.shape}")

    # ==========================================
    # 3. Model Initialization & Forward Pass
    # ==========================================
    print("\n[3] Initializing model and running forward pass...")

    model = HighCapacityRNAnet().to(device)

    # Move batch to device
    inputs = inputs.to(device)
    pair_indices = pair_indices.to(device)
    pair_masks = pair_masks.to(device)

    # Forward pass
    outputs = model(inputs, pair_indices, pair_masks)

    # Check output shape
    assert outputs.shape == (
        Config.BATCH_SIZE,
        107,
        5,
    ), f"Expected output shape ({Config.BATCH_SIZE}, 107, 5), got {outputs.shape}"

    print("    Forward pass successful.")
    print(f"    Output shape: {outputs.shape}")

    # ==========================================
    # 4. Loss Function Verification
    # ==========================================
    print("\n[4] Verifying MCRMSE Loss logic...")

    criterion = MCRMSELoss()

    # Create synthetic data
    # Preds: All ones
    # Targets: All zeros
    # Scored Length: 68
    # Error should be sqrt(mean((1-0)^2)) = 1.0 per column, average is 1.0

    dummy_preds = torch.ones((2, 107, 5), dtype=torch.float32)
    dummy_targets = torch.zeros((2, 107, 5), dtype=torch.float32)

    # Calculate loss (scoring_only=False means all 5 columns)
    loss_val = criterion(dummy_preds, dummy_targets, scoring_only=False)

    # Check value
    assert (
        abs(loss_val.item() - 1.0) < 1e-5
    ), f"Expected loss 1.0, got {loss_val.item()}"

    # Check scoring_only=True (subset of columns)
    # If we modify one non-scored column in preds, the score should not change
    # Scored targets indices: [0, 1, 3] (reactivity, deg_Mg_pH10, deg_Mg_50C)
    # Non-scored indices: [2, 4]

    # Change a non-scored column to 100 (huge error)
    dummy_preds_mod = dummy_preds.clone()
    dummy_preds_mod[:, :, 2] = 100.0

    loss_scored = criterion(dummy_preds_mod, dummy_targets, scoring_only=True)

    # The loss should still be 1.0 because we ignore column 2
    assert (
        abs(loss_scored.item() - 1.0) < 1e-5
    ), f"Expected scored loss 1.0, got {loss_scored.item()}"

    print("    Loss function logic verified.")

    # ==========================================
    # 5. Training Loop Demonstration
    # ==========================================
    print("\n[5] Simulating training loop...")

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.NUM_EPOCHS)

    # Train one epoch
    train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch=0)
    print(f"    Epoch 0 Train Loss: {train_loss:.4f}")

    assert train_loss > 0, "Train loss should be positive."
    assert np.isfinite(train_loss), "Train loss should be finite."

    # Validate
    val_score = validate(model, val_loader, device)
    print(f"    Validation MCRMSE: {val_score:.4f}")

    assert val_score >= 0, "Validation score should be non-negative."

    # Save dummy model for inference step
    torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
    print("    Model saved.")

    # ==========================================
    # 6. Inference & Submission
    # ==========================================
    print("\n[6] Generating submission...")

    generate_submission(
        model, test_loader, device, submission_path=Config.SUBMISSION_PATH
    )

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"    Submission shape: {sub_df.shape}")
    print(f"    Columns: {list(sub_df.columns)}")

    # Expected rows: Num_Test_Samples (20 in debug) * Seq_Len (107) = 2140
    expected_rows = Config.DEBUG_SUBSET_SIZE * 107
    assert (
        len(sub_df) == expected_rows
    ), f"Expected {expected_rows} rows, got {len(sub_df)}"

    # Expected columns
    expected_cols = ["id_seqpos"] + Config.TARGET_COLS
    assert (
        list(sub_df.columns) == expected_cols
    ), f"Expected columns {expected_cols}, got {list(sub_df.columns)}"

    print("    Submission file verified.")
    print("\nDemo execution completed successfully.")
