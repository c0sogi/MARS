import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import shutil

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, calculate_mcrmse
from library.data import get_dataloaders, get_structure_adj, one_hot
from library.model import DeepBiGRUNet, ChannelGatedInteraction
from library.train import train_one_epoch, evaluate, generate_submission


def main():
    print("Starting demonstration of RNA Degradation Prediction pipeline...")

    # =========================================================================
    # 1. Configuration Override for Speed
    # =========================================================================
    print("\n[1] Overriding Config for rapid demonstration...")
    # Enable debug mode to use a small subset (100 samples)
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 50

    # Reduce model complexity for speed
    Config.HIDDEN_DIM = 64
    Config.NUM_LAYERS = 2
    Config.CNN_FILTERS = 32
    Config.BATCH_SIZE = 8
    Config.EPOCHS = 1

    # Set a specific working directory for this demo
    Config.PROJECT_NAME = "demo_execution"
    Config.WORKING_DIR = os.path.join("./working", Config.PROJECT_NAME)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Update paths dependent on WORKING_DIR
    Config.TRAIN_CACHE_PATH = os.path.join(Config.WORKING_DIR, "train_cache.npz")
    Config.VAL_CACHE_PATH = os.path.join(Config.WORKING_DIR, "val_cache.npz")
    Config.TEST_CACHE_PATH = os.path.join(Config.WORKING_DIR, "test_cache.npz")
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    seed_everything(Config.SEED)
    device = torch.device(
        "cpu"
    )  # Use CPU for this lightweight demo to avoid GPU init overhead if busy
    print(f"Config updated. Device: {device}")

    # =========================================================================
    # 2. Unit Testing Utility Functions
    # =========================================================================
    print("\n[2] Verifying utility functions...")

    # Test get_structure_adj
    # Structure: ((..)) -> Indices: 0-5, 1-4. 2,3 unpaired.
    dummy_struct = "((..))"
    indices, mask = get_structure_adj(dummy_struct)

    expected_indices = np.array([5, 4, 0, 0, 1, 0])  # Unpaired default to 0
    expected_mask = np.array([1.0, 1.0, 0.0, 0.0, 1.0, 1.0])

    np.testing.assert_array_equal(
        indices, expected_indices, err_msg="Structure indices mismatch"
    )
    np.testing.assert_array_equal(
        mask, expected_mask, err_msg="Structure mask mismatch"
    )
    print("  - get_structure_adj: Passed")

    # Test calculate_mcrmse
    # Create dummy preds and targets. Shape (B, SeqLen, 5)
    # We only score the first 68 (Config.SEQ_SCORED).
    # Let's make preds exactly 1.0 off from targets in the scored region.
    B_test = 2
    preds_np = np.zeros((B_test, Config.SEQ_LENGTH, 5))
    targets_np = np.zeros((B_test, Config.SEQ_LENGTH, 5))

    # Set scored region difference to 1.0
    preds_np[:, : Config.SEQ_SCORED, :] = 1.0
    targets_np[:, : Config.SEQ_SCORED, :] = 0.0

    # RMSE of 1.0 is 1.0. Mean of RMSEs is 1.0.
    score = calculate_mcrmse(preds_np, targets_np)
    assert (
        abs(score - 1.0) < 1e-6
    ), f"MCRMSE calculation failed. Expected 1.0, got {score}"
    print("  - calculate_mcrmse: Passed")

    # =========================================================================
    # 3. Data Loading & Processing
    # =========================================================================
    print("\n[3] Loading DataLoaders (this triggers data processing)...")
    # Note: We use load_cached_data=False to force processing logic execution for the demo
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    print(f"  - Train batches: {len(train_loader)}")
    print(f"  - Val batches: {len(val_loader)}")
    print(f"  - Test batches: {len(test_loader)}")

    # Verify Batch Structure
    batch = next(iter(train_loader))
    features = batch["features"]
    pair_indices = batch["pair_indices"]
    pair_masks = batch["pair_masks"]
    targets = batch["targets"]
    ids = batch["id"]

    # Check shapes
    assert features.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LENGTH,
        Config.INPUT_DIM,
    ), f"Feature shape mismatch: {features.shape}"
    assert pair_indices.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LENGTH,
    ), f"Pair indices shape mismatch: {pair_indices.shape}"
    assert pair_masks.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LENGTH,
    ), f"Pair masks shape mismatch: {pair_masks.shape}"
    assert targets.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LENGTH,
        Config.NUM_TARGETS,
    ), f"Targets shape mismatch: {targets.shape}"

    print("  - Batch shapes verified.")

    # =========================================================================
    # 4. Model Instantiation & Component Check
    # =========================================================================
    print("\n[4] Initializing Model...")

    # Test Interaction Module specifically
    interaction = ChannelGatedInteraction(dim=Config.HIDDEN_DIM)
    dummy_x = torch.randn(Config.BATCH_SIZE, Config.SEQ_LENGTH, Config.HIDDEN_DIM)
    dummy_out = interaction(dummy_x, pair_indices, pair_masks)
    assert dummy_out.shape == dummy_x.shape, "Interaction module output shape mismatch"
    print("  - ChannelGatedInteraction: Passed")

    # Full Model
    model = DeepBiGRUNet().to(device)
    print("  - DeepBiGRUNet instantiated.")

    # Forward pass check
    features = features.to(device)
    pair_indices = pair_indices.to(device)
    pair_masks = pair_masks.to(device)

    with torch.no_grad():
        output = model(features, pair_indices, pair_masks)

    assert output.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LENGTH,
        Config.NUM_TARGETS,
    ), f"Model output shape mismatch: {output.shape}"
    print("  - Forward pass successful.")

    # =========================================================================
    # 5. Training Loop Simulation
    # =========================================================================
    print("\n[5] Running Training Loop (1 Epoch)...")

    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    # Train
    train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch=0)
    print(f"  - Train Loss: {train_loss:.4f}")
    assert not np.isnan(train_loss), "Training loss is NaN"

    # Evaluate
    val_score = evaluate(model, val_loader, device)
    print(f"  - Validation MCRMSE: {val_score:.4f}")
    assert val_score >= 0, "Validation score is negative"

    # Save model for submission generation
    torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
    print("  - Model saved.")

    # =========================================================================
    # 6. Submission Generation
    # =========================================================================
    print("\n[6] Generating Submission...")

    generate_submission(model, test_loader, device, Config.SUBMISSION_PATH)

    # Verify Submission File
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found"

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"  - Submission shape: {sub_df.shape}")
    print(f"  - Columns: {list(sub_df.columns)}")

    # Expected rows: Num_Test_Samples (subset) * Seq_Length
    # In debug mode with subset=50, test set might be smaller or equal depending on slice logic.
    # The slice logic in data.py slices inputs/ids.
    # subset size is 50.
    expected_rows = 50 * Config.SEQ_LENGTH
    assert (
        len(sub_df) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(sub_df)}"

    # Check columns
    expected_cols = ["id_seqpos"] + Config.TARGET_COLS
    assert list(sub_df.columns) == expected_cols, "Submission columns mismatch"

    print("  - Submission file verified.")
    print("\nDemonstration complete successfully.")


if __name__ == "__main__":
    main()
