import os
import torch
import pandas as pd
import numpy as np
import shutil

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, compute_mcrmse
from library.data import get_dataloaders
from library.model import RNAModel
from library.loss import MCRMSELoss
from library.train import train_one_epoch, validate, inference, generate_submission

if __name__ == "__main__":
    print("Starting Demo Execution...")

    # ==========================================
    # 1. Configuration Override for Speed
    # ==========================================
    print("\n[1] Configuring environment for fast demonstration...")

    # Modify Config for a quick run
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 50  # Use only 50 samples
    Config.BATCH_SIZE = 4  # Small batch size
    Config.MAX_EPOCHS = 1  # Only 1 epoch
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data

    # Update working directory for this demo
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission_demo.csv")

    # Ensure directories exist (Config.setup_directories only created the original ones)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    Device: {device}")
    print(f"    Working Directory: {Config.WORKING_DIR}")

    # ==========================================
    # 2. Data Loading & Verification
    # ==========================================
    print("\n[2] Loading DataLoaders...")

    # Force reload to ensure we process the debug subset
    # Note: We pass load_cached_data=False to ensure we generate the small debug cache now
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    print(f"    Train Batches: {len(train_loader)}")
    print(f"    Val Batches:   {len(val_loader)}")
    print(f"    Test Batches:  {len(test_loader)}")

    # Fetch one batch to verify structure
    batch = next(iter(train_loader))
    features = batch["features"].to(device)
    pair_indices = batch["pair_indices"].to(device)
    pair_masks = batch["pair_masks"].to(device)
    targets = batch["targets"].to(device)

    print("    Batch Shapes Verification:")
    print(
        f"    Features: {features.shape} (Expected: [{Config.BATCH_SIZE}, {Config.SEQ_LEN}, 14])"
    )
    print(
        f"    Targets:  {targets.shape}  (Expected: [{Config.BATCH_SIZE}, {Config.SEQ_LEN}, 5])"
    )

    # Assertions
    assert features.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        14,
    ), "Feature shape mismatch"
    assert targets.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        5,
    ), "Target shape mismatch"
    assert pair_indices.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
    ), "Pair indices shape mismatch"
    assert pair_masks.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
    ), "Pair masks shape mismatch"

    # ==========================================
    # 3. Model Initialization & Forward Pass
    # ==========================================
    print("\n[3] Initializing Model and running Forward Pass...")

    model = RNAModel().to(device)

    # Run forward pass
    preds = model(features, pair_indices, pair_masks)

    print(f"    Predictions Shape: {preds.shape}")

    # Assertions
    assert preds.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        5,
    ), "Prediction shape mismatch"
    assert not torch.isnan(preds).any(), "Model produced NaNs"

    # ==========================================
    # 4. Loss Computation
    # ==========================================
    print("\n[4] Computing Loss...")

    criterion = MCRMSELoss()
    loss = criterion(preds, targets)

    print(f"    Loss Value: {loss.item():.6f}")

    # Assertions
    assert loss.item() >= 0, "Loss should be non-negative"
    assert not torch.isnan(loss), "Loss is NaN"

    # ==========================================
    # 5. Training Loop Simulation
    # ==========================================
    print("\n[5] Simulating Training Loop (1 Epoch)...")

    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    # Train for one epoch
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    print(f"    Train Loss: {train_loss:.6f}")

    # Validate
    val_score = validate(model, val_loader, device)
    print(f"    Validation Score (MCRMSE): {val_score:.6f}")

    # Save model (simulating the checkpointing)
    torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
    assert os.path.exists(Config.MODEL_SAVE_PATH), "Model file was not saved"

    # ==========================================
    # 6. Inference & Submission
    # ==========================================
    print("\n[6] Running Inference and Generating Submission...")

    # Load model (to verify saving/loading works)
    model.load_state_dict(
        torch.load(Config.MODEL_SAVE_PATH, map_location=device, weights_only=True)
    )

    # Inference
    test_preds, test_ids = inference(model, test_loader, device)
    print(f"    Test Predictions Shape: {test_preds.shape}")

    # Generate Submission
    generate_submission(test_preds, test_ids, Config.SUBMISSION_PATH)

    # Verify Submission File
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found"

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"    Submission Rows: {len(df_sub)}")
    print(f"    Submission Columns: {list(df_sub.columns)}")

    # Check expected row count: num_test_samples * seq_len
    # In debug mode, test set is also truncated to DEBUG_SUBSET_SIZE (or smaller if file is small)
    expected_rows = len(test_ids) * Config.SEQ_LEN
    assert (
        len(df_sub) == expected_rows
    ), f"Expected {expected_rows} rows, got {len(df_sub)}"

    # Check columns
    expected_cols = ["id_seqpos"] + Config.TARGET_COLS
    assert list(df_sub.columns) == expected_cols, "Submission columns mismatch"

    # Check content validity
    assert df_sub["reactivity"].notna().all(), "Submission contains NaNs"

    print("\nDemo Execution Completed Successfully!")
