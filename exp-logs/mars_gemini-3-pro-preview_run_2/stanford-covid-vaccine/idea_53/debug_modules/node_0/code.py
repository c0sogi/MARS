import os
import torch
import numpy as np
import pandas as pd
import shutil
import warnings

# Import from the provided library
from library.config import Config
from library.utils import set_seed, mcrmse_loss
from library.model import SS_DFRN, process_data, RNADataset
from library.train import train_one_epoch, validate

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    print("==== Starting RNA Degradation Prediction Demo ====\n")

    # -------------------------------------------------------------------------
    # 1. Configuration Setup
    # -------------------------------------------------------------------------
    # We override the default Config to use a temporary working directory
    # and a small subset of data for speed.
    print("[1] Configuring environment for fast execution...")

    Config.WORKING_DIR = "./working/demo_execution"
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Update paths to store cache and outputs in the demo directory
    Config.TRAIN_CACHE = os.path.join(Config.WORKING_DIR, "train_cache.npz")
    Config.VAL_CACHE = os.path.join(Config.WORKING_DIR, "val_cache.npz")
    Config.TEST_CACHE = os.path.join(Config.WORKING_DIR, "test_cache.npz")
    Config.BEST_MODEL_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "demo_submission.csv")

    # Optimization settings for demo
    Config.DEBUG_SUBSET_SIZE = 50  # Use only 50 samples
    Config.BATCH_SIZE = 8  # Small batch size
    Config.EPOCHS = 2  # Only 2 epochs
    Config.DEVICE = (
        "cpu"  # Force CPU for simple demo stability (or use cuda if available)
    )
    if torch.cuda.is_available():
        Config.DEVICE = "cuda"

    print(f"    Working Directory: {Config.WORKING_DIR}")
    print(f"    Device: {Config.DEVICE}")
    print(f"    Subset Size: {Config.DEBUG_SUBSET_SIZE}")

    set_seed(Config.SEED)

    # -------------------------------------------------------------------------
    # 2. Data Processing
    # -------------------------------------------------------------------------
    print("\n[2] Processing Data...")

    # Process Training Data
    # This will read the CSV, process features, and save to Config.TRAIN_CACHE
    train_features, train_pidx, train_targets = process_data(
        Config.TRAIN_CSV, Config.TRAIN_CACHE, load_cached_data=False, is_test=False
    )

    # Assertions to verify data shape
    assert len(train_features) == Config.DEBUG_SUBSET_SIZE, "Train subset size mismatch"
    assert (
        train_features.shape[1] == 18
    ), f"Expected 18 input channels, got {train_features.shape[1]}"
    assert (
        train_features.shape[2] == 107
    ), f"Expected sequence length 107, got {train_features.shape[2]}"
    assert train_targets.shape[2] == 5, "Expected 5 target columns"
    print(f"    Train data processed. Shape: {train_features.shape}")

    # Process Validation Data
    val_features, val_pidx, val_targets = process_data(
        Config.VAL_CSV, Config.VAL_CACHE, load_cached_data=False, is_test=False
    )
    print(f"    Val data processed. Shape: {val_features.shape}")

    # Create Datasets and Loaders
    train_dataset = RNADataset(train_features, train_pidx, train_targets)
    val_dataset = RNADataset(val_features, val_pidx, val_targets)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False
    )

    # -------------------------------------------------------------------------
    # 3. Model Initialization and Logic Verification
    # -------------------------------------------------------------------------
    print("\n[3] Initializing SS-DFRN Model...")

    model = SS_DFRN().to(Config.DEVICE)

    # Verify Model Output Shape with a dummy batch
    dummy_x, dummy_pidx, dummy_y = next(iter(train_loader))
    dummy_x, dummy_pidx = dummy_x.to(Config.DEVICE), dummy_pidx.to(Config.DEVICE)

    # Pass 1: No Feedback
    with torch.no_grad():
        out_pass1 = model(dummy_x, dummy_pidx, feedback=None)

    assert out_pass1.shape == (
        dummy_x.shape[0],
        107,
        5,
    ), f"Model output shape mismatch. Expected ({dummy_x.shape[0]}, 107, 5), got {out_pass1.shape}"

    # Pass 2: With Feedback
    with torch.no_grad():
        out_pass2 = model(dummy_x, dummy_pidx, feedback=out_pass1)

    assert out_pass2.shape == out_pass1.shape, "Pass 2 output shape mismatch"

    print("    Model forward pass successful (Pass 1 & Pass 2).")

    # -------------------------------------------------------------------------
    # 4. Training Loop Demonstration
    # -------------------------------------------------------------------------
    print("\n[4] Running Training Loop...")

    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, Config.DEVICE)

        # Validate
        val_loss = validate(model, val_loader, Config.DEVICE)

        print(
            f"    Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}"
        )

        # Check for NaN
        if np.isnan(train_loss) or np.isnan(val_loss):
            raise ValueError("Loss is NaN, training failed.")

    # Save the model
    torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
    print(f"    Model saved to {Config.BEST_MODEL_PATH}")

    # -------------------------------------------------------------------------
    # 5. Inference and Submission
    # -------------------------------------------------------------------------
    print("\n[5] Generating Predictions on Test Set...")

    # Process Test Data
    test_features, test_pidx, test_ids = process_data(
        Config.TEST_CSV, Config.TEST_CACHE, load_cached_data=False, is_test=True
    )

    test_dataset = RNADataset(test_features, test_pidx)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False
    )

    model.eval()
    all_preds = []

    with torch.no_grad():
        for x, p_idx in test_loader:
            x, p_idx = x.to(Config.DEVICE), p_idx.to(Config.DEVICE)

            # Pass 1
            pred1 = model(x, p_idx, feedback=None)
            # Pass 2 (Final)
            pred2 = model(x, p_idx, feedback=pred1)

            all_preds.append(pred2.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    print(f"    Predictions generated. Shape: {all_preds.shape}")

    # Format Submission
    print("    Formatting submission file...")
    submission_rows = []
    for i, sample_id in enumerate(test_ids):
        seq_len = all_preds.shape[1]
        for pos in range(seq_len):
            row_id = f"{sample_id}_{pos}"
            vals = all_preds[i, pos, :]
            row = [row_id] + vals.tolist()
            submission_rows.append(row)

    columns = ["id_seqpos"] + Config.TARGET_COLS
    sub_df = pd.DataFrame(submission_rows, columns=columns)

    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not created"
    loaded_sub = pd.read_csv(Config.SUBMISSION_PATH)

    # Expected rows: Number of test samples * 107
    expected_rows = len(test_ids) * 107
    assert (
        len(loaded_sub) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(loaded_sub)}"

    print(f"    Submission saved to {Config.SUBMISSION_PATH}")
    print(f"    First 5 rows:\n{loaded_sub.head().to_string()}")

    print("\n==== Demo Completed Successfully ====")
