import os
import shutil
import torch
import numpy as np
import pandas as pd
import torch.optim as optim

# Import from the provided library files
from library.config import Config
from library.utils import seed_all, MaskedMCRMSELoss, GlobalRMSETracker
from library.data import get_dataloaders
from library.model import DensePartnerAwareNet
from library.train import train_one_epoch, evaluate, generate_submission


def main():
    print("=== Starting Demonstration Script ===\n")

    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    # Create a separate directory for this demo to avoid overwriting existing work
    demo_dir = "./working/demo_execution"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    print(f"Working Directory: {demo_dir}")

    # Override Config paths and parameters for the demo
    Config.WORKING_DIR = demo_dir
    Config.CACHE_DIR = os.path.join(demo_dir, "data_cache")
    Config.MODEL_SAVE_PATH = os.path.join(demo_dir, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "submission.csv")

    # Set parameters for speed
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 4
    Config.DEBUG_SUBSET_SIZE = 50  # Only use 50 samples
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Set seed for reproducibility
    seed_all(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 2. Data Pipeline Verification
    # -------------------------------------------------------------------------
    print("\n[Step 1] Verifying Data Loading...")

    # Load data with debug=True to use the subset
    # load_cached_data=False forces processing from source CSVs
    train_loader, val_loader, test_loader = get_dataloaders(
        debug=True, load_cached_data=False
    )

    # Fetch a single batch
    inputs, partner_indices, targets = next(iter(train_loader))

    print(
        f"  Batch shapes -> Inputs: {inputs.shape}, Partners: {partner_indices.shape}, Targets: {targets.shape}"
    )

    # Assertions
    # Inputs: (Batch, Seq_Len, Input_Channels) -> (4, 107, 18)
    assert inputs.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LENGTH,
        Config.INPUT_CHANNELS,
    ), f"Input shape mismatch. Expected {(Config.BATCH_SIZE, Config.SEQ_LENGTH, Config.INPUT_CHANNELS)}, got {inputs.shape}"

    # Partner Indices: (Batch, Seq_Len) -> (4, 107)
    assert partner_indices.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LENGTH,
    ), "Partner indices shape mismatch."

    # Targets: (Batch, Seq_Len, Num_Targets) -> (4, 107, 5)
    assert targets.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LENGTH,
        Config.NUM_TARGETS,
    ), "Targets shape mismatch."

    print("  Data loading verified successfully.")

    # 3. Model Architecture Verification
    # -------------------------------------------------------------------------
    print("\n[Step 2] Verifying Model Architecture...")

    # Instantiate model with reduced complexity for demo speed
    model = DensePartnerAwareNet(
        input_channels=Config.INPUT_CHANNELS,
        tcn_channels=16,  # Reduced from 64
        tcn_layers=2,  # Reduced from 6
        kernel_size=3,
        dropout=0.1,
        latent_dim=16,  # Reduced from 32
        gru_hidden=16,  # Reduced
        num_targets=Config.NUM_TARGETS,
    ).to(device)

    # Move batch to device
    inputs = inputs.to(device)
    partner_indices = partner_indices.to(device)
    targets = targets.to(device)

    # Forward pass
    preds = model(inputs, partner_indices)

    print(f"  Output shape: {preds.shape}")

    # Assertions
    assert preds.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LENGTH,
        Config.NUM_TARGETS,
    ), "Model output shape mismatch."

    # Check for NaN
    assert not torch.isnan(preds).any(), "Model output contains NaNs."

    print("  Model forward pass verified successfully.")

    # 4. Loss Function & Metric Verification
    # -------------------------------------------------------------------------
    print("\n[Step 3] Verifying Loss and Metrics...")

    # Loss
    criterion = MaskedMCRMSELoss(scored_indices=Config.SCORED_INDICES)
    loss = criterion(preds, targets)

    print(f"  Initial Loss: {loss.item():.6f}")
    assert loss.item() >= 0, "Loss cannot be negative."

    # Metric Tracker (Manual Test)
    tracker = GlobalRMSETracker(scored_indices=Config.SCORED_INDICES)

    # Create dummy perfect predictions
    dummy_preds = np.zeros((10, 107, 5))
    dummy_targets = np.zeros((10, 107, 5))

    # Introduce a known error in one scored column
    # Scored indices are [0, 1, 3]
    # Let's add error of 1.0 to index 0
    dummy_preds[:, :, 0] = 1.0
    dummy_targets[:, :, 0] = 0.0

    tracker.update(dummy_preds, dummy_targets)
    metric_val = tracker.compute()

    # Expected:
    # Col 0 RMSE = sqrt(1^2) = 1.0
    # Col 1 RMSE = 0.0
    # Col 3 RMSE = 0.0
    # MCRMSE = (1.0 + 0.0 + 0.0) / 3 = 0.3333
    print(f"  Tracker Manual Test Result: {metric_val:.4f}")
    assert np.isclose(
        metric_val, 1.0 / 3.0, atol=1e-4
    ), f"Tracker failed manual test. Expected 0.3333, got {metric_val}"

    print("  Loss and Tracker verified successfully.")

    # 5. Training Loop Integration
    # -------------------------------------------------------------------------
    print("\n[Step 4] Running Training Loop (1 Epoch)...")

    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    # Run training for 1 epoch
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    print(f"  Train Loss: {train_loss:.6f}")

    # Run evaluation
    val_mcrmse = evaluate(model, val_loader, device)
    print(f"  Val MCRMSE: {val_mcrmse:.6f}")

    # Save the model manually for the submission step
    torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
    print("  Training loop execution successful.")

    # 6. Inference and Submission
    # -------------------------------------------------------------------------
    print("\n[Step 5] Generating Submission...")

    # Generate submission
    generate_submission(model, test_loader, device, Config.SUBMISSION_PATH)

    # Verify file creation
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    # Verify content format
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"  Submission shape: {sub_df.shape}")
    print(f"  Columns: {sub_df.columns.tolist()}")

    # Expected rows: Number of test samples * Seq Length
    # In debug mode, we limit reading source CSVs, but test.json has 240 lines.
    # get_dataloaders with debug=True limits the dataframe via head(DEBUG_SUBSET_SIZE).
    # So we expect DEBUG_SUBSET_SIZE * 107 rows.
    expected_rows = Config.DEBUG_SUBSET_SIZE * Config.SEQ_LENGTH
    assert (
        len(sub_df) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(sub_df)}"

    # Expected columns: id_seqpos + 5 targets
    expected_cols = ["id_seqpos"] + Config.ALL_TARGET_COLS
    assert list(sub_df.columns) == expected_cols, "Submission columns mismatch."

    print("  Submission generated and verified successfully.")
    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
