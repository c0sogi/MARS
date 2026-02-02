import os
import sys
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import shutil

# Import library modules
# We import Config first to modify it before other modules use it
from library.config import Config

# ==========================================
# 1. Configuration & Setup
# ==========================================
# Modify Config for a fast demonstration run
print("Configuring demonstration parameters...")
Config.debug = True  # Use a small subset of data (100 samples)
Config.debug_subset_size = 50
Config.epochs = 2
Config.batch_size = 8
Config.working_dir = "./working/demo_execution"
Config.num_workers = 0  # Avoid multiprocessing overhead for small demo

# Since paths are defined at class level, we must update them manually
# after changing working_dir to ensure files go to the new location.
os.makedirs(Config.working_dir, exist_ok=True)
Config.train_cache = os.path.join(Config.working_dir, "cache", "train_data.npz")
Config.val_cache = os.path.join(Config.working_dir, "cache", "val_data.npz")
Config.test_cache = os.path.join(Config.working_dir, "cache", "test_data.npz")
Config.best_model_path = os.path.join(Config.working_dir, "best_model.pth")
Config.submission_path = os.path.join(Config.working_dir, "submission_demo.csv")

# Create cache directory
os.makedirs(os.path.dirname(Config.train_cache), exist_ok=True)

# Import remaining modules after config updates
from library.utils import set_seed
from library.data import get_loaders
from library.model import HCTADPBiGRU
from library.loss import MCRMSELoss
from library.train import train_one_epoch, validate, generate_submission


def run_demo():
    # Set seed for reproducibility
    set_seed(Config.seed)
    device = torch.device(Config.device)
    print(f"Running on device: {device}")

    # ==========================================
    # 2. Data Loading & Verification
    # ==========================================
    print("\n--- Loading Data ---")
    # Force reload to ensure we use the debug subset
    train_loader, val_loader, test_loader = get_loaders(load_cached_data=False)

    # Verify Data Shapes
    print("Verifying DataLoader shapes...")
    # Fetch one batch
    feat, p_idx, p_mask, tgt, ids = next(iter(train_loader))

    # Expected shapes based on Config
    # Features: (Batch, Seq_Len=107, Input_Dim=14)
    assert feat.shape == (
        Config.batch_size,
        Config.seq_len,
        Config.input_dim,
    ), f"Feature shape mismatch. Expected {(Config.batch_size, Config.seq_len, Config.input_dim)}, got {feat.shape}"

    # Pair Indices: (Batch, Seq_Len=107)
    assert p_idx.shape == (
        Config.batch_size,
        Config.seq_len,
    ), f"Pair indices shape mismatch. Got {p_idx.shape}"

    # Targets: (Batch, Pred_Len=68, Num_Classes=5)
    assert tgt.shape == (
        Config.batch_size,
        Config.pred_len,
        Config.num_classes,
    ), f"Target shape mismatch. Expected {(Config.batch_size, Config.pred_len, Config.num_classes)}, got {tgt.shape}"

    print("Data shapes verified successfully.")

    # ==========================================
    # 3. Model Initialization & Forward Pass
    # ==========================================
    print("\n--- Initializing Model ---")
    model = HCTADPBiGRU().to(device)

    # Verify Forward Pass
    feat = feat.to(device)
    p_idx = p_idx.to(device)
    p_mask = p_mask.to(device)

    output = model(feat, p_idx, p_mask)

    # Output shape should be (Batch, Seq_Len=107, Num_Classes=5)
    # Note: The model outputs predictions for the full sequence length (107),
    # slicing happens in the loss function.
    expected_out_shape = (Config.batch_size, Config.seq_len, Config.num_classes)
    assert (
        output.shape == expected_out_shape
    ), f"Model output shape mismatch. Expected {expected_out_shape}, got {output.shape}"

    print("Model forward pass verified.")

    # ==========================================
    # 4. Loss Function Verification
    # ==========================================
    print("\n--- Verifying Loss Function ---")
    criterion = MCRMSELoss()
    tgt = tgt.to(device)

    loss = criterion(output, tgt)

    assert loss.dim() == 0, "Loss should be a scalar tensor."
    assert not torch.isnan(loss), "Loss returned NaN."
    assert loss.item() >= 0, "Loss should be non-negative."

    print(f"Initial Loss check passed. Value: {loss.item():.4f}")

    # ==========================================
    # 5. Training Loop Demonstration
    # ==========================================
    print("\n--- Starting Training Loop (Demo) ---")
    optimizer = optim.AdamW(model.parameters(), lr=Config.learning_rate)

    best_val_score = float("inf")

    for epoch in range(Config.epochs):
        print(f"Epoch {epoch+1}/{Config.epochs}")

        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, Config.max_grad_norm
        )

        # Validate
        val_loss, val_mcrmse = validate(model, val_loader, criterion, device)

        print(
            f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val MCRMSE: {val_mcrmse:.4f}"
        )

        # Checkpoint
        if val_mcrmse < best_val_score:
            best_val_score = val_mcrmse
            torch.save(model.state_dict(), Config.best_model_path)
            print("  Saved best model.")

    assert os.path.exists(Config.best_model_path), "Best model file was not saved."
    print("Training loop completed.")

    # ==========================================
    # 6. Inference & Submission Generation
    # ==========================================
    print("\n--- Generating Submission ---")
    # Load best model
    model.load_state_dict(torch.load(Config.best_model_path, map_location=device))

    generate_submission(model, test_loader, device, Config.submission_path)

    # Verify Submission File
    assert os.path.exists(Config.submission_path), "Submission file not found."

    sub_df = pd.read_csv(Config.submission_path)
    print(f"Submission loaded. Shape: {sub_df.shape}")

    # Verify columns
    expected_cols = ["id_seqpos"] + Config.target_columns
    assert (
        list(sub_df.columns) == expected_cols
    ), f"Submission columns mismatch. Got {list(sub_df.columns)}"

    # Verify row count
    # Test set size in debug mode: min(debug_subset_size, actual_test_size)
    # The actual test set is 240 samples.
    # If debug_subset_size=50, we expect 50 samples * 107 positions = 5350 rows.
    # Note: get_loaders might load slightly fewer if drop_last=True for train, but test loader doesn't drop last.
    # However, 'debug_subset_size' applies to the initial dataframe slice.

    # Let's verify against the test_loader dataset size
    num_test_samples = len(test_loader.dataset)
    expected_rows = num_test_samples * Config.seq_len
    assert (
        len(sub_df) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows} (Samples {num_test_samples} * 107), got {len(sub_df)}"

    # Check for NaN values
    assert not sub_df.isnull().values.any(), "Submission contains NaN values."

    print("Submission file verified successfully.")
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    run_demo()
