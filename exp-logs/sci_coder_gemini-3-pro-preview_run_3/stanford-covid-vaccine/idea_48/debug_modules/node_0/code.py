import os
import sys
import torch
import numpy as np
import pandas as pd
import torch.optim as optim
import warnings

# Ensure the current directory is in the path to import library modules
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, MCRMSELoss, compute_mcrmse_numpy
from library.data import get_dataloaders
from library.model import RNAModel
from library.train import train_one_epoch, validate, inference, generate_submission

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def run_demo():
    print("=== Starting RNA Degradation Prediction Demo ===\n")

    # ------------------------------------------------------------------------
    # 1. Configuration & Setup
    # ------------------------------------------------------------------------
    print("[1] Configuring environment...")

    # Override Config for a fast demo run
    Config.working_dir = "./working/demo_execution"
    Config.model_save_path = os.path.join(Config.working_dir, "demo_model.pth")
    Config.submission_file = os.path.join(Config.working_dir, "demo_submission.csv")

    # Use distinct cache files for the demo to avoid conflicts
    Config.train_cache_path = os.path.join(Config.working_dir, "train_cache.npy")
    Config.val_cache_path = os.path.join(Config.working_dir, "val_cache.npy")
    Config.test_cache_path = os.path.join(Config.working_dir, "test_cache.npy")

    # Hyperparameters for speed
    Config.epochs = 2
    Config.batch_size = 4
    Config.debug_samples = 20  # Use only 20 samples
    Config.num_workers = 0  # Main process only for demo
    Config.learning_rate = 1e-3

    # Setup directories and seed
    Config.setup()
    seed_everything(Config.seed)
    device = torch.device(Config.device)
    print(f"    Device: {device}")
    print(f"    Debug Samples: {Config.debug_samples}")
    print(f"    Batch Size: {Config.batch_size}")

    # ------------------------------------------------------------------------
    # 2. Data Loading & Verification
    # ------------------------------------------------------------------------
    print("\n[2] Loading and Verifying Data...")

    # Load dataloaders with debug sampling (forces fresh processing if cache doesn't exist)
    train_loader, val_loader, test_loader = get_dataloaders(
        debug_samples=Config.debug_samples, load_cached_data=False
    )

    # Fetch one batch to verify structure
    batch = next(iter(train_loader))
    inputs = batch["inputs"]
    pair_indices = batch["pair_indices"]
    pair_masks = batch["pair_masks"]
    targets = batch["targets"]
    ids = batch["id"]

    # Assertions
    print("    Verifying batch shapes...")
    # Inputs: (Batch, SeqLen=107, InputDim=14)
    assert inputs.shape == (
        Config.batch_size,
        107,
        14,
    ), f"Unexpected input shape: {inputs.shape}"
    # Targets: (Batch, SeqLen=107, NumClasses=5)
    assert targets.shape == (
        Config.batch_size,
        107,
        5,
    ), f"Unexpected target shape: {targets.shape}"
    # Pair Indices: (Batch, SeqLen=107)
    assert pair_indices.shape == (
        Config.batch_size,
        107,
    ), f"Unexpected pair_indices shape: {pair_indices.shape}"

    # Verify data integrity
    # Pair indices should be within sequence length
    assert pair_indices.max() < 107, "Pair indices exceed sequence length"
    # Pair masks should be binary
    unique_masks = torch.unique(pair_masks)
    assert torch.all(
        torch.isin(unique_masks, torch.tensor([0.0, 1.0]))
    ), "Pair masks contain non-binary values"

    print("    Data integrity check passed.")

    # ------------------------------------------------------------------------
    # 3. Model Initialization & Forward Pass
    # ------------------------------------------------------------------------
    print("\n[3] Initializing Model...")
    model = RNAModel().to(device)

    print("    Running forward pass on dummy batch...")
    inputs = inputs.to(device)
    pair_indices = pair_indices.to(device)
    pair_masks = pair_masks.to(device)

    # Forward pass
    outputs = model(inputs, pair_indices, pair_masks)

    # Verify output shape (Batch, SeqLen, NumClasses)
    assert outputs.shape == (
        Config.batch_size,
        107,
        5,
    ), f"Model output shape mismatch: {outputs.shape}"
    assert outputs.requires_grad, "Output should require gradients for training"
    print("    Forward pass successful.")

    # ------------------------------------------------------------------------
    # 4. Loss Function Verification
    # ------------------------------------------------------------------------
    print("\n[4] Verifying MCRMSE Loss Logic...")
    criterion = MCRMSELoss()

    # Create synthetic data to verify calculation
    # Scored indices are [0, 1, 3] (reactivity, deg_Mg_pH10, deg_Mg_50C)
    # Let's make predictions 0 and targets 1 for these columns
    # RMSE for each column should be 1.0. Mean RMSE should be 1.0.

    synth_preds = torch.zeros((2, 107, 5))
    synth_targets = torch.zeros((2, 107, 5))

    # Set targets to 1.0 for all columns
    synth_targets[:, :, :] = 1.0

    # The loss function slices to pred_len (68) and selects indices [0, 1, 3]
    # Preds=0, Targets=1 -> Diff=1 -> SqDiff=1 -> MSE=1 -> RMSE=1 -> Mean(RMSE)=1
    loss_val = criterion(synth_preds, synth_targets)

    print(f"    Calculated Loss: {loss_val.item():.4f}")
    assert (
        abs(loss_val.item() - 1.0) < 1e-5
    ), f"Expected loss 1.0, got {loss_val.item()}"
    print("    Loss function logic verified.")

    # ------------------------------------------------------------------------
    # 5. Training Loop Demonstration
    # ------------------------------------------------------------------------
    print("\n[5] Running Training Loop (2 Epochs)...")
    optimizer = optim.AdamW(model.parameters(), lr=Config.learning_rate)

    for epoch in range(Config.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_score = validate(model, val_loader, device)
        print(
            f"    Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val MCRMSE = {val_score:.4f}"
        )

    # Save the model
    torch.save(model.state_dict(), Config.model_save_path)
    print(f"    Model saved to {Config.model_save_path}")

    # ------------------------------------------------------------------------
    # 6. Inference & Submission
    # ------------------------------------------------------------------------
    print("\n[6] Running Inference on Test Set...")

    # Load best model (in this case, just the final one)
    model.load_state_dict(torch.load(Config.model_save_path, map_location=device))

    test_preds, test_ids = inference(model, test_loader, device)
    print(f"    Test Predictions Shape: {test_preds.shape}")

    # Verify shape: (NumTestSamples, SeqLen, NumClasses)
    # Note: debug_samples applies to test set loading as well
    expected_test_samples = min(Config.debug_samples, 240)  # 240 is total test size
    assert test_preds.shape == (expected_test_samples, 107, 5)

    print("    Generating submission file...")
    generate_submission(test_preds, test_ids, Config.submission_file)

    # Verify submission file format
    df_sub = pd.read_csv(Config.submission_file)
    print(f"    Submission file loaded. Shape: {df_sub.shape}")

    # Expected rows: NumSamples * SeqLen
    expected_rows = expected_test_samples * 107
    assert (
        len(df_sub) == expected_rows
    ), f"Expected {expected_rows} rows, got {len(df_sub)}"

    # Expected columns
    expected_cols = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]
    assert list(df_sub.columns) == expected_cols, "Submission columns mismatch"

    print("    Submission format verified.")
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
