import os
import shutil
import torch
import pandas as pd
import numpy as np

# Import from the provided library files
from library.config import Config
from library.utils import set_seed
from library.data import get_dataloaders
from library.model import StackingDenseRefinedNet
from library.train import criterion_mcrmse, run_training, generate_submission


def main():
    print("=== Starting Demo of RNA Degradation Prediction Library ===\n")

    # 1. Setup and Configuration Overrides for Speed
    # We override Config attributes to run a fast demo (1 epoch, small subset)
    # and to isolate outputs in a specific directory.
    demo_dir = "./working/demo_execution"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    print(f"Configuring environment... Output directory: {demo_dir}")
    Config.CACHE_DIR = os.path.join(demo_dir, "data_cache")
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "submission.csv")
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Set seed for reproducibility
    set_seed(Config.SEED)

    # 2. Verify Data Loading and Shapes
    print("\n--- Verifying Data Loading ---")
    debug_size = 32  # Use only 32 samples for verification
    train_loader, val_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=False,  # Force processing to demonstrate it works
        debug_size=debug_size,
    )

    # Fetch one batch
    features, pair_indices, targets, mask = next(iter(train_loader))

    print(f"Batch shapes:")
    print(f"  Features:     {features.shape}")
    print(f"  Pair Indices: {pair_indices.shape}")
    print(f"  Targets:      {targets.shape}")
    print(f"  Mask:         {mask.shape}")

    # Assertions to verify data logic
    assert features.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        Config.INPUT_CHANNELS,
    ), f"Expected feature shape {(Config.BATCH_SIZE, Config.SEQ_LEN, Config.INPUT_CHANNELS)}, got {features.shape}"
    assert pair_indices.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
    ), "Incorrect pair_indices shape"
    assert targets.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        5,
    ), "Incorrect targets shape"
    assert mask.shape == (Config.BATCH_SIZE, Config.SEQ_LEN, 5), "Incorrect mask shape"
    print("Data loading verification passed.")

    # 3. Verify Model Architecture and Forward Pass
    print("\n--- Verifying Model Architecture ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = StackingDenseRefinedNet().to(device)

    # Move batch to device
    features = features.to(device)
    pair_indices = pair_indices.to(device)
    targets = targets.to(device)
    mask = mask.to(device)

    # Forward pass
    preds = model(features, pair_indices)

    print(f"Model output shape: {preds.shape}")

    # Assertions for model logic
    assert preds.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        5,
    ), f"Expected output shape {(Config.BATCH_SIZE, Config.SEQ_LEN, 5)}, got {preds.shape}"
    assert not torch.isnan(preds).any(), "Model output contains NaNs"
    print("Model forward pass verification passed.")

    # 4. Verify Loss Function
    print("\n--- Verifying Loss Function (MCRMSE) ---")
    loss = criterion_mcrmse(preds, targets, mask)
    print(f"Calculated Loss: {loss.item()}")

    assert loss.dim() == 0, "Loss should be a scalar"
    assert loss.item() >= 0, "Loss should be non-negative"
    print("Loss function verification passed.")

    # 5. Run Training Pipeline
    print("\n--- Running Training Pipeline (1 Epoch, Debug Subset) ---")
    # run_training handles the loop, validation, and saving the best model
    run_training(debug_size=debug_size, epochs=Config.EPOCHS)

    best_model_path = os.path.join(Config.CACHE_DIR, "best_model.pth")
    assert os.path.exists(best_model_path), "Best model file was not saved."
    print("Training pipeline execution successful.")

    # 6. Run Inference/Submission Pipeline
    print("\n--- Running Inference Pipeline ---")
    generate_submission()

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not generated."

    # Verify submission content
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission file shape: {sub_df.shape}")
    print(f"Submission columns: {sub_df.columns.tolist()}")

    # Expected rows: 240 test samples * 107 sequence length
    expected_rows = 240 * Config.SEQ_LEN
    assert (
        len(sub_df) == expected_rows
    ), f"Expected {expected_rows} rows in submission, got {len(sub_df)}"

    # Expected columns: id_seqpos + 5 targets
    expected_cols = ["id_seqpos"] + Config.ALL_TARGETS
    assert (
        list(sub_df.columns) == expected_cols
    ), "Submission columns do not match requirements."

    print("Inference pipeline verification passed.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
