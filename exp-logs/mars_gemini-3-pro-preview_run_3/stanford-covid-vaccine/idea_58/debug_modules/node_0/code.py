import sys
import os
import torch
import pandas as pd
import numpy as np

# Ensure the current directory is in the path for module imports
sys.path.append(os.getcwd())

from library.config import Config
from library.data_utils import load_dataset
from library.dataset import RNADataset
from library.model import HCSDBiGRU
from library.loss_metric import MCRMSELoss, compute_competition_metric
from library.train_eval import run_training, generate_submission, set_seed


def main():
    print("==== RNA Degradation Prediction Library Demo ====\n")

    # ---------------------------------------------------------
    # 1. Configuration Setup
    # ---------------------------------------------------------
    print("[1/5] Setting up configuration for fast execution...")

    # Override Config parameters for the demo to ensure speed
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 8
    Config.WORKING_DIR = "./working/demo_execution"

    # Define distinct cache paths for the demo to avoid conflicts
    Config.TRAIN_CACHE = os.path.join(Config.WORKING_DIR, "train_data_demo.npz")
    Config.VAL_CACHE = os.path.join(Config.WORKING_DIR, "val_data_demo.npz")
    Config.TEST_CACHE = os.path.join(Config.WORKING_DIR, "test_data_demo.npz")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission_demo.csv")

    # Ensure the working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seed for reproducibility
    set_seed(Config.SEED)
    print(f"      Working Directory: {Config.WORKING_DIR}")
    print("      Configuration updated successfully.")

    # ---------------------------------------------------------
    # 2. Data Loading & Verification
    # ---------------------------------------------------------
    print("\n[2/5] Verifying Data Loading and Processing...")

    # Load a small subset of the training data
    max_samples_demo = 32
    train_dataset = RNADataset(
        mode="train", load_cached_data=False, max_samples=max_samples_demo
    )

    # Verify dataset length
    assert (
        len(train_dataset) == max_samples_demo
    ), f"Dataset length mismatch. Expected {max_samples_demo}, got {len(train_dataset)}"

    # Retrieve a single sample
    sample = train_dataset[0]

    # Verify Data Shapes
    # Features: (Seq_Len=107, Channels=14)
    assert sample["features"].shape == (
        107,
        14,
    ), f"Features shape incorrect: {sample['features'].shape}"

    # BPP Indices: (Seq_Len=107,)
    assert sample["bpp_indices"].shape == (
        107,
    ), f"BPP Indices shape incorrect: {sample['bpp_indices'].shape}"

    # Targets: (Seq_Scored=68, Targets=5) - Only present in train/val
    assert sample["targets"].shape == (
        68,
        5,
    ), f"Targets shape incorrect: {sample['targets'].shape}"

    print(f"      Loaded {len(train_dataset)} samples.")
    print("      Sample shapes verified: Features (107, 14), Targets (68, 5).")

    # ---------------------------------------------------------
    # 3. Model Architecture & Loss Logic
    # ---------------------------------------------------------
    print("\n[3/5] Verifying Model Architecture and Loss Calculation...")

    # Initialize Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HCSDBiGRU().to(device)

    # Create a dummy batch from the dataset
    from torch.utils.data import DataLoader

    loader = DataLoader(train_dataset, batch_size=4, shuffle=False)
    batch = next(iter(loader))

    features = batch["features"].to(device)
    bpp_indices = batch["bpp_indices"].to(device)
    bpp_mask = batch["bpp_mask"].to(device)
    targets = batch["targets"].to(device)

    # Forward Pass
    outputs = model(features, bpp_indices, bpp_mask)

    # Check Output Shape: (Batch, Seq_Len, Num_Targets) -> (4, 107, 5)
    expected_shape = (4, 107, 5)
    assert (
        outputs.shape == expected_shape
    ), f"Model output shape mismatch. Expected {expected_shape}, got {outputs.shape}"

    # Compute Loss
    criterion = MCRMSELoss()
    loss = criterion(outputs, targets)

    # Check Loss Validity
    assert not torch.isnan(loss), "Loss is NaN."
    assert loss.item() >= 0, "Loss is negative."

    # Compute Competition Metric (Scored Columns Only)
    # Note: compute_competition_metric expects CPU tensors/arrays
    metric_val = compute_competition_metric(outputs, targets)

    print(f"      Forward pass successful. Output shape: {outputs.shape}")
    print(f"      Loss calculated: {loss.item():.4f}")
    print(f"      Competition Metric (Scored Cols): {metric_val:.4f}")

    # ---------------------------------------------------------
    # 4. Training Loop Execution
    # ---------------------------------------------------------
    print("\n[4/5] Running Training Loop (1 Epoch)...")

    # Run training using the library function
    # We use a slightly larger subset (64) to ensure batches flow correctly
    run_training(max_samples=64, epochs=1)

    # Verify that the best model was saved
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    assert os.path.exists(
        best_model_path
    ), f"Best model file not found at {best_model_path}"

    print("      Training completed successfully.")
    print("      Best model checkpoint verified.")

    # ---------------------------------------------------------
    # 5. Inference & Submission
    # ---------------------------------------------------------
    print("\n[5/5] Generating Submission...")

    # Generate submission for a subset of test data
    test_samples = 20
    generate_submission(max_samples=test_samples)

    # Verify Submission File
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission CSV not found."

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)

    # Verify Row Count: test_samples * 107 positions
    expected_rows = test_samples * 107
    assert (
        len(df_sub) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(df_sub)}"

    # Verify Columns
    expected_cols = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]
    assert (
        list(df_sub.columns) == expected_cols
    ), f"Submission columns mismatch. Got {list(df_sub.columns)}"

    print(f"      Submission saved to {Config.SUBMISSION_PATH}")
    print(f"      Dimensions verified: {df_sub.shape}")

    print("\n==== Demo Completed Successfully ====")


if __name__ == "__main__":
    main()
