import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import library modules
from library.config import Config
from library.dataset import RNADataset
from library.model import DeepResidualBiGRU
from library.loss import MCRMSELoss
from library.metrics import compute_scored_mcrmse
from library.train import run_training


def main():
    print("=== RNA Degradation Prediction Demo ===")

    # 1. Configuration Setup
    # Override Config paths to use a specific demo directory within ./working
    # This ensures we don't interfere with other runs and have a clean state.
    DEMO_DIR = "./working/demo_execution"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    Config.WORKING_DIR = DEMO_DIR
    Config.CACHE_DIR = os.path.join(DEMO_DIR, "cache")
    Config.MODEL_SAVE_PATH = os.path.join(DEMO_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(DEMO_DIR, "submission_demo.csv")
    Config.NUM_WORKERS = 0  # Set to 0 for simple sequential loading in demo

    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    print(f"Working Directory: {Config.WORKING_DIR}")

    # 2. Data Loading Demonstration
    print("\n--- 2. Data Loading Verification ---")
    # Load a small subset of training data via the Dataset class
    # We use 'debug=True' logic implicitly by loading the dataset and checking items

    # Note: The first load might take a moment to process parquet to npz
    train_dataset = RNADataset(split="train", load_cached_data=False)
    print(f"Training Dataset Size: {len(train_dataset)}")

    # Verify a single item
    sample_idx = 0
    sample = train_dataset[sample_idx]

    # Check keys
    expected_keys = {"features", "adjacency", "bpp_mask", "targets", "id"}
    assert expected_keys.issubset(
        sample.keys()
    ), f"Missing keys in dataset item. Found: {sample.keys()}"

    # Check Shapes
    # Features: (107, 14)
    assert sample["features"].shape == (
        107,
        14,
    ), f"Incorrect feature shape: {sample['features'].shape}"
    # Adjacency: (107,)
    assert sample["adjacency"].shape == (
        107,
    ), f"Incorrect adjacency shape: {sample['adjacency'].shape}"
    # Targets: (68, 5) - Training data has targets
    assert sample["targets"].shape == (
        68,
        5,
    ), f"Incorrect target shape: {sample['targets'].shape}"

    print("Data Loading verified successfully.")

    # 3. Model Architecture Demonstration
    print("\n--- 3. Model Architecture Verification ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DeepResidualBiGRU().to(device)

    # Create a dummy batch
    batch_size = 4
    dummy_features = torch.randn(batch_size, 107, 14).to(device)
    dummy_adjacency = torch.randint(0, 107, (batch_size, 107)).to(device)
    dummy_mask = torch.ones(batch_size, 107).to(device)

    # Forward Pass
    model.eval()
    with torch.no_grad():
        output = model(dummy_features, dummy_adjacency, dummy_mask)

    # Verify Output Shape: (Batch, Seq_Len, Num_Targets) -> (4, 107, 5)
    print(f"Model Output Shape: {output.shape}")
    assert output.shape == (
        batch_size,
        107,
        5,
    ), f"Expected (4, 107, 5), got {output.shape}"
    print("Model forward pass verified successfully.")

    # 4. Loss Function Demonstration
    print("\n--- 4. Loss Function Verification ---")
    criterion = MCRMSELoss()

    # Create dummy predictions (full length 107) and targets (scored length 68)
    # The loss function handles the slicing internally
    dummy_preds = torch.ones(batch_size, 107, 5).to(device) * 0.5
    dummy_targets = torch.zeros(batch_size, 68, 5).to(device)  # Ground truth is 0

    # Calculate loss manually:
    # Sliced preds: (4, 68, 5) -> values are 0.5
    # Diff: 0.5 - 0.0 = 0.5
    # Squared Diff: 0.25
    # Mean per column: 0.25
    # RMSE per column: sqrt(0.25) = 0.5
    # MCRMSE: mean(0.5) = 0.5

    loss = criterion(dummy_preds, dummy_targets)
    print(f"Calculated Loss: {loss.item():.4f}")

    # Allow small float error
    assert (
        abs(loss.item() - 0.5) < 1e-4
    ), f"Loss calculation incorrect. Expected ~0.5, got {loss.item()}"
    print("Loss function verified successfully.")

    # 5. Metric Demonstration
    print("\n--- 5. Metric Verification (Scored MCRMSE) ---")
    # compute_scored_mcrmse filters for specific columns: reactivity(0), deg_Mg_pH10(1), deg_Mg_50C(3)
    # It ignores columns 2 and 4.

    # Let's set up predictions where scored columns have error 1.0, and ignored columns have error 100.0
    # Preds: 1.0 everywhere
    # Targets: 0.0 everywhere
    # But for ignored columns (2, 4), let's make preds 101.0 to test filtering

    metric_preds = torch.ones(batch_size, 107, 5)
    metric_preds[:, :, 2] = 101.0  # Ignored column
    metric_preds[:, :, 4] = 101.0  # Ignored column

    metric_targets = torch.zeros(batch_size, 68, 5)

    score = compute_scored_mcrmse(metric_preds, metric_targets)
    print(f"Calculated Metric Score: {score:.4f}")

    # Expected:
    # Scored cols (0, 1, 3): Pred=1, Tgt=0 -> RMSE=1
    # Mean of RMSEs = 1.0
    assert (
        abs(score - 1.0) < 1e-4
    ), f"Metric calculation incorrect. Expected 1.0, got {score}"
    print("Metric calculation verified successfully.")

    # 6. Training Loop Demonstration
    print("\n--- 6. Training Loop Execution (Debug Mode) ---")
    # We run the training function provided in library.train
    # debug=True ensures we use a tiny subset of data for speed

    try:
        run_training(epochs=2, debug=True)
        print("Training loop completed successfully.")
    except Exception as e:
        print(f"Training loop failed with error: {e}")
        raise e

    # Verify model was saved
    if os.path.exists(Config.MODEL_SAVE_PATH):
        print(f"Model checkpoint found at {Config.MODEL_SAVE_PATH}")
    else:
        raise FileNotFoundError("Model checkpoint was not created.")

    # 7. Inference and Submission Generation
    print("\n--- 7. Inference on Test Set ---")

    # Load Test Data
    test_dataset = RNADataset(split="test", load_cached_data=False)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # Load Best Model
    model = DeepResidualBiGRU().to(device)
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    predictions = []
    ids = []

    print("Running inference...")
    with torch.no_grad():
        for batch in test_loader:
            feats = batch["features"].to(device)
            adj = batch["adjacency"].to(device)
            mask = batch["bpp_mask"].to(device)
            batch_ids = batch["id"]

            # Forward
            preds = model(feats, adj, mask)  # (B, 107, 5)

            predictions.append(preds.cpu().numpy())
            ids.extend(batch_ids)

    predictions = np.concatenate(predictions, axis=0)  # (N_test, 107, 5)

    print(f"Predictions shape: {predictions.shape}")
    assert predictions.shape[0] == len(
        test_dataset
    ), "Mismatch in number of predictions and test samples"
    assert predictions.shape[1] == 107, "Predictions should be length 107"
    assert predictions.shape[2] == 5, "Predictions should have 5 targets"

    # Generate Submission DataFrame
    # Format requires flattening: id_seqpos, reactivity, ...
    print("Generating submission file...")

    sub_ids = []
    sub_data = []

    for i, sample_id in enumerate(ids):
        # For each sample, we have 107 positions
        sample_preds = predictions[i]  # (107, 5)

        for seqpos in range(107):
            row_id = f"{sample_id}_{seqpos}"
            sub_ids.append(row_id)
            sub_data.append(sample_preds[seqpos])

    sub_df = pd.DataFrame(sub_data, columns=Config.TARGET_COLS)
    sub_df.insert(0, "id_seqpos", sub_ids)

    print(f"Submission DataFrame shape: {sub_df.shape}")
    print(sub_df.head())

    # Save submission
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
