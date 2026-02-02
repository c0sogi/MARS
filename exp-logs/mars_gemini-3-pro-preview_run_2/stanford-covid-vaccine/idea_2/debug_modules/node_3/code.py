import os
import torch
import pandas as pd
import numpy as np
import shutil

# Import library components
from library.config import Config
from library.utils import seed_everything, MCRMSELoss
from library.data import get_dataloaders
from library.model import GEHN
from library.train import Trainer
from library.predict import generate_submission


def main():
    print("=== RNA Degradation Prediction Pipeline Demonstration ===\n")

    # 1. Configuration Setup
    # Enable debug mode for speed (subset_size=100, epochs=2)
    config = Config(debug=True)

    # Use a specific directory for this demo to avoid conflicts
    config.working_dir = "./working/demo_execution"
    config.best_model_path = os.path.join(config.working_dir, "best_model.pth")
    config.predictions_path = os.path.join(config.working_dir, "predictions.npy")
    config.submission_path = os.path.join(config.working_dir, "submission.csv")

    # Ensure directory exists
    os.makedirs(config.working_dir, exist_ok=True)

    # Set seeds for reproducibility
    seed_everything(config.seed)
    print(f"Configuration initialized. Debug mode: {config.debug}")
    print(f"Working directory: {config.working_dir}")

    # 2. Data Loading & Verification
    print("\n--- Step 1: Data Loading ---")
    # We force reload to ensure we process the debug subset freshly
    # In a real run, load_cached_data=True is preferred for speed
    train_loader, val_loader, test_loader = get_dataloaders(
        config, load_cached_data=False
    )

    # Verify Train Loader
    batch = next(iter(train_loader))
    inputs, adj, targets = batch["inputs"], batch["adj"], batch["targets"]

    print(
        f"Batch shapes -> Inputs: {inputs.shape}, Adj: {adj.shape}, Targets: {targets.shape}"
    )

    # Assertions
    # Inputs: (Batch, 107, 14)
    assert inputs.shape[1] == 107, "Input sequence length mismatch"
    assert inputs.shape[2] == 14, "Input channel dimension mismatch"
    # Adj: (Batch, 107, 107)
    assert (
        adj.shape[1] == 107 and adj.shape[2] == 107
    ), "Adjacency matrix shape mismatch"
    # Targets: (Batch, 107, 5)
    assert targets.shape[2] == 5, "Target dimension mismatch"

    print("Data loading verification passed.")

    # 3. Model Initialization & Forward Pass Verification
    print("\n--- Step 2: Model Architecture & Loss ---")
    model = GEHN(config).to(config.device)
    criterion = MCRMSELoss(num_scored=config.pred_len)

    # Move batch to device
    inputs = inputs.to(config.device)
    adj = adj.to(config.device)
    targets = targets.to(config.device)

    # Forward pass
    model.eval()
    with torch.no_grad():
        preds = model(inputs, adj)

    print(f"Model output shape: {preds.shape}")

    # Assertions
    assert preds.shape == targets.shape, "Model output shape does not match targets"

    # Loss calculation
    loss = criterion(preds, targets)
    print(f"Calculated MCRMSE Loss: {loss.item():.4f}")

    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() >= 0, "Loss is negative"

    print("Model and loss verification passed.")

    # 4. Training Loop
    print("\n--- Step 3: Training (Debug Mode) ---")
    trainer = Trainer(config)

    # This will run for config.epochs (2 in debug mode)
    trainer.fit(train_loader, val_loader)

    # Verify model artifact creation
    assert os.path.exists(config.best_model_path), "Best model file was not saved."
    print("Training complete. Model artifact verified.")

    # 5. Inference & Submission
    print("\n--- Step 4: Inference & Submission Generation ---")
    # Generate submission using the trained model
    generate_submission(config, load_cached_data=True)

    # Verify Submission File
    assert os.path.exists(config.submission_path), "Submission file not found."

    sub_df = pd.read_csv(config.submission_path)
    print(f"Submission DataFrame shape: {sub_df.shape}")
    print(f"Submission columns: {sub_df.columns.tolist()}")

    # Calculate expected rows
    # In debug mode, test set is truncated to subset_size (100)
    # Each sample has 107 positions
    expected_rows = config.subset_size * config.seq_len
    assert (
        len(sub_df) == expected_rows
    ), f"Expected {expected_rows} rows in submission, found {len(sub_df)}"

    # Check required columns
    required_cols = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]
    for col in required_cols:
        assert col in sub_df.columns, f"Missing column {col} in submission"

    # Check values are numeric
    assert pd.api.types.is_numeric_dtype(
        sub_df["reactivity"]
    ), "Reactivity column is not numeric"

    print("Submission verification passed.")

    print("\n=== Demonstration Complete Successfully ===")


if __name__ == "__main__":
    main()
