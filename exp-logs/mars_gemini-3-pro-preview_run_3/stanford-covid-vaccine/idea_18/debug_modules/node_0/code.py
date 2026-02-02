import os
import sys
import torch
import numpy as np
import pandas as pd
import warnings
import shutil
from library.config import Config
from library.utils import seed_everything, calculate_mcrmse
from library.data import get_dataloaders, get_test_dataloader
from library.model import RISRBiGRU
from library.loss import MCRMSELoss
from library.train import Trainer

# Filter warnings for clean output
warnings.filterwarnings("ignore")


def run_demo():
    print("==== RNA Degradation Prediction Pipeline Demo ====")

    # 1. Configuration Setup
    # Enable debug mode for fast execution (2 epochs, small batch size)
    config = Config(debug=True)

    # Set a specific working directory for this demo
    config.working_dir = "./working/demo_execution"
    config.cache_dir = os.path.join(config.working_dir, "cache")
    config.best_model_path = os.path.join(config.working_dir, "best_model.pth")
    config.submission_path = os.path.join(config.working_dir, "submission.csv")

    # Ensure directories exist
    os.makedirs(config.working_dir, exist_ok=True)
    os.makedirs(config.cache_dir, exist_ok=True)

    # Set seeds for reproducibility
    seed_everything(config.seed)
    print(f"Configuration: Debug={config.debug}, Device={config.device}")
    print(f"Working Directory: {config.working_dir}")

    # 2. Data Loading & Verification
    print("\n[1/5] Loading Data...")
    train_loader, val_loader = get_dataloaders(config, load_cached_data=True)

    # Fetch one batch to verify shapes
    # Inputs: (B, L, 14), Adjacency: (B, L, W), Targets: (B, Pred_Len, 5)
    inputs, adjacency, targets = next(iter(train_loader))

    print(
        f"Batch Shapes -> Inputs: {inputs.shape}, Adjacency: {adjacency.shape}, Targets: {targets.shape}"
    )

    # Assertions
    assert inputs.shape == (
        config.batch_size,
        config.seq_len,
        config.input_channels,
    ), "Input shape mismatch"
    assert adjacency.shape == (
        config.batch_size,
        config.seq_len,
        config.window_size,
    ), "Adjacency shape mismatch"
    assert targets.shape == (
        config.batch_size,
        config.pred_len,
        config.num_targets,
    ), "Target shape mismatch"
    print("Data loading verification passed.")

    # 3. Model Initialization & Forward Pass
    print("\n[2/5] Initializing Model...")
    model = RISRBiGRU(config).to(config.device)

    # Move batch to device
    inputs = inputs.to(config.device)
    adjacency = adjacency.to(config.device)
    targets = targets.to(config.device)

    # Forward pass
    print("Executing forward pass...")
    outputs = model(inputs, adjacency)

    print(f"Output Shape: {outputs.shape}")

    # Assertions
    # Model outputs full sequence length (107), loss handles slicing to scored length (68)
    assert outputs.shape == (
        config.batch_size,
        config.seq_len,
        config.num_targets,
    ), "Model output shape mismatch"
    print("Model forward pass verification passed.")

    # 4. Loss Function Verification
    print("\n[3/5] Verifying Loss Function...")
    criterion = MCRMSELoss()
    loss = criterion(outputs, targets)

    print(f"Calculated Loss: {loss.item():.6f}")

    # Assertions
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() >= 0, "Loss is negative"

    # Manual Logic Check for MCRMSE
    # Create dummy data:
    # Target: 2 samples, 2 positions, 2 columns. All zeros.
    # Pred: All ones.
    # Error = 1. MSE = 1. RMSE = 1. Mean RMSE = 1.
    dummy_targets = torch.zeros(2, 2, 2)
    dummy_preds = torch.ones(2, 2, 2)
    dummy_loss = criterion(dummy_preds, dummy_targets)

    assert (
        abs(dummy_loss.item() - 1.0) < 1e-5
    ), f"Manual MCRMSE check failed. Expected 1.0, got {dummy_loss.item()}"
    print("Loss function logic verification passed.")

    # 5. Training Loop Demonstration
    print("\n[4/5] Running Training Loop (Debug Mode)...")
    trainer = Trainer(config, train_loader, val_loader)
    trainer.fit()

    # Verify model checkpoint was saved
    assert os.path.exists(
        config.best_model_path
    ), "Best model checkpoint not found after training"
    print("Training loop completed successfully.")

    # 6. Inference & Submission Generation
    print("\n[5/5] Running Inference on Test Set...")
    test_loader = get_test_dataloader(config, load_cached_data=True)

    # Load best model
    model.load_state_dict(
        torch.load(config.best_model_path, map_location=config.device)
    )
    model.eval()

    submission_data = []

    with torch.no_grad():
        for i, (inputs, adjacency, ids) in enumerate(test_loader):
            inputs = inputs.to(config.device)
            adjacency = adjacency.to(config.device)

            preds = model(inputs, adjacency)

            # Move to CPU
            preds = preds.cpu().numpy()

            # Verify prediction shape (B, 107, 5)
            if i == 0:
                assert (
                    preds.shape[1] == config.seq_len
                ), "Prediction sequence length mismatch"
                assert (
                    preds.shape[2] == config.num_targets
                ), "Prediction target dimension mismatch"

            # Process for submission format
            # We need to flatten: id_seqpos, values...
            # Note: The competition requires predictions for all positions (seq_length=107)
            # even though only seq_scored=68 are scored.

            batch_size = preds.shape[0]
            for b in range(batch_size):
                sample_id = ids[b]
                sample_preds = preds[b]  # (107, 5)

                for seqpos in range(config.seq_len):
                    row_id = f"{sample_id}_{seqpos}"
                    row_values = sample_preds[seqpos].tolist()
                    submission_data.append([row_id] + row_values)

            # Break after one batch for demo speed
            break

    # Create DataFrame
    cols = ["id_seqpos"] + config.target_cols
    sub_df = pd.DataFrame(submission_data, columns=cols)

    print(f"Generated submission samples:\n{sub_df.head(3)}")

    # Save submission
    sub_df.to_csv(config.submission_path, index=False)
    assert os.path.exists(config.submission_path), "Submission file was not created"

    print("\n==== Demo Completed Successfully ====")


if __name__ == "__main__":
    run_demo()
