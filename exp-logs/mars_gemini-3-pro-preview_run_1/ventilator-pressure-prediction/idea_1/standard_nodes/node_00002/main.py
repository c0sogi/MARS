import pandas as pd
import numpy as np
import torch
import os
import sys

# Import library modules
from library.config import Config
from library.utils import seed_everything, MaskedL1Loss
from library.dataset import prepare_data
from library.model import BidirectionalLSTM
from library.trainer import Trainer


def main():
    # 1. Setup and Configuration
    seed_everything(Config.SEED)
    Config.setup()

    # Fast Baseline Configuration
    # We adjust epochs and batch size to ensure the run completes quickly
    # while still providing a robust baseline on the A100 GPU.
    Config.EPOCHS = 20
    Config.BATCH_SIZE = 1024

    print(f"Configuration: Epochs={Config.EPOCHS}, Batch Size={Config.BATCH_SIZE}")

    # 2. Data Preparation
    # Using cached data to speed up loading
    print("Preparing data...")
    train_loader, val_loader, test_loader = prepare_data(
        load_cached_data=True, batch_size=Config.BATCH_SIZE
    )

    # 3. Model Initialization
    model = BidirectionalLSTM(
        input_dim=Config.INPUT_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        num_layers=Config.NUM_LAYERS,
        dropout=Config.DROPOUT,
        bidirectional=Config.BIDIRECTIONAL,
    )

    # 4. Training
    # The Trainer class handles the training loop, optimizer, and scheduler
    trainer = Trainer(model, Config)
    trainer.fit(train_loader, val_loader)

    # 5. Validation Assessment & Failure Analysis
    print("\n=== Validation Assessment ===")
    device = Config.DEVICE
    model.eval()

    val_loss_sum = 0.0
    val_valid_points = 0

    # Buffers for failure analysis
    all_preds = []
    all_targets = []
    all_inputs = []
    all_u_out = []

    with torch.no_grad():
        for batch in val_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            u_out = batch["u_out"].to(device)

            preds = model(x)

            # Metric Calculation: MAE on Inspiratory Phase Only
            # Mask: 1 where u_out is 0 (inspiratory), 0 otherwise
            mask = 1 - u_out
            abs_err = torch.abs(preds - y) * mask

            val_loss_sum += abs_err.sum().item()
            val_valid_points += mask.sum().item()

            # Store data for failure analysis (move to CPU to save GPU memory)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            all_u_out.append(u_out.cpu().numpy())
            all_inputs.append(x.cpu().numpy())

    # Compute Final Metric
    final_metric = val_loss_sum / val_valid_points if val_valid_points > 0 else 0.0
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("\n=== Failure Analysis ===")

    # Flatten collected data
    # Shapes: List of (B, 80) -> (N_total, 80) -> (N_total*80,)
    flat_preds = np.concatenate(all_preds).flatten()
    flat_targets = np.concatenate(all_targets).flatten()
    flat_u_out = np.concatenate(all_u_out).flatten()

    # Inputs: List of (B, 80, 5) -> (N_total, 80, 5) -> (N_total*80, 5)
    flat_inputs = np.concatenate(all_inputs)
    flat_inputs = flat_inputs.reshape(-1, flat_inputs.shape[-1])

    # Calculate absolute error vector
    errors = np.abs(flat_preds - flat_targets)

    # Filter for Inspiratory Phase (u_out == 0)
    insp_mask = flat_u_out == 0

    insp_errors = errors[insp_mask]
    insp_features = flat_inputs[insp_mask]

    # Create DataFrame for correlation analysis
    # Feature cols from Config: ["time_step", "u_in", "u_out", "R", "C"]
    df_analysis = pd.DataFrame(insp_features, columns=Config.FEATURE_COLS)
    df_analysis["error_magnitude"] = insp_errors

    # Calculate correlations between features and error magnitude
    correlations = df_analysis.corr()["error_magnitude"].drop("error_magnitude")

    print("Correlation between Error Magnitude and Input Features (Inspiratory Phase):")
    print(correlations.sort_values(ascending=False))

    # 7. Submission
    print("\n=== Generating Submission ===")
    trainer.predict(test_loader)
    print("Submission generation complete.")


if __name__ == "__main__":
    main()
