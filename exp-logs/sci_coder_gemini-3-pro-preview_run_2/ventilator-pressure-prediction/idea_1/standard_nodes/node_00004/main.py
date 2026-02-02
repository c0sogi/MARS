import torch
import numpy as np
import pandas as pd
import sys
import os

# Import library modules
from library.config import Config
from library.utils import seed_everything, load_checkpoint
from library.dataset import get_data_loaders
from library.model import BiLSTMRegressor
from library.trainer import Trainer


def main():
    # ==========================================
    # 1. Configuration for Fast Baseline
    # ==========================================
    # Override default Config values to ensure quick execution
    Config.EPOCHS = 5
    Config.DEBUG = False
    Config.DEBUG_SAMPLE_SIZE = 5000  # Limit to 5000 breaths for speed

    # Ensure reproducibility
    seed_everything(Config.SEED)

    print("Configuration configured for fast baseline.")
    print(f"Epochs: {Config.EPOCHS}")
    print(f"Debug Mode: {Config.DEBUG}")

    # ==========================================
    # 2. Data Loading
    # ==========================================
    print("\nInitializing Data Loaders...")
    # load_cached_data=True allows using existing preprocessed files if available.
    # If not, it processes raw data using the Config settings (including DEBUG limit).
    train_loader, val_loader, test_loader, test_ids = get_data_loaders(
        load_cached_data=False
    )

    # ==========================================
    # 3. Model Initialization
    # ==========================================
    print("\nInitializing Model...")
    model = BiLSTMRegressor()

    # ==========================================
    # 4. Training
    # ==========================================
    print("\nStarting Training...")
    trainer = Trainer(model, train_loader, val_loader, test_loader, test_ids)
    trainer.fit()

    # ==========================================
    # 5. Validation Assessment & Failure Analysis
    # ==========================================
    print("\nRunning Validation Assessment...")

    # Load the best model checkpoint
    load_checkpoint(model, filename=Config.MODEL_CHECKPOINT)
    model.eval()
    model.to(Config.DEVICE)

    all_preds = []
    all_targets = []
    all_inputs = []

    # Run inference on validation set
    # Disable gradients for speed and memory efficiency
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(Config.DEVICE)
            y = y.to(Config.DEVICE)

            preds = model(x)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            all_inputs.append(x.cpu().numpy())

    # Concatenate and flatten
    # Preds/Targets: (N_Breaths, Seq_Len) -> (Total_Steps,)
    all_preds = np.concatenate(all_preds, axis=0).flatten()
    all_targets = np.concatenate(all_targets, axis=0).flatten()

    # Inputs: (N_Breaths, Seq_Len, N_Features) -> (Total_Steps, N_Features)
    all_inputs = np.concatenate(all_inputs, axis=0)
    all_inputs = all_inputs.reshape(-1, all_inputs.shape[-1])

    # Feature mapping based on dataset.py: ["time_step", "u_in", "u_out", "R", "C", "u_in_cumsum"]
    # u_out is at index 2
    u_out = all_inputs[:, 2]

    # Calculate Metric: MAE on Inspiratory Phase (u_out == 0)
    inspiratory_mask = u_out == 0
    valid_preds = all_preds[inspiratory_mask]
    valid_targets = all_targets[inspiratory_mask]

    final_metric = np.mean(np.abs(valid_preds - valid_targets))

    # REQUIRED OUTPUT
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    print("\nPerforming Failure Analysis...")

    # Calculate absolute error for all points
    errors = np.abs(all_preds - all_targets)

    # Create DataFrame for correlation analysis
    feature_names = [
        "time_step",
        "u_in",
        "u_out",
        "R",
        "C",
        "u_in_cumsum",
        "R_u_in",
        "vol_C",
    ]
    df_analysis = pd.DataFrame(all_inputs, columns=feature_names)
    df_analysis["error"] = errors

    # Compute correlation
    correlations = (
        df_analysis.corr()["error"].drop("error").sort_values(ascending=False)
    )

    print("Correlation between Model Error and Input Features:")
    print(correlations)

    # ==========================================
    # 6. Submission Generation
    # ==========================================
    print("\nGenerating Submission...")
    trainer.predict()
    print("Workflow completed successfully.")


if __name__ == "__main__":
    main()
