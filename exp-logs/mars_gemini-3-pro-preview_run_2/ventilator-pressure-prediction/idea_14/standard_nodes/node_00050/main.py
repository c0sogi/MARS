import os
import sys
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import seed_everything, load_checkpoint
from library.data import get_data_loaders
from library.train import Trainer
from library.inference import generate_submission
from library.model import DFL_GI_BiLSTM


def main():
    # ==========================================
    # 1. Configuration Overrides for Fast Baseline
    # ==========================================
    # The default 200 epochs is too long for a 2-hour limit.
    # 30 epochs with batch size 256 on A100 is efficient and sufficient for a strong baseline.
    Config.EPOCHS = 30
    Config.T_MAX = 30

    # Ensure we use the full dataset to maximize performance and meet the threshold.
    Config.DEBUG = False

    # ==========================================
    # 2. Setup & Data Loading
    # ==========================================
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    print(
        f"Initializing Fast Baseline Run (Epochs: {Config.EPOCHS}, Device: {device})..."
    )

    # Load data (utilizing cache for speed)
    train_loader, val_loader, _ = get_data_loaders(load_cached_data=True)

    # ==========================================
    # 3. Training
    # ==========================================
    print("Starting Training Pipeline...")
    trainer = Trainer()
    trainer.fit(train_loader, val_loader)

    # ==========================================
    # 4. Validation & Failure Analysis
    # ==========================================
    print("\nStarting Validation and Failure Analysis...")

    # Load the best model saved during training
    model = DFL_GI_BiLSTM().to(device)
    if os.path.exists(Config.MODEL_PATH):
        load_checkpoint(Config.MODEL_PATH, model, device=device)
    else:
        print("Error: Best model checkpoint not found!")
        return

    model.eval()

    # Storage for analysis
    all_preds = []
    all_targets = []
    all_u_out = []
    all_inputs = []

    # Inference Loop
    with torch.no_grad():
        for inputs, targets, u_out in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            u_out = u_out.to(device)

            preds = model(inputs)

            # Move to CPU for analysis
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_u_out.append(u_out.cpu().numpy())
            all_inputs.append(inputs.cpu().numpy())

    # Flatten and Process
    all_preds = np.concatenate(all_preds).flatten()
    all_targets = np.concatenate(all_targets).flatten()
    all_u_out = np.concatenate(all_u_out).flatten()

    # Reshape inputs: (Batch * Seq, Features)
    all_inputs = np.concatenate(all_inputs, axis=0)
    all_inputs = all_inputs.reshape(-1, all_inputs.shape[-1])

    # Filter for Inspiratory Phase (u_out == 0)
    insp_mask = all_u_out == 0

    insp_preds = all_preds[insp_mask]
    insp_targets = all_targets[insp_mask]
    insp_inputs = all_inputs[insp_mask]

    # Calculate Final Metric
    abs_errors = np.abs(insp_preds - insp_targets)
    mae = np.mean(abs_errors)

    # Print Metric in required format
    print(f"Final Validation Metric: {mae}")

    # Failure Analysis: Correlation between Input Features and Error Magnitude
    print("\nFailure Analysis (Correlation with Absolute Error):")
    feature_names = Config.INPUT_FEATURES

    # Create DataFrame for correlation calculation
    analysis_df = pd.DataFrame(insp_inputs, columns=feature_names)
    analysis_df["abs_error"] = abs_errors

    # Compute correlations
    correlations = (
        analysis_df.corr()["abs_error"].drop("abs_error").sort_values(ascending=False)
    )
    print(correlations)

    # ==========================================
    # 5. Submission Generation
    # ==========================================
    THRESHOLD = 0.1619843989610672

    if mae < THRESHOLD:
        print(
            f"\nValidation MAE ({mae}) is below threshold ({THRESHOLD}). Generating submission..."
        )
        # generate_submission handles loading test data, inference, and saving to ./submission/submission.csv
        generate_submission()
    else:
        print(
            f"\nValidation MAE ({mae}) did not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
