import os
import sys
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Ensure library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.trainer import Trainer
from library.data import get_dataloaders
from library.utils import set_seed


class RunConfig(Config):
    """
    Custom configuration for the runfile execution.
    Optimizes for speed and sets correct output paths.
    """

    # Reduce epochs for a fast baseline while ensuring convergence
    EPOCHS = 15

    # Set the required submission path
    SUBMISSION_PATH = "./submission/submission.csv"

    # Ensure the working directory matches the cached data location
    # Inherits WORKING_DIR from Config (./working/idea_6) which is correct


def main():
    # 1. Setup
    # Ensure submission directory exists
    os.makedirs(os.path.dirname(RunConfig.SUBMISSION_PATH), exist_ok=True)

    print("Initializing Trainer...")
    trainer = Trainer(RunConfig)

    # 2. Training
    print("Starting Training...")
    # Load cached data to speed up initialization
    trainer.fit(load_cached_data=True)

    # 3. Validation & Metric Calculation
    print("Starting Validation and Failure Analysis...")
    _, val_loader = get_dataloaders(RunConfig, load_cached_data=True)

    model = trainer.model
    model.eval()
    device = trainer.device

    all_preds = []
    all_targets = []
    all_u_out = []
    all_X = []

    # Inference loop on validation set
    with torch.no_grad():
        for X, y, u_out in val_loader:
            X = X.to(device)

            # Forward pass
            preds = model(X)

            # Collect data (move to CPU)
            all_preds.append(preds.squeeze(-1).cpu().numpy())
            all_targets.append(y.cpu().numpy())
            all_u_out.append(u_out.cpu().numpy())
            all_X.append(X.cpu().numpy())

    # Concatenate batches
    all_preds = np.concatenate(all_preds).flatten()
    all_targets = np.concatenate(all_targets).flatten()
    all_u_out = np.concatenate(all_u_out).flatten()

    # Reshape X to (Total_Steps, N_Features)
    # X comes out as (Batch, Seq, Feat), we flatten Batch and Seq
    all_X = np.concatenate(all_X).reshape(-1, RunConfig.INPUT_DIM)

    # Calculate Metric (MAE on Inspiratory Phase)
    # Mask: 1 for inspiratory (u_out == 0), 0 for expiratory
    # Note: u_out is float/int, so we check equality to 0
    insp_mask = all_u_out == 0

    if np.sum(insp_mask) == 0:
        mae = 0.0
    else:
        mae = np.mean(np.abs(all_preds[insp_mask] - all_targets[insp_mask]))

    print(f"Final Validation Metric: {mae}")

    # 4. Failure Analysis
    # Calculate absolute errors
    errors = np.abs(all_preds - all_targets)

    # Filter for inspiratory phase (metric focus)
    insp_errors = errors[insp_mask]
    insp_X = all_X[insp_mask]

    print("Failure Analysis (Correlation with Error Magnitude):")
    features = RunConfig.FEATURES

    for i, feat in enumerate(features):
        # Calculate Pearson correlation between feature values and error magnitude
        if len(insp_errors) > 1:
            corr, _ = pearsonr(insp_errors, insp_X[:, i])
            print(f"  {feat}: {corr:.4f}")
        else:
            print(f"  {feat}: N/A (Insufficient data)")

    # 5. Submission
    THRESHOLD = 0.4283660650253296

    if mae < THRESHOLD:
        print(
            f"Validation metric {mae} is lower than threshold {THRESHOLD}. Generating submission..."
        )
        trainer.predict(load_cached_data=True)
    else:
        print(
            f"Validation metric {mae} is not lower than threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
