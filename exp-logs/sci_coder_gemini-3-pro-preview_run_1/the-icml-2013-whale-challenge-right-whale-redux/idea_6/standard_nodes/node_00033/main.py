import sys
import os
import torch
import numpy as np
import pandas as pd

# Append current working directory to ensuring library imports work correctly
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed, calculate_auc
from library.dataset import get_dataloaders
from library.trainer import Trainer


def main():
    # ==========================================
    # 1. Setup and Initialization
    # ==========================================
    # Set random seeds for reproducibility
    set_seed(Config.SEED)
    print(f"Initializing run on device: {Config.DEVICE}")

    # ==========================================
    # 2. Data Loading
    # ==========================================
    print("Loading datasets...")
    # load_cached_data=True ensures we use pre-processed .npy files if available
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # ==========================================
    # 3. Model Training
    # ==========================================
    print("Initializing Trainer...")
    trainer = Trainer()

    print("Starting training loop...")
    # The trainer handles the training loop, validation, and saving/loading the best model
    trainer.train(train_loader, val_loader)

    # ==========================================
    # 4. Validation Assessment & Failure Analysis
    # ==========================================
    print("Performing final validation and failure analysis...")

    # Ensure model is in eval mode and on the correct device
    model = trainer.model
    model.eval()

    all_targets = []
    all_preds = []

    # storage for features to correlate with error
    feat_spec_mean = []
    feat_spec_std = []
    feat_spec_max = []

    # Inference loop for validation set
    with torch.no_grad():
        for data, target in val_loader:
            data = data.to(Config.DEVICE)
            target = target.to(Config.DEVICE)

            # Forward pass
            output = model(data)
            probs = torch.sigmoid(output).cpu().numpy().flatten()
            targets_np = target.cpu().numpy().flatten()

            all_preds.extend(probs)
            all_targets.extend(targets_np)

            # Extract simple signal statistics from the spectrogram for failure analysis
            # data shape: (Batch, 1, Freq, Time)
            # Flatten spatial dims to compute stats per sample
            B = data.size(0)
            flat_data = data.view(B, -1)

            feat_spec_mean.extend(flat_data.mean(dim=1).cpu().numpy())
            feat_spec_std.extend(flat_data.std(dim=1).cpu().numpy())
            feat_spec_max.extend(flat_data.max(dim=1).values.cpu().numpy())

    # Calculate and Print Final Metric
    val_auc = calculate_auc(all_targets, all_preds)
    # Required format: Final Validation Metric: <value>
    print(f"Final Validation Metric: {val_auc}")

    # Failure Analysis: Correlation between Error and Input Features
    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    errors = np.abs(all_targets - all_preds)

    df_analysis = pd.DataFrame(
        {
            "error": errors,
            "spec_mean": feat_spec_mean,
            "spec_std": feat_spec_std,
            "spec_max": feat_spec_max,
        }
    )

    print("\n--- Failure Analysis (Correlation with Prediction Error) ---")
    # Calculate correlation of features with the error
    correlations = df_analysis.corr()["error"].drop("error")
    print(correlations)
    print("----------------------------------------------------------")

    # ==========================================
    # 5. Submission Generation
    # ==========================================
    THRESHOLD = 0.9934990421176494

    if val_auc > THRESHOLD:
        print(
            f"\nMetric ({val_auc}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )
        trainer.predict(test_loader)
    else:
        print(
            f"\nMetric ({val_auc}) does not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
