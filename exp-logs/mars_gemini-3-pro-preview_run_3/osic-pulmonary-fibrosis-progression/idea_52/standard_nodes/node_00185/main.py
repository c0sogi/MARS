import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings

# Import library modules
from library.config import Config
from library.utils import seed_everything, inverse_transform
from library.data import get_dataloaders
from library.model import AILRNet
from library.engine import fit, predict, evaluate
from library.loss import LaplaceLogLikelihoodLoss

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def analyze_failure(model, val_loader, device):
    """
    Performs failure analysis on the validation set.
    Calculates correlation between absolute error and input features.
    """
    model.eval()
    all_mus = []
    all_sigmas = []
    all_targets = []

    # Get the dataframe from the dataset
    val_df = val_loader.dataset.df.copy()

    with torch.no_grad():
        for batch in val_loader:
            images, restricted, context, targets = [x.to(device) for x in batch]
            mu, sigma = model(images, restricted, context)

            all_mus.append(mu.cpu().numpy())
            all_sigmas.append(sigma.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_mus = np.concatenate(all_mus)
    all_sigmas = np.concatenate(all_sigmas)
    all_targets = np.concatenate(all_targets)

    # Inverse transform to get predictions in ml
    # Note: inverse_transform applies max(sigma, 70) and unscales using global stats
    pred_fvc, _ = inverse_transform(all_mus, all_sigmas)

    # Inverse transform targets (standardized -> ml)
    # We pass a dummy sigma as we only care about the mean (value)
    true_fvc, _ = inverse_transform(all_targets, np.zeros_like(all_targets))

    # Calculate Absolute Error
    errors = np.abs(true_fvc - pred_fvc)

    # Add error to dataframe for correlation analysis
    # Note: val_loader with shuffle=False preserves dataset order
    if len(errors) == len(val_df):
        val_df["Error"] = errors

        # Calculate correlations
        numeric_features = ["Weeks", "Baseline_FVC", "Age", "Percent"]
        print("\nFailure Analysis Correlations:")
        for feat in numeric_features:
            if feat in val_df.columns:
                corr = val_df[feat].corr(val_df["Error"])
                print(f"Correlation (Error vs {feat}): {corr:.6f}")
    else:
        print(
            "Warning: Mismatch between prediction count and dataframe length. Skipping detailed correlation analysis."
        )


def main():
    # 1. Configuration and Setup
    # Override EPOCHS for fast baseline execution as per requirements
    Config.EPOCHS = 15

    Config.setup()
    seed_everything(Config.SEED)

    # 2. Data Loading
    # Uses cached data if available, otherwise processes it
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders()

    # 3. Model Initialization
    print(f"Initializing {Config.MODEL_NAME} on {Config.DEVICE}...")
    model = AILRNet().to(Config.DEVICE)

    # 4. Training
    # fit() handles the training loop, validation, and saving the best model
    fit(model, train_loader, val_loader, Config.DEVICE, epochs=Config.EPOCHS)

    # 5. Load Best Model
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        print("Loading best model checkpoint...")
        model.load_state_dict(torch.load(best_model_path, map_location=Config.DEVICE))
    else:
        print("Warning: Best model checkpoint not found. Using current model state.")

    # 6. Final Validation
    loss_fn = LaplaceLogLikelihoodLoss()
    val_loss, val_metric = evaluate(model, val_loader, Config.DEVICE, loss_fn)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {val_metric}")

    # 7. Failure Analysis
    analyze_failure(model, val_loader, Config.DEVICE)

    # 8. Submission
    # Threshold from instructions
    THRESHOLD = -6.573619738753321

    if val_metric > THRESHOLD:
        print(f"Metric {val_metric} > {THRESHOLD}. Generating submission...")
        predict(model, test_loader, Config.DEVICE)
    else:
        print(f"Metric {val_metric} <= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
