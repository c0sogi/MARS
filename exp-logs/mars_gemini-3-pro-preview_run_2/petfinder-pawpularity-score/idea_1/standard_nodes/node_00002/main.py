import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr

# Import from provided library files
from library.config import Config
from library.dataset import get_dataloaders
from library.model import PawpularityResNet18
from library.engine import train_model, predict_and_submit


def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def perform_failure_analysis(model, val_loader, device):
    """
    Performs inference on validation set, calculates RMSE, and analyzes correlations
    between error magnitude and features.
    """
    print("\nStarting Failure Analysis on Validation Set...")
    model.eval()

    all_preds = []
    all_targets = []

    # Efficient inference
    with torch.no_grad():
        for images, metadata, targets in val_loader:
            images = images.to(device)
            metadata = metadata.to(device)

            outputs = model(images, metadata)

            # Collect results
            all_preds.extend(outputs.cpu().numpy().flatten())
            all_targets.extend(targets.numpy().flatten())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Clip predictions to valid range [1, 100] for final metric calculation logic
    # (Though raw RMSE is often calculated on raw outputs, clipping is part of the task logic)
    all_preds_clipped = np.clip(all_preds, 1.0, 100.0)

    # Calculate RMSE
    mse = np.mean((all_preds_clipped - all_targets) ** 2)
    rmse = np.sqrt(mse)

    # REQUIRED OUTPUT
    print(f"Final Validation Metric: {rmse}")

    # Calculate Errors
    errors = np.abs(all_preds_clipped - all_targets)

    # Load validation metadata to correlate features with error
    # We can access the dataframe directly from the dataset
    val_df = val_loader.dataset.df.copy()

    # Ensure alignment (DataLoader shuffle is False for val, but good to be safe)
    # The loop above iterated sequentially, so indices match if shuffle=False
    val_df["error"] = errors
    val_df["prediction"] = all_preds_clipped

    # Calculate correlations
    print("\nCorrelation between Absolute Error and Features:")
    feature_cols = Config.METADATA_COLS + ["Pawpularity"]

    correlations = []
    for col in feature_cols:
        if col in val_df.columns:
            # Handle potential constant columns which produce NaN correlation
            if val_df[col].std() == 0:
                corr = 0.0
            else:
                corr, _ = pearsonr(val_df[col], val_df["error"])
            correlations.append((col, corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    for feature, corr in correlations:
        print(f"  {feature:<15}: {corr:.4f}")


def run():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # 2. Train Model
    # This uses the library engine to train and save the best model to Config.MODEL_PATH
    print("\n=== Starting Training Phase ===")
    best_val_rmse = train_model(debug=False)
    print(f"Training finished. Best RMSE reported by engine: {best_val_rmse}")

    # 3. Load Best Model for Analysis
    print("\n=== Loading Best Model for Analysis ===")
    model = PawpularityResNet18().to(device)
    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError("Model file not found after training.")

    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))

    # Get dataloaders (we only need val here, but the function returns 3)
    _, val_loader, _ = get_dataloaders(debug=False)

    # 4. Validate & Failure Analysis
    perform_failure_analysis(model, val_loader, device)

    # 5. Submission
    print("\n=== Generating Submission ===")
    if best_val_rmse < 19.316452026367188:
        print(
            f"Validation RMSE {best_val_rmse:.4f} is better than baseline. Generating submission..."
        )
        predict_and_submit(debug=False)
    else:
        print(
            f"Validation RMSE {best_val_rmse:.4f} did not improve upon baseline (19.3165). Skipping submission."
        )

    print("Pipeline completed successfully.")


if __name__ == "__main__":
    run()
