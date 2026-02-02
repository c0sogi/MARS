import os
import torch
import pandas as pd
import numpy as np
import sys

# Import from provided library files
from library.config import Config
from library.train import run_training
from library.predict import generate_submission
from library.dataset import get_data_loaders
from library.model import VentilatorModel
from library.utils import masked_mae_metric, seed_everything


def main():
    # --------------------------------------------------------------------------
    # 1. Configuration & Setup
    # --------------------------------------------------------------------------
    # Ensure reproducibility
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    print("=== Starting Fast Baseline Pipeline ===")
    print(
        f"Configuration: Epochs={Config.EPOCHS}, Batch Size={Config.BATCH_SIZE}, Device={device}"
    )

    # --------------------------------------------------------------------------
    # 2. Training
    # --------------------------------------------------------------------------
    # Run training on the full dataset.
    # We rely on the library function to handle caching and training loop.
    print("\n--- Initiating Training ---")
    run_training(debug_limit=None, load_cached_data=True, save_path="model.pth")

    # --------------------------------------------------------------------------
    # 3. Validation & Metrics
    # --------------------------------------------------------------------------
    print("\n--- Performing Validation & Failure Analysis ---")

    # Re-load validation loader
    # We pass None for debug_limit to validate on the full hold-out set
    _, val_loader, _ = get_data_loaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=True,
    )

    # Load the best model saved during training
    model = VentilatorModel(config=Config).to(device)
    model_path = os.path.join(Config.WORKING_DIR, "model.pth")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Training may have failed."
        )

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Inference Loop
    all_preds = []
    all_targets = []
    all_u_out = []
    all_features = []

    with torch.no_grad():
        for X, y, u_out in val_loader:
            X = X.to(device)
            y = y.to(device)
            u_out = u_out.to(device)

            # Forward pass (we only need the final prediction)
            pred, _ = model(X)

            # Move to CPU to save GPU memory during aggregation
            all_preds.append(pred.cpu())
            all_targets.append(y.cpu())
            all_u_out.append(u_out.cpu())
            all_features.append(X.cpu())

    # Concatenate all batches
    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    u_outs = torch.cat(all_u_out)
    features = torch.cat(all_features)

    # Compute Final Metric
    # The metric function expects tensors
    val_mae = masked_mae_metric(preds, targets, u_outs)

    # Print exactly as required
    print(f"Final Validation Metric: {val_mae}")

    # --------------------------------------------------------------------------
    # 4. Failure Analysis
    # --------------------------------------------------------------------------
    # Analyze errors specifically in the inspiratory phase (u_out == 0)
    # as this is the only phase scored by the metric.

    # Create a mask for the inspiratory phase
    mask = u_outs == 0

    # Calculate absolute errors
    errors = torch.abs(preds - targets)

    # Apply mask to errors and features
    masked_errors = errors[mask].numpy()
    masked_features = features[mask].numpy()

    # Create DataFrame for correlation analysis
    # Map feature tensor columns back to names
    df_analysis = pd.DataFrame(masked_features, columns=Config.INPUT_FEATURES)
    df_analysis["error_magnitude"] = masked_errors

    # Compute correlation
    corr_matrix = df_analysis.corr()
    error_correlations = (
        corr_matrix["error_magnitude"]
        .drop("error_magnitude")
        .sort_values(ascending=False)
    )

    print("\nCorrelation between Model Error and Input Features (Inspiratory Phase):")
    print(error_correlations)

    # --------------------------------------------------------------------------
    # 5. Submission Generation
    # --------------------------------------------------------------------------
    THRESHOLD = 0.2164510190486908

    if val_mae < THRESHOLD:
        print(f"\nValidation metric {val_mae} is lower than threshold {THRESHOLD}.")
        print("Generating submission file...")
        generate_submission(model_filename="model.pth", load_cached_data=True)
    else:
        print(f"\nValidation metric {val_mae} is NOT lower than threshold {THRESHOLD}.")
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
