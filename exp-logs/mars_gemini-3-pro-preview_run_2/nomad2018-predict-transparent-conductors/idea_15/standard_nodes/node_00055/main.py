import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_log_error

# Import library modules
from library.config import Config
from library.utils import set_seed, Standardizer
from library.data import get_dataloaders
from library.model import HASCNet
from library.train import run_training, generate_submission

# -------------------------------------------------------------------------
# Configuration Overrides for Fast Baseline
# -------------------------------------------------------------------------
# Override parameters to ensure execution finishes within the time limit
Config.NUM_EPOCHS = 20
Config.PATIENCE = 5
# Config.DEBUG_DATA_LIMIT = None # Use full data for better results, it's small enough


def calculate_metric(y_true, y_pred):
    """
    Calculates Column-wise Root Mean Squared Logarithmic Error.
    """
    # Ensure non-negative for log
    y_pred = np.maximum(y_pred, 0)
    y_true = np.maximum(y_true, 0)

    # Calculate MSLE for each column
    msle = mean_squared_log_error(y_true, y_pred, multioutput="raw_values")

    # Take sqrt to get RMSLE per column
    rmsle = np.sqrt(msle)

    # Mean of column-wise RMSLE
    return np.mean(rmsle)


def main():
    # Set seed for reproducibility
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    print("==================================================")
    print("Step 1: Model Training")
    print("==================================================")

    # Run training (handles data loading, scaling, training loop, checkpointing)
    run_training(load_cached_data=True)

    print("\n==================================================")
    print("Step 2: Validation & Failure Analysis")
    print("==================================================")

    # Load the best model
    model = HASCNet().to(device)
    checkpoint_path = os.path.join(Config.WORKING_DIR, "checkpoints", "best_model.pth")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Get Validation Loader
    # We ignore train/test loaders here, just need val
    # Note: get_dataloaders handles the scaler logic internally/caches it
    _, val_loader, _, _ = get_dataloaders(load_cached_data=True)

    # Load the scaler used during training to inverse transform predictions
    target_scaler = Standardizer(device=device)
    scaler_path = os.path.join(Config.WORKING_DIR, "target_scaler.npz")
    target_scaler.load(scaler_path)

    # Inference Loop
    val_preds = []
    val_targets = []
    val_global_features = []

    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)

            # Predict
            out = model(data)

            # Inverse transform
            out_inv = target_scaler.inverse_transform(out)
            y_inv = target_scaler.inverse_transform(data.y)

            val_preds.append(out_inv.cpu().numpy())
            val_targets.append(y_inv.cpu().numpy())
            val_global_features.append(data.global_x.cpu().numpy())

    # Concatenate
    y_pred = np.concatenate(val_preds, axis=0)
    y_true = np.concatenate(val_targets, axis=0)
    global_x = np.concatenate(val_global_features, axis=0)

    # Calculate Metric
    metric = calculate_metric(y_true, y_pred)
    print(f"Final Validation Metric: {metric}")

    # Failure Analysis
    # Calculate mean absolute error per sample
    # y_true shape: (N, 2), y_pred shape: (N, 2)
    errors = np.abs(y_true - y_pred)
    mean_error = np.mean(errors, axis=1)  # (N,)

    print("\nCorrelation between Error Magnitude and Global Features:")
    feature_names = [
        "Lattice_a",
        "Lattice_b",
        "Lattice_c",
        "Angle_alpha",
        "Angle_beta",
        "Angle_gamma",
        "Comp_Al",
        "Comp_Ga",
        "Comp_In",
    ]

    # Ensure global_x has 9 columns as expected
    if global_x.shape[1] == 9:
        for i, name in enumerate(feature_names):
            corr = np.corrcoef(mean_error, global_x[:, i])[0, 1]
            print(f"  {name:<15}: {corr:.4f}")
    else:
        print(f"  Global features shape mismatch. Expected 9, got {global_x.shape[1]}")

    print("\n==================================================")
    print("Step 3: Submission Generation")
    print("==================================================")

    threshold = 0.05085437756413089
    if metric < threshold:
        print(f"Validation metric {metric} is below threshold {threshold}.")
        print("Generating submission file...")
        generate_submission(load_cached_data=True)
    else:
        print(f"Validation metric {metric} is NOT below threshold {threshold}.")
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
