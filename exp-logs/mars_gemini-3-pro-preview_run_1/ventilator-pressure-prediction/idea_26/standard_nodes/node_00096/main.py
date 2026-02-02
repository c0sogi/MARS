import sys
import os
import torch
import numpy as np
import pandas as pd

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, get_device
from library.data import prepare_data
from library.model import WideProjectedNet
from library.train import Trainer, get_feature_names, generate_submission

# =========================================================================
# Configuration Overrides for Fast Baseline
# =========================================================================
# Limit training epochs to ensure execution finishes well within the 2-hour limit.
# 10 epochs with OneCycleLR on the full dataset (Batch Size 512) is efficient
# enough on an A100 GPU while providing sufficient convergence to attempt the threshold.
Config.EPOCHS = 10
Config.DEBUG = False


def calculate_correlation(x, y):
    """
    Compute Pearson correlation coefficient between two 1D arrays.
    Handles edge cases like constant arrays.
    """
    if len(x) != len(y):
        raise ValueError("Arrays must have same length")
    if len(x) == 0:
        return 0.0

    # Check for zero variance to avoid division by zero in correlation calculation
    if np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return 0.0

    return np.corrcoef(x, y)[0, 1]


def run_failure_analysis(model, val_loader, device, feature_names):
    """
    Evaluates the model on the validation set, computes the specific MAE metric,
    and analyzes the correlation between prediction errors and input features.
    """
    print("\n=== Failure Analysis ===")
    model.eval()

    all_preds = []
    all_targets = []
    all_inputs = []
    all_u_out = []

    # Efficient Inference Loop (No Gradients)
    with torch.no_grad():
        for X, u_out, y in val_loader:
            X = X.to(device)
            y = y.to(device)
            u_out = u_out.to(device)

            # Forward pass (we only need the final prediction head)
            pred, _ = model(X)

            # Move to CPU immediately to free up GPU memory
            all_preds.append(pred.cpu())
            all_targets.append(y.cpu())
            all_inputs.append(X.cpu())
            all_u_out.append(u_out.cpu())

    # Concatenate and flatten all batches
    # Predictions/Targets: (N_total * Seq_len,)
    preds = torch.cat(all_preds).numpy().flatten()
    targets = torch.cat(all_targets).numpy().flatten()
    u_out = torch.cat(all_u_out).numpy().flatten()

    # Inputs: (N_total * Seq_len, N_features)
    inputs = torch.cat(all_inputs).numpy()
    inputs = inputs.reshape(-1, inputs.shape[-1])

    # Filter for Inspiratory Phase (u_out == 0)
    # Using < 0.5 to safely handle floating point binary masks
    mask = u_out < 0.5

    valid_preds = preds[mask]
    valid_targets = targets[mask]
    valid_inputs = inputs[mask]

    # 1. Compute Metric (MAE on Inspiratory Phase)
    errors = np.abs(valid_preds - valid_targets)
    mae = np.mean(errors)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {mae}")

    # 2. Correlation Analysis
    print("\nCorrelation between Error Magnitude and Input Features:")
    correlations = []

    for i, feature in enumerate(feature_names):
        feat_vals = valid_inputs[:, i]
        corr = calculate_correlation(errors, feat_vals)
        correlations.append((feature, corr))

    # Sort by absolute correlation magnitude
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    # Print top features associated with error
    print(f"{'Feature':<25} | {'Correlation':<10}")
    print("-" * 40)
    for feat, corr in correlations[:10]:
        print(f"{feat:<25} | {corr:.4f}")

    return mae


def main():
    # 1. Setup Environment
    Config.setup()
    seed_everything(Config.SEED)
    device = get_device()
    print(f"Running on device: {device}")

    # 2. Data Preparation
    # Loads data, applies engineering, scales, and creates DataLoaders
    print("Preparing data pipeline...")
    train_loader, val_loader, test_loader, test_ids = prepare_data()

    # 3. Model Initialization
    # Extract feature names to configure the model's input dimension and context injection
    feature_names = get_feature_names()
    input_dim = len(feature_names)
    print(f"Input features: {input_dim}")

    model = WideProjectedNet(input_dim=input_dim, feature_names=feature_names)
    model = model.to(device)

    # 4. Training
    print(f"Starting training for {Config.EPOCHS} epochs...")
    trainer = Trainer(model, device, train_loader, val_loader)
    trainer.fit()

    # 5. Evaluation & Failure Analysis
    print("Loading best model checkpoint for analysis...")
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))

    val_mae = run_failure_analysis(model, val_loader, device, feature_names)

    # 6. Conditional Submission
    # Threshold specified in task requirements
    THRESHOLD = 0.2164510190486908

    if val_mae < THRESHOLD:
        print(
            f"\nValidation MAE ({val_mae:.6f}) < Threshold ({THRESHOLD:.6f}). Generating submission..."
        )
        generate_submission(model, test_loader, test_ids, device)
    else:
        print(
            f"\nValidation MAE ({val_mae:.6f}) >= Threshold ({THRESHOLD:.6f}). Submission skipped."
        )


if __name__ == "__main__":
    main()
