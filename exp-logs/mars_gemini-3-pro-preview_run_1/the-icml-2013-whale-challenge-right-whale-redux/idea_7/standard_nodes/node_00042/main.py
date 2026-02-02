import os
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import set_seed, calculate_auc
from library.trainer import train_model, predict_and_submit
from library.model import MultiResResNet34CRNN
from library.data_loader import get_dataloaders


def perform_failure_analysis(model, val_loader, device):
    """
    Runs inference on validation set, calculates AUC, and analyzes error correlations.
    """
    model.eval()

    all_targets = []
    all_probs = []
    all_errors = []

    # Features for correlation analysis
    feat_means = []
    feat_maxs = []

    print("Running Failure Analysis on Validation Set...")

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device).view(-1, 1)

            # Forward pass
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)

            # Move to CPU
            probs_np = probs.cpu().numpy().flatten()
            targets_np = targets.cpu().numpy().flatten()
            inputs_np = inputs.cpu().numpy()

            # Calculate absolute errors
            errors = np.abs(targets_np - probs_np)

            # Collect data
            all_targets.extend(targets_np)
            all_probs.extend(probs_np)
            all_errors.extend(errors)

            # Extract simple signal features from the spectrogram tensors
            # inputs shape: (Batch, 3, 128, Time)
            # Mean intensity per sample
            feat_means.extend(np.mean(inputs_np, axis=(1, 2, 3)))
            # Max intensity per sample
            feat_maxs.extend(np.max(inputs_np, axis=(1, 2, 3)))

    # Convert to numpy arrays
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    all_errors = np.array(all_errors)
    feat_means = np.array(feat_means)
    feat_maxs = np.array(feat_maxs)

    # 1. Calculate Metric
    auc = calculate_auc(all_targets, all_probs)

    # 2. Failure Analysis Correlations
    # We use numpy corrcoef which returns a matrix [[1, r], [r, 1]]
    corr_mean = np.corrcoef(all_errors, feat_means)[0, 1]
    corr_max = np.corrcoef(all_errors, feat_maxs)[0, 1]
    corr_label = np.corrcoef(all_errors, all_targets)[0, 1]

    print("\n--- Failure Analysis Report ---")
    print(f"Correlation (Error vs Spec Mean): {corr_mean:.4f}")
    print(f"Correlation (Error vs Spec Max):  {corr_max:.4f}")
    print(f"Correlation (Error vs Label):     {corr_label:.4f}")
    print("-------------------------------\n")

    return auc


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Train Model
    # We limit epochs to 15 to ensure execution within 2 hours while allowing convergence.
    # The A100 is fast enough to handle 15 epochs on ~18k samples quickly.
    print("Initializing Training Pipeline...")
    best_model_path = train_model(debug=False, epochs=15, load_cached_data=True)

    # 3. Load Best Model for Validation
    print(f"Loading best model from {best_model_path} for evaluation...")
    model = MultiResResNet34CRNN(pretrained=False).to(device)
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    # Get validation loader (uses cached data)
    _, val_loader, _ = get_dataloaders(load_cached_data=True, debug_subset=False)

    # 4. Validation & Failure Analysis
    val_auc = perform_failure_analysis(model, val_loader, device)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {val_auc}")

    # 5. Submission
    # Threshold defined in task description
    THRESHOLD = 0.9934990421176494

    if val_auc > THRESHOLD:
        print(
            f"Validation score ({val_auc}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )
        predict_and_submit(best_model_path, debug=False, load_cached_data=True)
    else:
        print(
            f"Validation score ({val_auc}) did not exceed threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
