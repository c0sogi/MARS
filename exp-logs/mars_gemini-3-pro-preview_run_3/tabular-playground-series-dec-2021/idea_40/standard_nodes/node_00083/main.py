import sys
import os
import numpy as np
import torch
import torch.nn as nn

# Ensure library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.data_utils import get_dataloaders
from library.train_utils import run_training, validate, predict

# ==========================================
# Configuration Overrides for Fast Baseline
# ==========================================
# Adjust settings to ensure execution within time limits while maximizing A100 utilization
Config.EPOCHS = 20  # Reduced from 60 to ensure completion < 1 hour
Config.BATCH_SIZE = 10240  # Increased to leverage A100 40GB VRAM
Config.NUM_WORKERS = 8  # Increased to utilize available vCPUs


def perform_failure_analysis(model, val_loader, device):
    """
    Analyzes model failure modes by correlating error magnitude with input features.
    """
    print("\nRunning Failure Analysis...")
    model.eval()

    all_inputs = []
    all_errors = []

    # Disable gradients for inference
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass (Primary head)
            logits, _ = model(inputs)
            _, preds = torch.max(logits, 1)

            # Calculate Error (1.0 if incorrect, 0.0 if correct)
            errors = (preds != labels).float().cpu().numpy()

            # Collect inputs (move to CPU)
            all_inputs.append(inputs.cpu().numpy())
            all_errors.append(errors)

    # Concatenate all batches
    X_val = np.concatenate(all_inputs, axis=0)
    errors_val = np.concatenate(all_errors, axis=0)

    n_features = X_val.shape[1]
    correlations = []

    # Calculate Point-Biserial Correlation for each feature
    # (Correlation between continuous feature and binary error)
    for i in range(n_features):
        feat_vals = X_val[:, i]
        # Handle constant columns to avoid division by zero
        if np.std(feat_vals) < 1e-9:
            corr = 0.0
        else:
            corr = np.corrcoef(feat_vals, errors_val)[0, 1]
            # Handle NaN result from corrcoef
            if np.isnan(corr):
                corr = 0.0
        correlations.append((i, corr))

    # Sort by absolute correlation magnitude
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 10 Features correlated with Error Magnitude:")
    print(f"{'Feature Index':<15} {'Correlation':<15}")
    print("-" * 35)
    for idx, corr in correlations[:10]:
        print(f"{idx:<15} {corr:<15.6f}")


def main():
    # 1. Reproducibility
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Config.SEED)

    # 2. Data Loading
    print("Loading and processing data...")
    # load_cached_data=True allows using pre-processed .npy files if they exist
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        load_cached_data=True
    )

    # 3. Training
    print("Initiating training pipeline...")
    # run_training handles the loop, validation, and saving best model
    model = run_training(train_loader, val_loader)

    # 4. Final Validation Assessment
    print("Performing final validation...")
    criterion = nn.CrossEntropyLoss()
    # Explicitly calculate metric on the full validation set
    val_loss, val_acc = validate(model, val_loader, criterion, Config.DEVICE)

    # REQUIRED: Print the final validation metric in full precision
    print(f"Final Validation Metric: {val_acc}")

    # 5. Failure Analysis
    perform_failure_analysis(model, val_loader, Config.DEVICE)

    # 6. Submission Generation
    # Strict threshold check
    THRESHOLD = 0.9626291666666666

    if val_acc > THRESHOLD:
        print(
            f"Validation metric ({val_acc}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )
        predict(model, test_loader, test_ids)
    else:
        print(
            f"Validation metric ({val_acc}) does not exceed threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
