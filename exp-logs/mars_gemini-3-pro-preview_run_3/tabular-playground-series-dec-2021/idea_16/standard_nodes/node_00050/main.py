import sys
import os
import shutil
import numpy as np
import torch
import pandas as pd

# Ensure library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config, TrainConfig, DataConfig
from library.utils import seed_everything, get_device
from library.data import get_dataloaders
from library.train import run_training, validate, generate_submission
from library.model import ParallelLowRankDCNResNet


def main():
    # 1. Setup & Configuration
    seed_everything(Config.SEED)
    device = get_device()
    print(f"Orchestrating pipeline on device: {device}")

    # 2. Data Loading
    # Clear cache to ensure schema changes are picked up
    if os.path.exists(Config.CACHE_DIR):
        shutil.rmtree(Config.CACHE_DIR)

    # Loading cached data for efficiency
    print("Initializing data loaders...")
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        batch_size=TrainConfig.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=True,
    )

    # 3. Training
    # run_training encapsulates the training loop, validation, and early stopping.
    # It returns the model loaded with the best weights found during training.
    print("Starting training phase...")
    model = run_training(train_loader, val_loader)

    # 4. Final Validation Assessment
    # We perform a dedicated validation pass to ensure we capture the exact metric of the final model.
    print("Performing final validation assessment...")
    criterion = torch.nn.CrossEntropyLoss()
    val_loss, val_acc = validate(model, val_loader, criterion, device)

    # REQUIRED: Print the final validation metric in the specific format
    print(f"Final Validation Metric: {val_acc}")

    # 5. Failure Analysis
    print("\nRunning Failure Analysis on Validation Set...")
    model.eval()

    val_inputs_list = []
    val_preds_list = []
    val_targets_list = []

    # Collect predictions and inputs
    # Note: We iterate through val_loader again to get inputs for correlation
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            _, predicted = outputs.max(1)

            val_inputs_list.append(inputs.cpu().numpy())
            val_preds_list.append(predicted.cpu().numpy())
            val_targets_list.append(targets.cpu().numpy())

    # Concatenate all batches
    val_inputs = np.concatenate(val_inputs_list, axis=0)
    val_preds = np.concatenate(val_preds_list, axis=0)
    val_targets = np.concatenate(val_targets_list, axis=0)

    # Calculate Error Vector (1 = Error, 0 = Correct)
    errors = (val_preds != val_targets).astype(int)
    error_rate = errors.mean()
    print(f"Overall Error Rate: {error_rate:.6f}")

    # Compute Correlation between Features and Error
    # Reconstruct feature names
    feature_names = DataConfig.CONT_COLS + DataConfig.BINARY_COLS

    correlations = []
    # Iterate over columns
    for i in range(val_inputs.shape[1]):
        feature_col = val_inputs[:, i]

        # Skip constant columns to avoid division by zero in correlation
        if np.std(feature_col) == 0 or np.std(errors) == 0:
            corr = 0.0
        else:
            # Pearson correlation
            corr = np.corrcoef(feature_col, errors)[0, 1]

        correlations.append((feature_names[i], corr))

    # Sort by magnitude of correlation (absolute value) to find strongest signals
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 10 Features Correlated with Prediction Errors:")
    print(f"{'Feature':<35} {'Correlation':<12}")
    print("-" * 50)
    for name, corr in correlations[:10]:
        print(f"{name:<35} {corr:.6f}")

    # 6. Submission Generation
    # Threshold defined in task requirements
    THRESHOLD = 0.9625041666666667

    if val_acc > THRESHOLD:
        print(
            f"\nValidation metric ({val_acc:.8f}) exceeds threshold ({THRESHOLD:.8f})."
        )
        print("Generating submission file...")
        generate_submission(model, test_loader, test_ids)
    else:
        print(
            f"\nValidation metric ({val_acc:.8f}) does NOT meet threshold ({THRESHOLD:.8f})."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
