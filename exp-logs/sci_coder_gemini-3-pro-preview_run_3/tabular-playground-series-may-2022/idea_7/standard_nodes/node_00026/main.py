import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, get_device
from library.data import prepare_data
from library.engine import run_training, validate, generate_submission


def main():
    # 1. Setup and Configuration
    set_seed()
    device = get_device()

    # Optimize configuration for A100 GPU and time constraints
    # Increasing batch size speeds up training significantly on A100
    Config.BATCH_SIZE = 2048
    # We keep epochs at 30 to ensure convergence to the high AUC threshold
    # With BS=2048, this will still be very fast (< 20 mins)
    Config.EPOCHS = 30

    print("Configuration optimized for fast execution:")
    print(f"Batch Size: {Config.BATCH_SIZE}")
    print(f"Epochs: {Config.EPOCHS}")

    # 2. Data Preparation
    # We use the full dataset (debug=False) to ensure we can hit the target AUC
    train_loader, val_loader, test_loader = prepare_data(
        load_cached_data=True, debug=False
    )

    # 3. Training
    # run_training handles the loop, early stopping, and returns the model with best weights loaded
    model = run_training(train_loader, val_loader, test_loader)

    # 4. Final Validation Assessment
    print("\nPerforming Final Validation...")
    criterion = nn.BCEWithLogitsLoss()

    # We re-run validation to ensure we have the exact metric for the loaded best model
    val_loss, val_auc = validate(model, val_loader, criterion, device)

    # REQUIRED: Print the final validation metric in the specified format
    print(f"Final Validation Metric: {val_auc}")

    # 5. Failure Analysis
    print("\nPerforming Failure Analysis...")
    model.eval()

    all_targets = []
    all_preds = []
    all_cont_features = []

    # Collect validation data and predictions
    with torch.no_grad():
        for batch in val_loader:
            x_cat = batch["x_cat"].to(device)
            x_cont = batch["x_cont"].to(device)
            targets = batch["target"]

            logits = model(x_cat, x_cont)
            probs = torch.sigmoid(logits)

            all_targets.append(targets.numpy())
            all_preds.append(probs.cpu().numpy())
            all_cont_features.append(batch["x_cont"].numpy())

    all_targets = np.concatenate(all_targets).flatten()
    all_preds = np.concatenate(all_preds).flatten()
    all_cont_features = np.vstack(all_cont_features)

    # Calculate Error Magnitude
    errors = np.abs(all_targets - all_preds)

    # Calculate correlations between error and continuous features
    feature_names = Config.CONT_FEATURES
    correlations = []

    print("Correlation between Model Error and Input Features:")
    for i, feature_name in enumerate(feature_names):
        if i < all_cont_features.shape[1]:
            feat_values = all_cont_features[:, i]
            # Compute correlation
            corr = np.corrcoef(feat_values, errors)[0, 1]
            correlations.append((feature_name, corr))

    # Sort by absolute correlation strength
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    # Print top 10 correlations
    for name, corr in correlations[:10]:
        print(f"{name}: {corr:.6f}")

    # 6. Submission Generation
    # Threshold defined in the task
    THRESHOLD = 0.9971550270448856

    if val_auc > THRESHOLD:
        print(
            f"\nValidation AUC ({val_auc}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission(model, test_loader)
    else:
        print(
            f"\nValidation AUC ({val_auc}) does not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
