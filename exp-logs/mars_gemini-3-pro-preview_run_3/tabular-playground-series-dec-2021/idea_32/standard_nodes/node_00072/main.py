import sys
import os
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import seed_everything
from library.data import get_data_loaders
from library.train import train_model, validate, predict
from library.model import AsymmetricDCNResNet


def main():
    # 1. Setup & Configuration
    # Ensure reproducibility and device configuration
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # 2. Fast Baseline Training
    # We use a subsample of the data and fewer epochs to ensure quick execution
    # while verifying the model architecture and pipeline stability.
    print("\n=== Phase 1: Fast Baseline Training ===")

    # Configuration for fast baseline
    FAST_TRAIN_SIZE = 200000  # Train on 200k samples for speed
    FAST_EPOCHS = 5  # Limit to 5 epochs

    # Train the model
    # Note: train_model handles data loading (subsampled), training loop,
    # and saving the best model checkpoint. We disable automatic submission generation
    # here to handle it conditionally later.
    model = train_model(
        epochs=FAST_EPOCHS,
        batch_size=Config.BATCH_SIZE,
        learning_rate=Config.LEARNING_RATE,
        debug_sample_size=FAST_TRAIN_SIZE,
        create_submission=False,
    )

    # 3. Full Validation Evaluation
    # We must evaluate on the FULL validation set to get the official metric.
    # We reload data loaders with debug_sample_size=None to get full datasets.
    # We set load_cached_data=False to ensure we don't accidentally load the
    # subsampled cache generated during the training phase.
    print("\n=== Phase 2: Full Validation Evaluation ===")

    _, val_loader, test_loader = get_data_loaders(
        batch_size=Config.BATCH_SIZE,
        load_cached_data=False,  # Force reprocessing of full dataset
        debug_sample_size=None,  # Full dataset
    )

    criterion = torch.nn.CrossEntropyLoss()

    # Execute validation
    # This uses the best weights loaded into 'model' by train_model
    val_loss, val_acc = validate(model, val_loader, criterion, device)

    # REQUIRED: Print the final validation metric in the specific format
    print(f"Final Validation Metric: {val_acc}")

    # 4. Failure Analysis
    # Calculate and print the correlation between the model's error magnitude
    # and the input features to identify systematic error patterns.
    print("\n=== Phase 3: Failure Analysis ===")
    model.eval()

    all_errors = []
    all_inputs = []

    # Iterate through the full validation set to collect predictions and inputs
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            # Define Error Magnitude: 1.0 if incorrect, 0.0 if correct
            # This binary error serves as the magnitude for classification failure analysis
            errors = (predicted != targets).float().cpu().numpy()

            # Collect input features (preprocessed/scaled)
            input_data = inputs.cpu().numpy()

            all_errors.append(errors)
            all_inputs.append(input_data)

    # Concatenate all batches
    all_errors = np.concatenate(all_errors)
    all_inputs = np.concatenate(all_inputs, axis=0)

    # Compute Pearson correlation between Error and each Feature
    n_features = all_inputs.shape[1]
    feature_correlations = []

    for i in range(n_features):
        feat_col = all_inputs[:, i]
        # Handle constant features (std=0) to avoid division by zero
        if np.std(feat_col) < 1e-9:
            corr = 0.0
        else:
            corr = np.corrcoef(feat_col, all_errors)[0, 1]
        feature_correlations.append((i, corr))

    # Sort features by absolute correlation strength
    feature_correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 10 Features correlated with Prediction Error:")
    print(f"{'Feat Idx':<10} {'Correlation':<15}")
    print("-" * 25)
    for idx, corr in feature_correlations[:10]:
        print(f"{idx:<10} {corr:.6f}")

    # 5. Submission Generation
    # Generate predictions for the test set IF AND ONLY IF the validation metric
    # exceeds the specified threshold.
    THRESHOLD = 0.9626291666666666

    print("\n=== Phase 4: Submission Check ===")
    if val_acc > THRESHOLD:
        print(
            f"Validation Accuracy ({val_acc:.8f}) > Threshold ({THRESHOLD}). Generating submission..."
        )
        # Use the full test_loader loaded in Phase 2
        predict(model, test_loader, device, debug_sample_size=None)
    else:
        print(
            f"Validation Accuracy ({val_acc:.8f}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
