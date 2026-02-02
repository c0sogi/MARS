import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import library components
from library.config import Config
from library.utils import set_seed, FocalLoss
from library.data_processing import get_datasets
from library.train import run_training, validate
from library.inference import generate_predictions


def main():
    # 1. Configuration Overrides for Fast Baseline
    # Limit data and epochs to ensure execution within time limits
    Config.DEBUG = True
    Config.DEBUG_SAMPLES = 150000  # Train on a subset
    Config.EPOCHS = 3  # Reduce epochs
    Config.NUM_WORKERS = 2  # Moderate workers

    # Ensure reproducibility
    set_seed(Config.SEED)

    print("=== Starting Runfile Execution ===")

    # 2. Train the Model
    # run_training handles data loading, training loop, and saving the best model
    model = run_training()

    # 3. Final Validation Assessment
    print("\n=== Performing Final Validation ===")
    device = torch.device(Config.DEVICE)
    model.to(device)
    model.eval()

    # Reload validation dataset (fast due to caching)
    _, val_ds = get_datasets(load_cached_data=True)

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Define criterion for validation loss calculation
    criterion = FocalLoss(alpha=Config.FOCAL_ALPHA, gamma=Config.FOCAL_GAMMA)

    # Run validation to get the exact final metric
    val_loss, val_mcc, val_thresh = validate(model, val_loader, criterion, device)

    # REQUIRED OUTPUT: Print the final validation metric
    print(f"Final Validation Metric: {val_mcc}")

    # 4. Failure Analysis
    print("\n=== Failure Analysis ===")
    # We want to correlate Error Magnitude (|Prob - Target|) with Input Features

    # Collect data
    all_kin_cont = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for batch in val_loader:
            x_kin_cont, x_kin_cat, x_vis, targets = [b.to(device) for b in batch]

            # Get probabilities
            logits = model(x_kin_cont, x_kin_cat, x_vis)
            probs = torch.sigmoid(logits).squeeze()

            all_kin_cont.append(x_kin_cont.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    X_kin = np.concatenate(all_kin_cont, axis=0)
    y_true = np.concatenate(all_targets, axis=0)
    y_prob = np.concatenate(all_probs, axis=0)

    # Calculate Error Magnitude
    errors = np.abs(y_prob - y_true)

    # Reconstruct Feature Names (Logic from data_processing.py)
    feature_names = []
    shifts = range(-Config.WINDOW_SIZE, Config.WINDOW_SIZE + 1)
    track_base = [
        "x_position",
        "y_position",
        "speed",
        "acceleration",
        "orientation",
        "direction",
        "sa",
    ]

    # Base kinematic features
    for col in track_base:
        for s in shifts:
            feature_names.append(f"{col}_lag_{s}_1")
            feature_names.append(f"{col}_lag_{s}_2")

    # Derived features
    feature_names.extend(["distance", "closing_speed", "relative_angle"])

    # Calculate Correlations
    correlations = []
    # Ensure dimensions match (X_kin might have been standardized, but correlations are scale-invariant)
    if X_kin.shape[1] == len(feature_names):
        for i, name in enumerate(feature_names):
            # Pearson correlation
            if np.std(X_kin[:, i]) > 0 and np.std(errors) > 0:
                corr = np.corrcoef(X_kin[:, i], errors)[0, 1]
                correlations.append((name, corr))
            else:
                correlations.append((name, 0.0))

        # Sort by absolute correlation
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)

        print("Top 10 Features Correlated with Error Magnitude:")
        for name, corr in correlations[:10]:
            print(f"  {name}: {corr:.4f}")
    else:
        print(
            "Warning: Feature dimension mismatch. Skipping detailed feature correlation."
        )
        print(f"Expected {len(feature_names)} features, got {X_kin.shape[1]}.")

    # 5. Submission Generation
    # Threshold check
    submission_threshold = 0.6634847318478787

    if val_mcc > submission_threshold:
        print(
            f"\nValidation MCC ({val_mcc}) > Threshold ({submission_threshold}). Generating submission..."
        )
        generate_predictions(load_cached_data=True)
    else:
        print(
            f"\nValidation MCC ({val_mcc}) <= Threshold ({submission_threshold}). Skipping submission."
        )


if __name__ == "__main__":
    main()
