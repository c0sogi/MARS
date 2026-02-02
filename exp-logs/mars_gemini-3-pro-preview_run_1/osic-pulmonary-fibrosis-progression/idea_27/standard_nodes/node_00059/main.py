import sys
import os
import torch
import numpy as np
import warnings
import pandas as pd

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

# Import provided library modules
from library.config import Config, seed_everything
from library import train, model, data, utils


def main():
    # 1. Setup & Configuration
    # Ensure reproducibility
    seed_everything(Config.SEED)

    # Adjust Config for fast baseline execution as requested
    # 1100 samples is small, but we limit epochs to ensure runtime < 2 hours
    Config.EPOCHS = 30

    print("=== Starting Runfile Pipeline ===")

    # 2. Training
    # Execute the training routine provided in library.train
    # This handles data loading, training loop, validation, and checkpointing
    print("\n[Step 1] Training Model...")
    train.run_training()

    # 3. Validation & Metrics
    print("\n[Step 2] Performing Validation & Failure Analysis...")

    # Load Validation Data (using cache)
    _, val_loader, _ = data.get_dataloaders(load_cached_data=True)

    # Initialize Model and Load Best Checkpoint
    device = torch.device(Config.DEVICE)
    net = model.VCDAN().to(device)

    if not os.path.exists(Config.MODEL_SAVE_PATH):
        print(f"CRITICAL ERROR: Model checkpoint not found at {Config.MODEL_SAVE_PATH}")
        return

    net.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    net.eval()

    # Containers for analysis
    all_targets = []
    all_preds = []
    all_confs = []
    all_tabular = []

    # Inference Loop (No Grad)
    with torch.no_grad():
        for batch in val_loader:
            # Move data to device
            img_ax = batch["img_ax"].to(device)
            img_cor = batch["img_cor"].to(device)
            tabular = batch["tabular"].to(device)
            target = batch["target"].to(device)

            # Forward pass
            fvc_pred, conf_pred = net(img_ax, img_cor, tabular)

            # Collect results (move to CPU numpy)
            all_targets.append(target.cpu().numpy())
            all_preds.append(fvc_pred.cpu().numpy())
            all_confs.append(conf_pred.cpu().numpy())
            all_tabular.append(tabular.cpu().numpy())

    # Concatenate batches
    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    y_conf = np.concatenate(all_confs)
    X_tab = np.concatenate(all_tabular)

    # Calculate Final Metric
    final_metric = utils.calculate_metric(y_true, y_pred, y_conf)
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    # Calculate absolute error magnitude
    abs_errors = np.abs(y_true - y_pred)

    # Feature names corresponding to the tabular vector construction in library.data
    # Order: [Age, Sex, Smoking, Percent, Baseline_FVC, Rel_Week]
    feature_names = [
        "Age",
        "Sex",
        "SmokingStatus",
        "Percent",
        "Baseline_FVC",
        "Relative_Week",
    ]

    print("\nFailure Analysis - Correlation with Absolute Error:")
    for i, feat_name in enumerate(feature_names):
        feat_values = X_tab[:, i]

        # Calculate correlation (handle constant features to avoid NaN)
        if np.std(feat_values) < 1e-9:
            corr = 0.0
        else:
            corr = np.corrcoef(abs_errors, feat_values)[0, 1]

        print(f"  {feat_name}: {corr:.8f}")

    # 5. Submission Generation
    THRESHOLD = -6.510164260864258

    if final_metric > THRESHOLD:
        print(
            f"\n[Step 3] Metric ({final_metric}) meets threshold ({THRESHOLD}). Generating submission..."
        )
        # Use the provided submission generation function which handles test loading and formatting
        model.generate_submission()
    else:
        print(
            f"\n[Step 3] Metric ({final_metric}) did not meet threshold ({THRESHOLD}). Submission skipped."
        )

    print("\n=== Pipeline Complete ===")


if __name__ == "__main__":
    main()
