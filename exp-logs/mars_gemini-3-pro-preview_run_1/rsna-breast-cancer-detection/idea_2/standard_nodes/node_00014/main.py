import os
import sys
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import warnings
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import from provided library files
from library.config import Config
from library.utils import seed_everything
from library.data import get_dataloaders
from library.model import MultiTaskEfficientNet, predict_and_submit, probabilistic_f1
from library.train import train_epoch, validate_epoch

# Suppress warnings
warnings.filterwarnings("ignore")


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Override Config for Fast Baseline
    # 3 Epochs is sufficient to verify the pipeline and likely beat the weak baseline
    # without exceeding the 2-hour soft limit on an A100.
    EPOCHS = 5
    THRESHOLD = 0.04437665641307831

    print(f"Starting Optimized Late Fusion Run (Epochs={EPOCHS})...")
    print(f"Device: {device}")

    # 2. Data Loading
    print("Loading Data...")
    # Using cached data to speed up startup
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=True, debug=False
    )

    # 3. Model Initialization
    print(f"Initializing Model: {Config.BACKBONE}")
    model = MultiTaskEfficientNet(Config.BACKBONE, pretrained=True)
    model.to(device)

    # 4. Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=Config.MIN_LR)

    # 5. Training Loop
    best_pf1 = -1.0
    best_model_path = Config.MODEL_SAVE_PATH

    # Ensure directory exists
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

    for epoch in range(EPOCHS):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)

        # Validate
        val_pf1 = validate_epoch(model, val_loader, device)

        # Scheduler Step
        scheduler.step()

        # Save Best
        if val_pf1 > best_pf1:
            print(f"New Best pF1: {val_pf1} (was {best_pf1}). Saving model.")
            best_pf1 = val_pf1
            torch.save(model.state_dict(), best_model_path)

    # 6. Final Evaluation & Failure Analysis
    print("\n==== Final Evaluation ====")
    if os.path.exists(best_model_path):
        print("Loading best model for analysis...")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print("Warning: No model saved. Using current weights.")

    # Re-run validation to get predictions for analysis and confirm metric
    model.eval()
    all_cancer_probs = []
    all_cancer_targets = []

    with torch.no_grad():
        for (images, metas), cancer_targets, density_targets in val_loader:
            images = images.to(device)
            metas = metas.to(device)
            cancer_logits, _ = model((images, metas))

            probs = torch.sigmoid(cancer_logits).cpu().numpy().flatten()
            all_cancer_probs.extend(probs)
            all_cancer_targets.extend(cancer_targets.numpy())

    all_cancer_probs = np.array(all_cancer_probs)
    all_cancer_targets = np.array(all_cancer_targets)

    final_metric = probabilistic_f1(all_cancer_probs, all_cancer_targets)

    # REQUIRED OUTPUT
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    print("\n==== Failure Analysis ====")
    # Calculate absolute error
    errors = np.abs(all_cancer_probs - all_cancer_targets)

    # Get metadata from the dataset
    # Note: The loader preserves order (shuffle=False for val)
    val_df = val_loader.dataset.df.copy()

    # Ensure lengths match
    if len(val_df) == len(errors):
        val_df["error"] = errors

        # Select features to correlate
        # Map density back to numeric if it's not already
        if "density_label" in val_df.columns:
            # density_label is already numeric (0-3 or -1)
            pass

        features = ["age", "implant", "density_label"]
        correlations = {}

        for feat in features:
            if feat in val_df.columns:
                # Simple correlation
                try:
                    corr = val_df[feat].corr(val_df["error"])
                    correlations[feat] = corr
                except:
                    correlations[feat] = np.nan

        print("Correlation between Error Magnitude and Input Features:")
        for feat, corr in correlations.items():
            print(f"  {feat}: {corr:.4f}")

        # Identify worst cases
        print("Top 3 Worst Predictions:")
        val_df_sorted = val_df.sort_values("error", ascending=False)
        print(val_df_sorted[["patient_id", "image_id", "cancer", "error"]].head(3))
    else:
        print("Error: Mismatch between validation set size and prediction size.")

    # 7. Conditional Submission
    print("\n==== Submission Check ====")
    if final_metric > THRESHOLD:
        print(
            f"Metric ({final_metric}) > Threshold ({THRESHOLD}). Generating submission..."
        )
        predict_and_submit(model, test_loader, device, Config.SUBMISSION_PATH)
    else:
        print(
            f"Metric ({final_metric}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
