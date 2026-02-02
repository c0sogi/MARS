import os
import sys
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.cuda.amp import GradScaler

# Import components from the provided library files
from library.config import Config
from library.utils import seed_everything, get_device
from library.data import get_dataloaders
from library.model import SMTSINModel
from library.train import train_one_epoch, validate, inference

# ==========================================
# Configuration Overrides for Fast Baseline
# ==========================================
# Limit epochs to ensure completion within 2 hours while using full data
Config.EPOCHS = 3
# Ensure we use the full dataset (debug=False) for meaningful results
# Batch size 24 on A100 is efficient.


def failure_analysis(model, val_loader, device):
    """
    Performs failure analysis on the validation set.
    Computes correlation between error magnitude and metadata features.
    """
    print("\n=== Failure Analysis ===")

    # Load validation metadata to get features
    # We read the processed parquet if available, or the csv
    # The dataloader order matches the dataframe order if shuffle=False
    val_meta_path = Config.VAL_META_PATH
    if not os.path.exists(val_meta_path):
        print("Validation metadata not found for analysis.")
        return

    val_df = pd.read_csv(val_meta_path)

    # Collect predictions and targets
    model.eval()
    all_preds = []
    all_targets = []

    # We need to ensure we iterate exactly as the loader does
    with torch.no_grad():
        for images, metadata, targets in val_loader:
            images = images.to(device)
            metadata = metadata.to(device)

            with torch.amp.autocast(device_type="cuda", enabled=Config.USE_AMP):
                outputs = model(images, metadata)

            # Extract cancer probabilities
            probs = torch.sigmoid(outputs["cancer"]).float().cpu().numpy().flatten()
            t_cancer = targets["cancer"].numpy().flatten()

            all_preds.extend(probs)
            all_targets.extend(t_cancer)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Calculate Error Magnitude
    # For probabilistic targets (0 or 1), error is abs(prob - target)
    errors = np.abs(all_preds - all_targets)

    # Align with DataFrame
    # The val_loader length might differ slightly if drop_last was True (it is False for val)
    # or if the loader was created from a cached dataframe that differs.
    # We truncate to the minimum length to be safe, though they should match.
    min_len = min(len(val_df), len(errors))
    val_df = val_df.iloc[:min_len].copy()
    val_df["error"] = errors[:min_len]

    # Feature Correlations
    # Map Density to numeric for correlation
    density_map = {"A": 1, "B": 2, "C": 3, "D": 4}
    val_df["density_num"] = val_df["density"].map(density_map)

    # Features to check
    features = ["age", "density_num", "implant"]

    print("Correlation between Error Magnitude and Features:")
    for feat in features:
        if feat in val_df.columns:
            # Drop NaNs for correlation calculation
            valid_rows = val_df[[feat, "error"]].dropna()
            if len(valid_rows) > 0:
                corr = valid_rows[feat].corr(valid_rows["error"])
                print(f"  {feat}: {corr:.6f}")
            else:
                print(f"  {feat}: N/A (No valid data)")
        else:
            print(f"  {feat}: Not found in metadata")


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = get_device()
    print(f"Running on device: {device}")

    # 2. Data Loading
    # debug=False ensures we use the full dataset
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=True, debug=False
    )

    # 3. Model Initialization
    model = SMTSINModel().to(device)

    # 4. Optimization Setup
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scaler = GradScaler(enabled=Config.USE_AMP)

    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        epochs=Config.EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=Config.PCT_START,
        div_factor=Config.DIV_FACTOR,
        final_div_factor=Config.FINAL_DIV_FACTOR,
    )

    # 5. Training Loop
    best_pf1 = 0.0
    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, scaler, epoch
        )

        # Validate
        val_pf1 = validate(model, val_loader, device)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val pF1: {val_pf1:.6f}"
        )

        # Save Best
        if val_pf1 > best_pf1:
            best_pf1 = val_pf1
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)

    # 6. Final Reporting
    # Print exact metric as required
    print(f"Final Validation Metric: {best_pf1}")

    # 7. Failure Analysis
    # Load best model for analysis
    if os.path.exists(Config.BEST_MODEL_PATH):
        model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
        failure_analysis(model, val_loader, device)

    # 8. Submission
    THRESHOLD = 0.044888656586408615

    if best_pf1 > THRESHOLD:
        print(f"\nValidation metric {best_pf1} exceeds threshold {THRESHOLD}.")
        print("Generating submission...")
        inference(model, test_loader, device)
    else:
        print(f"\nValidation metric {best_pf1} does not exceed threshold {THRESHOLD}.")
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
