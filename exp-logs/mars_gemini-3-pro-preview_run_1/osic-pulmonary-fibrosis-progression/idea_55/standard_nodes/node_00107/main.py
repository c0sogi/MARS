import os
import sys
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import warnings

# Import from provided library
from library.config import Config
from library.utils import seed_everything, metric_score
from library.data import get_dataloaders
from library.model import NSLHN
from library.train import (
    LaplaceLogLikelihoodLoss,
    train_epoch,
    valid_epoch,
    generate_submission,
)

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def analyze_failures(model, val_loader, device):
    """
    Performs inference on validation set and analyzes error correlations.
    """
    model.eval()
    all_targets = []
    all_preds = []
    all_sigmas = []
    all_metas = []

    # Columns to collect for analysis
    meta_columns = ["Baseline_Age", "Baseline_Percent", "Weeks", "Baseline_FVC"]

    # We need to extract metadata corresponding to the batches.
    # The loader returns batches of tensors. We need to access the source dataframe
    # or pass metadata through the loader.
    # LungDataset returns 'tabular' (normalized) and some raw scalars.
    # We can reconstruct or just use the raw scalars returned in the batch.
    # The batch keys are: 'patient_id', 'axial', 'coronal', 'tabular', 'delta_week', 'baseline_fvc', 'target'
    # 'tabular' contains [Age_Norm, Sex_Bin, Smoke_Ex, Smoke_Nev, Smoke_Cur, Percent_Norm]
    # We can use 'baseline_fvc' and 'delta_week' directly.
    # We can approximate Age/Percent from the normalized tabular features or just read from val.csv
    # Since the loader is not shuffled (val_loader shuffle=False), we can align with val_df.

    with torch.no_grad():
        for batch in val_loader:
            axial = batch["axial"].to(device)
            coronal = batch["coronal"].to(device)
            tabular = batch["tabular"].to(device)
            target = batch["target"].to(device)
            delta_week = batch["delta_week"].to(device)
            baseline_fvc = batch["baseline_fvc"].to(device)

            # Forward pass
            alpha, sigma_base, sigma_growth = model(axial, coronal, tabular)

            # Calculate predictions
            pred_fvc = baseline_fvc + alpha * delta_week
            pred_sigma = sigma_base + sigma_growth * torch.abs(delta_week)

            all_targets.append(target.cpu().numpy())
            all_preds.append(pred_fvc.cpu().numpy())
            all_sigmas.append(pred_sigma.cpu().numpy())

    # Concatenate results
    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    sigma = np.concatenate(all_sigmas)

    # Calculate Metric
    final_score = metric_score(y_true, y_pred, sigma)

    # Load validation metadata to correlate
    val_df = pd.read_csv(Config.VAL_CSV)

    # Ensure alignment (Loader is shuffle=False)
    if len(val_df) != len(y_true):
        print(
            f"Warning: Validation DF length ({len(val_df)}) != Prediction length ({len(y_true)})"
        )
        # Truncate to match just in case
        min_len = min(len(val_df), len(y_true))
        val_df = val_df.iloc[:min_len]
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]

    # Calculate Absolute Error
    abs_error = np.abs(y_true - y_pred)

    # Create Analysis DataFrame
    analysis_df = val_df.copy()
    analysis_df["Abs_Error"] = abs_error

    print(f"Final Validation Metric: {final_score}")

    print("\nFailure Analysis (Correlation with Absolute Error):")
    # Correlate Error with numerical features
    features_to_check = ["Age", "Percent", "Weeks", "FVC"]
    for feat in features_to_check:
        if feat in analysis_df.columns:
            corr = analysis_df["Abs_Error"].corr(analysis_df[feat])
            print(f"  Correlation with {feat}: {corr:.4f}")

    return final_score


def main():
    # 1. Setup & Configuration
    # Override Config for Fast Baseline
    Config.EPOCHS = 15
    Config.PATIENCE = 5
    Config.BATCH_SIZE = 32

    Config.setup()
    seed_everything(Config.SEED)

    device = torch.device(Config.DEVICE)

    # 2. Data Loading
    print("Loading datasets...")
    # Use full dataset (debug=False) but limited epochs for speed/quality balance
    train_loader, val_loader, test_loader = get_dataloaders(debug=False)

    # 3. Model Initialization
    print("Initializing model...")
    model = NSLHN().to(device)

    # 4. Training Setup
    criterion = LaplaceLogLikelihoodLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)

    best_score = -float("inf")
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    patience_counter = 0

    print(f"Starting training for {Config.EPOCHS} epochs...")

    # 5. Training Loop
    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, val_score = valid_epoch(model, val_loader, criterion, device)

        # Step Scheduler
        scheduler.step()

        # Logging (Minimal)
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.4f} | Val Metric: {val_score:.6f}"
        )

        # Checkpoint
        if val_score > best_score:
            best_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    # 6. Evaluation & Failure Analysis
    print("\nLoading best model for analysis...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    final_metric = analyze_failures(model, val_loader, device)

    # 7. Submission
    # Threshold from requirements
    THRESHOLD = -6.510164260864258

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) > Threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission(model, test_loader, device)
    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
