import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

# Import library components
from library.config import Config
from library.data import get_data_loaders
from library.model import MNPADSModel
from library.train import train_one_epoch, validate, generate_submission, set_seed


def calculate_rmsle(y_true_log, y_pred_log):
    """
    Calculates the Root Mean Squared Logarithmic Error (RMSLE).
    Since inputs are already log(1+x) transformed, this is just RMSE.
    """
    return np.sqrt(mean_squared_error(y_true_log, y_pred_log, multioutput="raw_values"))


def perform_failure_analysis(model, val_loader, device):
    """
    Analyzes the correlation between input features and prediction errors on the validation set.
    """
    model.eval()
    all_global_features = []
    all_targets = []
    all_preds = []

    # Collect data
    with torch.no_grad():
        for batch in val_loader:
            atomic_features = batch["atomic_features"].to(device)
            global_features = batch["global_features"].to(device)
            mask = batch["mask"].to(device)
            targets = batch["targets"].to(device)

            outputs = model(atomic_features, global_features, mask)

            all_global_features.append(global_features.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_preds.append(outputs.cpu().numpy())

    all_global_features = np.concatenate(all_global_features, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)

    # Calculate errors (MAE on log scale)
    errors = np.abs(all_targets - all_preds)
    mean_errors = np.mean(
        errors, axis=1
    )  # Average error across both targets per sample

    # Create DataFrame for correlation analysis
    # Global feature names based on data.py implementation
    # Order: lengths(3), angles(3), volume(1), density(1), stoich(3), total_atoms(1), apf(1)
    feature_names = [
        "lat_len_a",
        "lat_len_b",
        "lat_len_c",
        "lat_ang_alpha",
        "lat_ang_beta",
        "lat_ang_gamma",
        "vol",
        "density",
        "stoich_Al",
        "stoich_Ga",
        "stoich_In",
        "total_atoms",
        "apf",
    ]

    df_analysis = pd.DataFrame(all_global_features, columns=feature_names)
    df_analysis["error"] = mean_errors

    # Compute correlations
    correlations = (
        df_analysis.corr()["error"].drop("error").sort_values(ascending=False, key=abs)
    )

    print("\nFailure Analysis - Correlation between Global Features and Model Error:")
    print("-" * 65)
    print(f"{'Feature':<30} | {'Correlation':<10}")
    print("-" * 65)
    for feat, corr in correlations.items():
        print(f"{feat:<30} | {corr:.4f}")
    print("-" * 65)


def main():
    # 1. Setup
    # Override epochs for fast baseline execution as requested
    Config.EPOCHS = 50
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # 2. Data Loading
    # load_cached_data=True attempts to use preprocessed .npz files from ./working
    train_loader, val_loader, test_loader = get_data_loaders(load_cached_data=True)

    # 3. Model Initialization
    model = MNPADSModel(config=Config).to(device)

    # 4. Optimization
    # Targets are log1p transformed, so MSE on these targets approximates MSLE
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
        min_lr=Config.MIN_LR,
    )

    # 5. Training Loop
    print(f"\nStarting training for {Config.EPOCHS} epochs...")
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # 6. Final Evaluation
    print("\nLoading best model for evaluation...")
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    # Compute Final Validation Metric
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for batch in val_loader:
            atomic_features = batch["atomic_features"].to(device)
            global_features = batch["global_features"].to(device)
            mask = batch["mask"].to(device)
            targets = batch["targets"].to(device)

            outputs = model(atomic_features, global_features, mask)

            all_targets.append(targets.cpu().numpy())
            all_preds.append(outputs.cpu().numpy())

    all_targets = np.concatenate(all_targets, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)

    # Calculate column-wise RMSLE (since data is log-transformed, this is RMSE)
    # Target 0: formation_energy, Target 1: bandgap_energy
    rmsle_per_col = calculate_rmsle(all_targets, all_preds)
    final_metric = np.mean(rmsle_per_col)

    print(f"Validation RMSLE (Formation Energy): {rmsle_per_col[0]}")
    print(f"Validation RMSLE (Bandgap Energy):   {rmsle_per_col[1]}")
    print(f"Final Validation Metric: {final_metric}")

    # 7. Failure Analysis
    perform_failure_analysis(model, val_loader, device)

    # 8. Submission Generation
    # Threshold check: 0.05479004207787702
    threshold = 0.05479004207787702

    if final_metric < threshold:
        print(
            f"\nMetric {final_metric} is below threshold {threshold}. Generating submission..."
        )
        generate_submission(model, test_loader, device, Config.SUBMISSION_OUTPUT_PATH)
    else:
        print(
            f"\nMetric {final_metric} is NOT below threshold {threshold}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
