import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from scipy.stats import pearsonr

# Ensure library is in path
sys.path.append(".")

from library.config import Config
from library.utils import seed_everything, calculate_mcrmse
from library.data import get_dataloaders
from library.model import RNAModel, predict_and_submit
from library.train import train_epoch, validate


def run_failure_analysis(model, val_loader, device):
    """
    Analyzes model errors on the validation set and correlates them with metadata features.
    """
    print("\nRunning Failure Analysis...")

    # Load validation metadata to get features
    if not os.path.exists(Config.VAL_DATA_PATH):
        print("Validation metadata not found. Skipping analysis.")
        return

    df_val = pd.read_parquet(Config.VAL_DATA_PATH)

    # Ensure model is in eval mode
    model.eval()

    all_preds = []
    all_targets = []

    # Generate predictions for validation set
    with torch.no_grad():
        for seq, loop, dist, targets, mask in val_loader:
            seq = seq.to(device)
            loop = loop.to(device)
            dist = dist.to(device)

            preds = model(seq, loop, dist)

            # Slice to scored positions (first 68)
            preds = preds[:, : Config.PRED_LEN, :]
            targets = targets[:, : Config.PRED_LEN, :]

            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # Concatenate
    preds_arr = np.concatenate(all_preds, axis=0)  # Shape: (N_samples, 68, 3)
    targets_arr = np.concatenate(all_targets, axis=0)  # Shape: (N_samples, 68, 3)

    # Verify alignment
    if len(df_val) != len(preds_arr):
        print(
            f"Warning: Metadata length ({len(df_val)}) matches prediction length ({len(preds_arr)}) mismatch."
        )
        # Truncate to minimum length to proceed safely
        min_len = min(len(df_val), len(preds_arr))
        df_val = df_val.iloc[:min_len]
        preds_arr = preds_arr[:min_len]
        targets_arr = targets_arr[:min_len]

    # Calculate RMSE per sample (scalar error metric)
    # Mean over positions (axis 1) and targets (axis 2)
    mse_per_sample = np.mean((preds_arr - targets_arr) ** 2, axis=(1, 2))
    rmse_per_sample = np.sqrt(mse_per_sample)

    df_val["error_rmse"] = rmse_per_sample

    # Feature Engineering for Analysis
    df_val["len_A"] = df_val["sequence"].apply(lambda x: x.count("A"))
    df_val["len_G"] = df_val["sequence"].apply(lambda x: x.count("G"))
    df_val["len_C"] = df_val["sequence"].apply(lambda x: x.count("C"))
    df_val["len_U"] = df_val["sequence"].apply(lambda x: x.count("U"))

    # Calculate Correlations
    features = ["signal_to_noise", "SN_filter", "len_A", "len_G", "len_C", "len_U"]

    print("-" * 40)
    print("Correlation between Input Features and Error (RMSE):")
    print("-" * 40)

    for feat in features:
        if feat in df_val.columns:
            # Drop NaNs if any
            valid_mask = df_val[feat].notna() & df_val["error_rmse"].notna()
            if valid_mask.sum() > 1:
                corr, _ = pearsonr(
                    df_val.loc[valid_mask, feat], df_val.loc[valid_mask, "error_rmse"]
                )
                print(f"{feat:<20}: {corr:.6f}")
            else:
                print(f"{feat:<20}: N/A (Insufficient data)")
        else:
            print(f"{feat:<20}: Not found in metadata")
    print("-" * 40)


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = Config.DEVICE

    # Override Config for Fast Baseline
    # We use 20 epochs to ensure convergence with the smaller model.
    Config.EPOCHS = 20

    print(f"Initializing run on {device}...")

    # 2. Data Loading
    # Uses caching to speed up subsequent runs
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Model Initialization
    model = RNAModel().to(device)

    # 4. Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)

    # 5. Training Loop
    best_mcrmse = float("inf")
    best_model_path = os.path.join(Config.SUBMISSION_DIR, "best_model.pth")

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, device, Config.GRAD_CLIP
        )

        # Step Scheduler
        scheduler.step()

        # Validate
        val_mcrmse = validate(model, val_loader, device)

        print(
            f"Epoch {epoch+1:02d}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val MCRMSE: {val_mcrmse:.6f}"
        )

        # Save Best
        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            torch.save(model.state_dict(), best_model_path)

    print("Training complete.")

    # 6. Final Evaluation
    # Load best model
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print("Loaded best model state.")

    # Calculate Final Metric on full validation set
    final_metric = validate(model, val_loader, device)
    print(f"Final Validation Metric: {final_metric}")

    # 7. Failure Analysis
    run_failure_analysis(model, val_loader, device)

    # 8. Submission Logic
    # Threshold defined in task
    THRESHOLD = 0.6176461577

    if final_metric < THRESHOLD:
        print(
            f"Validation metric ({final_metric}) meets threshold ({THRESHOLD}). Generating submission..."
        )
        predict_and_submit(model, test_loader)
    else:
        print(
            f"Validation metric ({final_metric}) did not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
