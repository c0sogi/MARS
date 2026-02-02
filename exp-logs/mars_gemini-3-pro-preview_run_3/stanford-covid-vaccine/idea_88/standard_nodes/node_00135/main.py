import sys
import os
import torch
import pandas as pd
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Ensure local library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed, save_checkpoint
from library.model import HighCapacityBiGRU
from library.data import get_dataloaders
from library.train import train_epoch, validate, inference


def get_val_predictions_and_errors(model, loader, device):
    """
    Runs inference on validation set to get raw predictions and computes per-sample error.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for features, pair_indices, pair_masks, targets in loader:
            features = features.to(device)
            pair_indices = pair_indices.to(device)
            pair_masks = pair_masks.to(device)

            outputs = model(features, pair_indices, pair_masks)

            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.numpy())

    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_targets, axis=0)

    # Slice to scored sequence length
    y_pred = y_pred[:, : Config.SEQ_SCORED, :]
    y_true = y_true[:, : Config.SEQ_SCORED, :]

    # Scored columns: reactivity(0), deg_Mg_pH10(1), deg_Mg_50C(3)
    scored_indices = [0, 1, 3]

    y_pred = y_pred[:, :, scored_indices]
    y_true = y_true[:, :, scored_indices]

    # Compute RMSE per sample (averaged over columns)
    # MSE per element: (N, 68, 3)
    mse = (y_pred - y_true) ** 2
    # Mean over sequence length (dim 1): (N, 3)
    mse_per_col = np.mean(mse, axis=1)
    # RMSE per col: (N, 3)
    rmse_per_col = np.sqrt(mse_per_col)
    # Mean RMSE per sample: (N,)
    sample_errors = np.mean(rmse_per_col, axis=1)

    return sample_errors


def run_failure_analysis(model, val_loader, device):
    """
    Correlates model error with metadata features.
    """
    print("\n==== Failure Analysis ====")

    # 1. Get per-sample errors
    sample_errors = get_val_predictions_and_errors(model, val_loader, device)

    # 2. Load Metadata
    if not os.path.exists(Config.VAL_METADATA_PATH):
        print("Validation metadata not found. Skipping analysis.")
        return

    val_df = pd.read_parquet(Config.VAL_METADATA_PATH)

    # Ensure alignment (val_loader is shuffle=False)
    if len(val_df) != len(sample_errors):
        print(
            f"Warning: Metadata length ({len(val_df)}) != Predictions length ({len(sample_errors)})"
        )
        return

    # 3. Construct Analysis DataFrame
    analysis_df = pd.DataFrame()
    analysis_df["error"] = sample_errors

    # Add metadata features
    if "signal_to_noise" in val_df.columns:
        analysis_df["signal_to_noise"] = val_df["signal_to_noise"]
    if "SN_filter" in val_df.columns:
        analysis_df["SN_filter"] = val_df["SN_filter"]

    # Add sequence properties
    analysis_df["len"] = val_df["sequence"].apply(len)
    for nuc in ["A", "G", "C", "U"]:
        analysis_df[f"pct_{nuc}"] = val_df["sequence"].apply(
            lambda s: s.count(nuc) / len(s)
        )

    # 4. Compute Correlations
    correlations = analysis_df.corr()["error"].sort_values(ascending=False)
    print("Correlation between Error and Features:")
    print(correlations.drop("error"))

    return correlations


def main():
    # 1. Configuration Override for Fast Baseline
    Config.EPOCHS = 15  # Sufficient for small dataset, fast execution

    # 2. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # 3. Data Loading
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 4. Model Initialization
    model = HighCapacityBiGRU().to(device)
    optimizer = AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR)

    # 5. Training Loop
    print(f"Starting training for {Config.EPOCHS} epochs...")
    best_score = float("inf")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)

        # Validate
        val_score = validate(model, val_loader, device)

        # Step Scheduler
        scheduler.step()

        # Save Best
        is_best = val_score < best_score
        if is_best:
            best_score = val_score
            save_checkpoint(
                {"state_dict": model.state_dict(), "best_score": best_score},
                is_best=True,
                checkpoint_dir=Config.WORKING_DIR,
            )

    # 6. Load Best Model
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        checkpoint = torch.load(
            best_model_path, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["state_dict"])
        print("Loaded best model.")

    # 7. Final Metric
    final_val_score = validate(model, val_loader, device)
    print(f"Final Validation Metric: {final_val_score}")

    # 8. Failure Analysis
    run_failure_analysis(model, val_loader, device)

    # 9. Submission Logic
    THRESHOLD = 0.5884495377540588
    if final_val_score < THRESHOLD:
        print(
            f"Validation score {final_val_score} < {THRESHOLD}. Generating submission..."
        )
        inference(model, test_loader, device)
    else:
        print(f"Validation score {final_val_score} >= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
