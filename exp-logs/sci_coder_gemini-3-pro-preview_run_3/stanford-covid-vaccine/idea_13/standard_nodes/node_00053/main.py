import os
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Import from provided library files
from library.config import Config, setup_reproducibility
from library.utils import get_device
from library.data import get_dataloaders
from library.model import GatedSpatialConvBiGRU
from library.train import train_one_epoch, validate, generate_submission


def main():
    # 1. Setup & Configuration
    setup_reproducibility(Config.SEED)
    device = get_device()

    # Override Config for Fast Baseline
    Config.EPOCHS = 15
    Config.T_MAX = 15
    Config.SUBMISSION_FILE = "./submission/submission.csv"

    # Ensure submission directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_FILE), exist_ok=True)

    # 2. Data Loading
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Model Initialization
    model = GatedSpatialConvBiGRU(Config).to(device)

    # 4. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
    )

    # 5. Training Loop
    best_mcrmse = float("inf")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)

        # Validate
        val_mcrmse = validate(model, val_loader, device)

        # Step Scheduler
        scheduler.step()

        # Log
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val MCRMSE: {val_mcrmse:.6f}"
        )

        # Checkpoint
        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)

    # 6. Final Evaluation
    print(f"Final Validation Metric: {best_mcrmse}")

    # 7. Failure Analysis
    # Reload best model
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    model.eval()

    # Collect predictions and targets for analysis
    all_ids = []
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["inputs"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            targets = batch["targets"].cpu().numpy()
            ids = batch["id"]

            outputs = model(inputs, pair_indices)
            # Slice to scored length
            outputs_scored = outputs[:, : Config.SEQ_SCORED, :].cpu().numpy()

            all_preds.append(outputs_scored)
            all_targets.append(targets)
            all_ids.extend(ids)

    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_targets, axis=0)

    # Identify scored columns indices
    scored_indices = [
        i for i, col in enumerate(Config.TARGET_COLS) if col in Config.SCORED_COLS
    ]

    # Filter predictions to scored columns
    y_pred_scored = y_pred[:, :, scored_indices]
    y_true_scored = y_true[:, :, scored_indices]

    # Calculate per-sample error (Mean of RMSEs across scored columns)
    # MSE per column per sample: (N, 68, 3) -> mean over seq(1) -> (N, 3)
    mse_per_sample_col = np.mean((y_pred_scored - y_true_scored) ** 2, axis=1)
    rmse_per_sample_col = np.sqrt(mse_per_sample_col)
    # Mean over columns -> (N,)
    sample_errors = np.mean(rmse_per_sample_col, axis=1)

    # Load Metadata
    val_df = pd.read_parquet(Config.VAL_METADATA)

    # Merge errors with metadata
    error_df = pd.DataFrame({"id": all_ids, "error": sample_errors})
    analysis_df = val_df.merge(error_df, on="id")

    # Feature Engineering for Correlation
    for nuc in ["A", "G", "U", "C"]:
        analysis_df[f"pct_{nuc}"] = analysis_df["sequence"].apply(
            lambda s: s.count(nuc) / len(s)
        )

    # Compute Correlations
    print("Failure Analysis (Correlation with Error):")
    features_to_check = [
        "signal_to_noise",
        "SN_filter",
        "pct_A",
        "pct_G",
        "pct_U",
        "pct_C",
    ]

    for feat in features_to_check:
        if feat in analysis_df.columns:
            corr, _ = pearsonr(analysis_df[feat], analysis_df["error"])
            print(f"{feat}: {corr:.4f}")

    # 8. Submission
    THRESHOLD = 0.7247761841173526

    if best_mcrmse < THRESHOLD:
        generate_submission(model, test_loader, device, Config.SUBMISSION_FILE)


if __name__ == "__main__":
    main()
