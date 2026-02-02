import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import sys

# Import provided library modules
from library.config import Config
from library.utils import set_seed, MCRMSE
from library.dataset import get_dataloaders
from library.model import RNAModel
from library.engine import train_one_epoch, validate, predict_and_submit


def run_failure_analysis(model, val_loader, device):
    """
    Computes per-sample error on the validation set and correlates it with
    metadata features to identify sources of failure.
    """
    print("\nRunning Failure Analysis...")
    model.eval()

    all_preds = []
    all_targets = []
    all_ids = []

    # 1. Inference on Validation Set
    with torch.no_grad():
        for batch in val_loader:
            # Move to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            preds = model(batch)
            targets = batch["targets"]
            ids = batch["id"]

            # Slice to scored sequence length (first 68 positions)
            preds_scored = preds[:, : Config.SEQ_SCORED, :]

            all_preds.append(preds_scored.cpu())
            all_targets.append(targets.cpu())
            all_ids.extend(ids)

    all_preds = torch.cat(all_preds, dim=0)  # (N_samples, 68, 3)
    all_targets = torch.cat(all_targets, dim=0)  # (N_samples, 68, 3)

    # 2. Calculate Error Metric Per Sample
    # We use the mean RMSE across the 3 target columns for each sample as the error magnitude.
    # Squared Error: (N, 68, 3)
    squared_error = (all_preds - all_targets) ** 2

    # Mean over sequence length (dim 1) -> (N, 3)
    mse_per_sample_col = torch.mean(squared_error, dim=1)

    # RMSE per sample col -> (N, 3)
    rmse_per_sample_col = torch.sqrt(mse_per_sample_col)

    # Mean over the 3 target columns -> (N,)
    mean_rmse_per_sample = torch.mean(rmse_per_sample_col, dim=1).numpy()

    # 3. Load Metadata for Correlation
    # We read the validation parquet file to get features like signal_to_noise
    df_val = pd.read_parquet(Config.VAL_FILE)

    # Map calculated errors to the dataframe using IDs
    error_map = {id_: err for id_, err in zip(all_ids, mean_rmse_per_sample)}
    df_val["model_error"] = df_val["id"].map(error_map)

    # 4. Feature Engineering for Analysis
    # Nucleotide counts
    df_val["A_count"] = df_val["sequence"].apply(lambda x: x.count("A"))
    df_val["G_count"] = df_val["sequence"].apply(lambda x: x.count("G"))
    df_val["C_count"] = df_val["sequence"].apply(lambda x: x.count("C"))
    df_val["U_count"] = df_val["sequence"].apply(lambda x: x.count("U"))

    # List of features to check correlations against
    potential_features = [
        "signal_to_noise",
        "SN_filter",
        "seq_length",
        "A_count",
        "G_count",
        "C_count",
        "U_count",
    ]
    valid_features = [f for f in potential_features if f in df_val.columns]

    # 5. Compute and Print Correlations
    print("Correlation between Model Error (Mean RMSE) and Features:")
    correlations = (
        df_val[valid_features + ["model_error"]]
        .corr()["model_error"]
        .drop("model_error")
    )
    print(correlations.sort_values(ascending=False))


def main():
    # 1. Configuration and Setup
    # Limit epochs for a fast baseline execution
    Config.EPOCHS = 15
    Config.setup()
    set_seed(Config.SEED)

    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. Data Loading
    print("Loading datasets...")
    train_loader, val_loader, test_loader = get_dataloaders(
        debug=Config.DEBUG, load_cached_data=True
    )

    # 3. Model Initialization
    model = RNAModel().to(device)

    # Optimizer and Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)

    # Loss Function (MSE)
    criterion = nn.MSELoss()

    # 4. Training Loop
    best_score = float("inf")
    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_score = validate(model, val_loader, device)

        # Scheduler Step
        scheduler.step()

        # Logging
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val MCRMSE: {val_score:.6f}"
        )

        # Save Best Model
        if val_score < best_score:
            best_score = val_score
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)

    print("Training complete.")

    # 5. Final Evaluation
    # Load the best model checkpoint
    if os.path.exists(Config.BEST_MODEL_PATH):
        model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
        print("Loaded best model for final evaluation.")

    # Compute Final Metric
    final_metric = validate(model, val_loader, device)
    # REQUIRED FORMAT
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    run_failure_analysis(model, val_loader, device)

    # 7. Submission Generation
    # Threshold check
    THRESHOLD = 0.6199890971183777

    if final_metric < THRESHOLD:
        print(
            f"Validation metric {final_metric} is below threshold {THRESHOLD}. Generating submission..."
        )
        predict_and_submit(test_loader, device)
    else:
        print(
            f"Validation metric {final_metric} is NOT below threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
