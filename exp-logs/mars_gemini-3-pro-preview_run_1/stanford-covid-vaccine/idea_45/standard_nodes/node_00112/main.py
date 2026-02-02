import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from library.config import Config
from library.utils import seed_everything, MCRMSE
from library.data import get_dataloaders
from library.model import RNAModel
from library.train import train_one_epoch, validate, generate_submission


def main():
    # 1. Setup and Configuration
    seed_everything(Config.SEED)

    # Override Config paths to match requirements
    Config.SUBMISSION_SAVE_PATH = "./submission/submission.csv"
    os.makedirs(os.path.dirname(Config.SUBMISSION_SAVE_PATH), exist_ok=True)
    Config.initialize_workspace()

    device = torch.device(Config.DEVICE)

    # 2. Data Loading
    # Load cached data if available to speed up execution
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Model Initialization
    model = RNAModel().to(device)

    # 4. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.T_MAX)

    # 5. Training Loop
    best_mcrmse = float("inf")

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_mcrmse = validate(model, val_loader, device)

        # Scheduler Step
        scheduler.step()

        # Save Best Model
        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)

    # 6. Final Evaluation & Failure Analysis
    print("Loading best model for evaluation...")
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    # Collect Validation Predictions for Analysis
    all_preds = []
    all_targets = []
    all_ids = []

    with torch.no_grad():
        for batch in val_loader:
            sequence = batch["sequence"].to(device)
            loop_type = batch["loop_type"].to(device)
            pair_dist = batch["pair_dist"].to(device)
            targets = batch["target"].to(device)
            ids = batch["id"]

            preds = model(sequence, loop_type, pair_dist)

            # Slice to scored positions
            preds_sliced = preds[:, : Config.PRED_LEN, :]
            targets_sliced = targets[:, : Config.PRED_LEN, :]

            all_preds.append(preds_sliced)
            all_targets.append(targets_sliced)
            all_ids.extend(ids)

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Compute Final Metric
    final_metric = MCRMSE(all_targets, all_preds).item()
    print(f"Final Validation Metric: {final_metric}")

    # --- Failure Analysis ---
    print("\nPerforming Failure Analysis...")

    # Calculate per-sample RMSE (averaged over the 3 scored columns)
    # Shape: (N, 68, 3) -> (N,)
    squared_diff = (all_targets - all_preds) ** 2
    # Mean over sequence length (dim 1) and targets (dim 2)
    mse_per_sample = torch.mean(squared_diff, dim=(1, 2))
    rmse_per_sample = torch.sqrt(mse_per_sample).cpu().numpy()

    # Load Metadata
    df_val = pd.read_parquet(Config.VAL_FILE)

    # Create a mapping from ID to error
    error_map = dict(zip(all_ids, rmse_per_sample))

    # Add error to dataframe
    df_val["model_error"] = df_val["id"].map(error_map)

    # Feature Engineering for Analysis
    df_val["GC_content"] = df_val["sequence"].apply(
        lambda x: (x.count("G") + x.count("C")) / len(x)
    )

    # Calculate Correlations
    analysis_cols = ["signal_to_noise", "SN_filter", "GC_content"]
    print("Correlation between Model Error (RMSE) and Features:")

    for col in analysis_cols:
        if col in df_val.columns:
            # Drop NaNs if any (though data should be clean)
            valid_df = df_val[[col, "model_error"]].dropna()
            if len(valid_df) > 1:
                corr, _ = pearsonr(valid_df[col], valid_df["model_error"])
                print(f"  {col}: {corr:.4f}")

    # 7. Submission Logic
    THRESHOLD = 0.6176461577
    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric}) is better than threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission(model, test_loader, device)
    else:
        print(
            f"\nMetric ({final_metric}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
