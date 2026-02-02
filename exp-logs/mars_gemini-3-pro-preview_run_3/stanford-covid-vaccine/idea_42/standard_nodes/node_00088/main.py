import os
import time
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from scipy.stats import pearsonr

# Import from library
from library.config import Config
from library.utils import set_seed, scored_mcrmse
from library.data import get_dataloaders
from library.model import RNAModel
from library.train import train_one_epoch, validate


def main():
    # =========================================================================
    # 1. Configuration & Setup
    # =========================================================================
    # Override Config for Fast Baseline Execution
    Config.EPOCHS = 15  # Reduced from 25 for speed
    Config.SUBMISSION_PATH = "./submission/submission.csv"

    # Create submission directory
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # =========================================================================
    # 2. Data Loading
    # =========================================================================
    # Load dataloaders using cached data if available
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # =========================================================================
    # 3. Model Initialization
    # =========================================================================
    model = RNAModel().to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.ETA_MIN
    )

    # =========================================================================
    # 4. Training Loop
    # =========================================================================
    best_score = float("inf")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device)

        # Validate
        val_score = validate(model, val_loader, device)

        # Scheduler Step
        scheduler.step()

        # Save Best Model
        if val_score < (best_score - Config.MIN_DELTA):
            best_score = val_score
            torch.save(model.state_dict(), Config.MODEL_PATH)

    # =========================================================================
    # 5. Final Evaluation & Failure Analysis
    # =========================================================================
    # Load best model
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.eval()

    # --- Calculate Final Metric on Full Validation Set ---
    # We need sample-wise errors for failure analysis, so we'll do a custom pass
    val_ids = []
    val_preds = []
    val_targets = []

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["inputs"].to(device)
            adj = batch["adj_indices"].to(device)
            targets = batch["targets"].to(device)
            ids = batch["id"]

            outputs = model(inputs, adj)

            val_preds.append(outputs.cpu())
            val_targets.append(targets.cpu())
            val_ids.extend(ids)

    # Concatenate
    y_pred_global = torch.cat(val_preds, dim=0)
    y_true_global = torch.cat(val_targets, dim=0)

    # Compute Final Metric
    final_metric = scored_mcrmse(y_pred_global, y_true_global).item()
    print(f"Final Validation Metric: {final_metric}")

    # --- Failure Analysis ---
    # Calculate MCRMSE per sample
    # 1. Slice to scored length
    seq_scored = Config.SEQ_SCORED
    pred_sliced = y_pred_global[:, :seq_scored, :]
    true_sliced = y_true_global[:, :seq_scored, :]

    # 2. Filter Scored Columns
    scored_indices = [
        i for i, col in enumerate(Config.TARGET_COLS) if col in Config.SCORED_COLS
    ]
    pred_filtered = pred_sliced[:, :, scored_indices]
    true_filtered = true_sliced[:, :, scored_indices]

    # 3. Compute RMSE per sample (average over columns and sequence)
    # Shape: (N, Seq, Cols) -> (N,)
    mse_per_sample = torch.mean((pred_filtered - true_filtered) ** 2, dim=(1, 2))
    rmse_per_sample = torch.sqrt(mse_per_sample).numpy()

    # Load Validation Metadata to get features
    val_df = pd.read_parquet(Config.VAL_DATA_PATH)
    # Ensure alignment by ID
    val_df = val_df.set_index("id").loc[val_ids].reset_index()

    # Add error to dataframe
    val_df["error_rmse"] = rmse_per_sample

    # Feature Engineering for Correlation
    val_df["len_sequence"] = val_df["sequence"].apply(len)
    val_df["pct_A"] = val_df["sequence"].apply(lambda x: x.count("A") / len(x))
    val_df["pct_G"] = val_df["sequence"].apply(lambda x: x.count("G") / len(x))
    val_df["pct_C"] = val_df["sequence"].apply(lambda x: x.count("C") / len(x))
    val_df["pct_U"] = val_df["sequence"].apply(lambda x: x.count("U") / len(x))
    val_df["pct_paired"] = val_df["structure"].apply(
        lambda x: (x.count("(") + x.count(")")) / len(x)
    )

    # Calculate Correlations
    analysis_cols = [
        "signal_to_noise",
        "len_sequence",
        "pct_A",
        "pct_G",
        "pct_C",
        "pct_U",
        "pct_paired",
    ]
    print("\nFailure Analysis - Error Correlations:")
    for col in analysis_cols:
        if col in val_df.columns:
            # Handle potential NaNs in signal_to_noise or other columns
            valid_idx = val_df[col].notna() & val_df["error_rmse"].notna()
            if valid_idx.sum() > 1:
                corr, _ = pearsonr(
                    val_df.loc[valid_idx, col], val_df.loc[valid_idx, "error_rmse"]
                )
                print(f"  {col}: {corr:.4f}")

    # =========================================================================
    # 6. Submission Generation
    # =========================================================================
    THRESHOLD = 0.5884495377540588

    if final_metric < THRESHOLD:
        ids_list = []
        preds_list = []

        with torch.no_grad():
            for batch in test_loader:
                inputs = batch["inputs"].to(device)
                adj = batch["adj_indices"].to(device)
                ids = batch["id"]

                outputs = model(inputs, adj)

                preds_list.append(outputs.cpu().numpy())
                ids_list.extend(ids)

        all_preds = np.concatenate(preds_list, axis=0)

        # Flatten for submission format
        submission_data = []
        target_cols = Config.TARGET_COLS

        for i, sample_id in enumerate(ids_list):
            sample_preds = all_preds[i]  # (107, 5)

            for seqpos in range(Config.SEQ_LEN):
                row_id = f"{sample_id}_{seqpos}"
                row_preds = sample_preds[seqpos]

                row_dict = {"id_seqpos": row_id}
                for col_idx, col_name in enumerate(target_cols):
                    row_dict[col_name] = float(row_preds[col_idx])

                submission_data.append(row_dict)

        submission_df = pd.DataFrame(submission_data)
        cols_order = ["id_seqpos"] + target_cols
        submission_df = submission_df[cols_order]

        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)


if __name__ == "__main__":
    main()
