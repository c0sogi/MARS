import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

# Import provided library components
from library.config import Config
from library.utils import seed_everything, mcrmse_metric
from library.data import get_dataloaders
from library.model import ScaledResidualWideStreamBiGRU
from library.train import train_one_epoch, validate, generate_submission


def run_failure_analysis(val_ids, all_preds, all_targets):
    """
    Analyzes model errors against metadata features.
    """
    # Calculate error per sample (MCRMSE per sample)
    # Shape: (N_samples, Seq_Len, N_targets)
    diff_sq = (all_targets - all_preds) ** 2

    # MSE per sample per target (average over sequence length)
    mse_per_sample_target = np.mean(diff_sq, axis=1)  # (N, 3)

    # RMSE per sample per target
    rmse_per_sample_target = np.sqrt(mse_per_sample_target)  # (N, 3)

    # Mean RMSE per sample (average over targets)
    error_per_sample = np.mean(rmse_per_sample_target, axis=1)  # (N,)

    # Load metadata to correlate
    if not os.path.exists(Config.VAL_DATA_PATH):
        print("Validation metadata not found, skipping detailed failure analysis.")
        return

    df_val = pd.read_parquet(Config.VAL_DATA_PATH)

    # Create analysis dataframe
    analysis_df = pd.DataFrame({"id": val_ids, "error": error_per_sample})

    # Merge with metadata
    # We are interested in signal_to_noise, SN_filter, and sequence composition
    cols_of_interest = ["id", "signal_to_noise", "SN_filter", "sequence"]
    available_cols = [c for c in cols_of_interest if c in df_val.columns]

    merged_df = pd.merge(analysis_df, df_val[available_cols], on="id", how="inner")

    # Derive sequence features
    if "sequence" in merged_df.columns:
        merged_df["len_A"] = merged_df["sequence"].apply(lambda x: x.count("A"))
        merged_df["len_G"] = merged_df["sequence"].apply(lambda x: x.count("G"))
        merged_df["len_C"] = merged_df["sequence"].apply(lambda x: x.count("C"))
        merged_df["len_U"] = merged_df["sequence"].apply(lambda x: x.count("U"))

    # Compute correlations
    print("\nFailure Analysis (Correlation with Error Magnitude):")
    numeric_cols = ["signal_to_noise", "SN_filter", "len_A", "len_G", "len_C", "len_U"]

    for col in numeric_cols:
        if col in merged_df.columns:
            # Handle potential non-numeric types just in case, though they should be numeric
            if pd.api.types.is_numeric_dtype(merged_df[col]):
                corr = merged_df["error"].corr(merged_df[col])
                print(f"  {col}: {corr:.4f}")


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 2. Data Loading
    # Using cached data for efficiency
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Model Initialization
    model = ScaledResidualWideStreamBiGRU().to(device)

    # 4. Optimization Setup
    # Using 15 epochs for a fast but effective baseline
    EPOCHS = 15
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.MSELoss()

    # 5. Training Loop
    best_mcrmse = float("inf")
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    for epoch in range(EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_mcrmse = validate(model, val_loader, device)

        # Step Scheduler
        scheduler.step()

        # Checkpoint
        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            torch.save(model.state_dict(), best_model_path)

    # 6. Final Evaluation
    # Load best model
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # Collect predictions and targets for full validation set
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            # Move to device
            sequence = batch["sequence"].to(device)
            loop_type = batch["loop_type"].to(device)
            pair_dist = batch["pair_dist"].to(device)
            targets = batch["targets"].to(device)

            # Inference
            preds = model(sequence, loop_type, pair_dist)

            # Mask to scored positions (first 68) for metric calculation
            preds_masked = preds[:, : Config.PRED_LEN, :]
            targets_masked = targets[:, : Config.PRED_LEN, :]

            all_preds.append(preds_masked.cpu().numpy())
            all_targets.append(targets_masked.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Compute Final Metric
    final_metric = mcrmse_metric(all_targets, all_preds)
    print(f"Final Validation Metric: {final_metric}")

    # 7. Failure Analysis
    # Get IDs from validation dataset to link with metadata
    val_ids = val_loader.dataset.ids
    run_failure_analysis(val_ids, all_preds, all_targets)

    # 8. Submission Generation
    # Threshold condition: metric < 0.6176461577
    THRESHOLD = 0.6176461577

    if final_metric < THRESHOLD:
        generate_submission(model, test_loader, device, Config.SUBMISSION_PATH)
    else:
        print(
            f"Validation metric {final_metric} did not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
