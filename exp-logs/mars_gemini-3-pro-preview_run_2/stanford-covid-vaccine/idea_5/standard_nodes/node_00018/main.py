import os
import sys
import numpy as np
import pandas as pd
import torch
import scipy.stats as stats

# Import from provided libraries
from library.config import Config
from library.train import run_training
from library.inference import run_inference
from library.model import get_dataloaders, HybridNet


def main():
    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    # Override Config for Fast Baseline execution
    # Reducing epochs to ensure completion within time limits while allowing convergence
    Config.EPOCHS = 15

    print(f"Configuration:")
    print(f"  Device: {Config.DEVICE}")
    print(f"  Epochs: {Config.EPOCHS}")
    print(f"  Batch Size: {Config.BATCH_SIZE}")
    print(f"  Working Dir: {Config.WORKING_DIR}")

    # -------------------------------------------------------------------------
    # 2. Training
    # -------------------------------------------------------------------------
    print("\n=== Starting Training Phase ===")
    # run_training handles data loading, model init, training loop, and saving best model
    run_training(load_cached_data=True)

    # -------------------------------------------------------------------------
    # 3. Validation & Metric Calculation
    # -------------------------------------------------------------------------
    print("\n=== Starting Validation Phase ===")

    # Load Validation Data
    # We use the library function to ensure consistent preprocessing
    _, val_loader, _ = get_dataloaders(load_cached_data=True)

    # Load Best Model
    device = torch.device(Config.DEVICE)
    model = HybridNet().to(device)

    if not os.path.exists(Config.MODEL_SAVE_PATH):
        print(f"Error: Model file not found at {Config.MODEL_SAVE_PATH}")
        return

    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    # Run Inference on Validation Set
    all_preds = []
    all_targets = []
    all_ids = []

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["inputs"].to(device)
            targets = batch["targets"].to(device)
            ids = batch["ids"]

            # Forward pass
            outputs = model(inputs)

            # Collect results
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_ids.extend(ids)

    # Concatenate
    all_preds = np.concatenate(all_preds, axis=0)  # (N, 107, 5)
    all_targets = np.concatenate(all_targets, axis=0)  # (N, 107, 5)

    # Compute MCRMSE Metric
    # Logic matches library.loss.MCRMSELoss but implemented in numpy for full set
    # 1. Mask to seq_scored (first 68 bases)
    # 2. Select scored columns: reactivity(0), deg_Mg_pH10(1), deg_Mg_50C(3)

    seq_scored = Config.SEQ_SCORED
    # Indices corresponding to ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
    # Config.ALL_TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    scored_indices = [0, 1, 3]

    preds_scored = all_preds[:, :seq_scored, scored_indices]
    targets_scored = all_targets[:, :seq_scored, scored_indices]

    # MSE per column (averaged over N and Sequence Length)
    mse_per_col = np.mean((preds_scored - targets_scored) ** 2, axis=(0, 1))
    rmse_per_col = np.sqrt(mse_per_col)

    # MCRMSE is the mean of the RMSEs
    final_metric = np.mean(rmse_per_col)

    print(f"Final Validation Metric: {final_metric}")

    # -------------------------------------------------------------------------
    # 4. Failure Analysis
    # -------------------------------------------------------------------------
    print("\n=== Starting Failure Analysis ===")

    # Calculate error per sample for correlation analysis
    # We define sample error as the mean RMSE across the scored columns/positions for that sample
    # Shape: (N, 68, 3)
    squared_diffs = (preds_scored - targets_scored) ** 2
    # Mean over positions (axis 1) -> (N, 3)
    mse_per_sample_col = np.mean(squared_diffs, axis=1)
    rmse_per_sample_col = np.sqrt(mse_per_sample_col)
    # Mean over columns (axis 1) -> (N,)
    sample_errors = np.mean(rmse_per_sample_col, axis=1)

    # Load Validation Metadata to get features
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)

    # Create a dataframe for analysis
    # Note: val_loader order is deterministic (shuffle=False), but we use IDs to be safe
    error_df = pd.DataFrame({"id": all_ids, "error": sample_errors})

    # Merge with metadata
    analysis_df = val_df.merge(error_df, on="id", how="inner")

    # Feature Engineering for Analysis
    analysis_df["count_A"] = analysis_df["sequence"].apply(lambda x: x.count("A"))
    analysis_df["count_G"] = analysis_df["sequence"].apply(lambda x: x.count("G"))
    analysis_df["count_C"] = analysis_df["sequence"].apply(lambda x: x.count("C"))
    analysis_df["count_U"] = analysis_df["sequence"].apply(lambda x: x.count("U"))

    # Calculate Correlations
    features_to_analyze = [
        "signal_to_noise",
        "count_A",
        "count_G",
        "count_C",
        "count_U",
    ]
    correlations = {}

    print("Correlation between Model Error and Input Features:")
    for feat in features_to_analyze:
        if feat in analysis_df.columns:
            # Drop NaNs if any (signal_to_noise might have them, though unlikely in clean data)
            valid_data = analysis_df[[feat, "error"]].dropna()
            if len(valid_data) > 1:
                corr, _ = stats.pearsonr(valid_data[feat], valid_data["error"])
                correlations[feat] = corr
                print(f"  {feat}: {corr:.6f}")
            else:
                print(f"  {feat}: Insufficient data")
        else:
            print(f"  {feat}: Feature not found")

    # -------------------------------------------------------------------------
    # 5. Submission Generation
    # -------------------------------------------------------------------------
    THRESHOLD = 0.6477736930052439

    if final_metric < THRESHOLD:
        print(f"\nValidation metric ({final_metric}) passed threshold ({THRESHOLD}).")
        print("Generating submission...")

        # run_inference handles loading test data, model, predicting, and saving CSV
        run_inference(load_cached_data=True)

        if os.path.exists(Config.SUBMISSION_PATH):
            print(f"Submission successfully generated at {Config.SUBMISSION_PATH}")
        else:
            print("Error: Submission file was not created.")
    else:
        print(
            f"\nValidation metric ({final_metric}) did not meet threshold ({THRESHOLD})."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
