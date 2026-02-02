import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import from provided libraries
from library.config import Config
from library.utils import set_seed, mcrmse_loss
from library.data import get_data, RNADataset, collate_fn
from library.model import RNAModel
from library.train import train_model, generate_submission

# 1. Configuration Override for Fast Baseline
# Reduce epochs to ensure execution finishes well within the time limit
Config.EPOCHS = 15


def main():
    # Set reproducibility
    set_seed(Config.SEED)

    print("=== Starting Runfile Execution ===")

    # 2. Train the Model
    # This function handles data loading, model init, training loop, and saving best_model.pth
    train_model()

    # 3. Validation & Metric Calculation
    print("\n=== Performing Final Validation & Failure Analysis ===")

    # Load validation data
    val_data = get_data(mode="val", load_cached_data=True)
    val_dataset = RNADataset(val_data, mode="val")
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
    )

    # Load the best model
    model = RNAModel(config=Config).to(Config.DEVICE)
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        raise FileNotFoundError(f"Model file not found at {Config.MODEL_SAVE_PATH}")

    model.load_state_dict(
        torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE)
    )
    model.eval()

    all_preds = []
    all_targets = []
    all_ids = []

    # Inference loop (No Grad for speed/memory)
    with torch.no_grad():
        for batch in val_loader:
            seq = batch["seq"].to(Config.DEVICE)
            loop = batch["loop"].to(Config.DEVICE)
            dist = batch["dist"].to(Config.DEVICE)
            targets = batch["target"].to(Config.DEVICE)
            ids = batch["id"]

            preds = model(seq, loop, dist)

            # Slice to scored length (68) for metric calculation
            preds_scored = preds[:, : Config.PRED_LENGTH, :]
            targets_scored = targets[:, : Config.PRED_LENGTH, :]

            all_preds.append(preds_scored.cpu())
            all_targets.append(targets_scored.cpu())
            all_ids.extend(ids)

    # Concatenate results
    # Shape: (N_samples, Seq_Len, Channels)
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Calculate Global MCRMSE
    final_metric = mcrmse_loss(all_preds, all_targets).item()
    print(f"Final Validation Metric: {final_metric:.20f}")

    # 4. Failure Analysis
    # Calculate MCRMSE per sample to correlate with metadata
    # Step 1: MSE per sample, per column (average over sequence length dim=1)
    mse_per_sample_col = torch.mean((all_preds - all_targets) ** 2, dim=1)  # (N, 3)
    # Step 2: RMSE per sample, per column
    rmse_per_sample_col = torch.sqrt(mse_per_sample_col)  # (N, 3)
    # Step 3: MCRMSE per sample (average over columns dim=1)
    mcrmse_per_sample = torch.mean(rmse_per_sample_col, dim=1).numpy()  # (N,)

    # Load metadata to get features
    df_val = pd.read_parquet(Config.VAL_METADATA_PATH)

    # Ensure alignment: Filter and sort metadata to match the order of all_ids
    df_val = df_val.set_index("id").loc[all_ids].reset_index()

    # Add error metric to dataframe
    df_val["error_mcrmse"] = mcrmse_per_sample

    # Feature Engineering for Analysis
    # GC Content
    df_val["gc_content"] = df_val["sequence"].apply(
        lambda x: (x.count("G") + x.count("C")) / len(x)
    )

    # Calculate Correlations
    analysis_features = ["signal_to_noise", "SN_filter", "gc_content"]
    correlations = {}

    print("\nFailure Analysis (Pearson Correlation with Sample Error):")
    for feat in analysis_features:
        if feat in df_val.columns:
            # Handle potential NaNs if any (though metadata is clean)
            if df_val[feat].dtype == object:
                df_val[feat] = pd.to_numeric(df_val[feat], errors="coerce")

            corr = df_val[feat].corr(df_val["error_mcrmse"])
            correlations[feat] = corr
            print(f"  {feat}: {corr:.4f}")

    # 5. Submission Generation
    THRESHOLD = 0.6199890971183777
    if final_metric < THRESHOLD:
        print(
            f"\nMetric {final_metric:.6f} < Threshold {THRESHOLD:.6f}. Generating submission..."
        )
        generate_submission()
    else:
        print(
            f"\nMetric {final_metric:.6f} >= Threshold {THRESHOLD:.6f}. Skipping submission."
        )


if __name__ == "__main__":
    main()
