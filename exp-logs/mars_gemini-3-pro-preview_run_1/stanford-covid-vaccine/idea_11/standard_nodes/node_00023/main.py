import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

# Ensure the library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything
from library.data import load_and_process_data
from library.model import StructureAugmentedHybridNetwork
from library.train import train_model, generate_submission


def run_pipeline():
    # 1. Setup and Configuration
    # Override epochs for a fast baseline execution as requested
    Config.EPOCHS = 10
    seed_everything(Config.SEED)

    print(
        f"Configuration: Epochs={Config.EPOCHS}, Batch Size={Config.BATCH_SIZE}, Device={Config.DEVICE}"
    )

    # 2. Train the Model
    # This function handles the training loop and saves 'best_model.pth'
    print("\n=== Starting Training Phase ===")
    train_model()

    # 3. Validation and Metric Calculation
    print("\n=== Starting Validation Phase ===")

    # Load best model
    device = torch.device(Config.DEVICE)
    model = StructureAugmentedHybridNetwork(config=Config).to(device)
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    if not os.path.exists(best_model_path):
        print("Error: Best model not found. Exiting.")
        return

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # Load Validation Data
    datasets = load_and_process_data(load_cached_data=True)
    val_loader = DataLoader(
        datasets["val"],
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Accumulators for MCRMSE
    # We need sum of squared errors per column (3 scored columns)
    total_squared_error = torch.zeros(Config.SCORABLE_LENGTH, 3).to(device)
    total_samples = 0

    # Accumulators for Failure Analysis (per sample error)
    sample_errors = []
    sample_ids = []

    with torch.no_grad():
        for batch in val_loader:
            seq = batch["seq"].to(device)
            loop = batch["loop"].to(device)
            pair = batch["pair_index"].to(device)
            targets = batch["targets"].to(device)  # (B, 68, 3)
            ids = batch["id"]

            # Forward
            preds = model(seq, loop, pair)
            preds_scored = preds[:, : Config.SCORABLE_LENGTH, :]

            # Squared Error (B, 68, 3)
            sq_err = (preds_scored - targets) ** 2

            # Update global accumulators for MCRMSE
            total_squared_error += sq_err.sum(dim=0)
            total_samples += seq.size(0)

            # Calculate per-sample MSE (scalar) for failure analysis
            # Mean over length (68) and channels (3)
            per_sample_mse = sq_err.mean(dim=(1, 2)).cpu().numpy()

            sample_errors.extend(per_sample_mse)
            sample_ids.extend(ids)

    # Calculate Final Metric
    rmse_per_col = torch.sqrt(total_squared_error / total_samples)
    final_mcrmse = rmse_per_col.mean().item()

    # PRINT REQUIRED METRIC
    print(f"Final Validation Metric: {final_mcrmse}")

    # 4. Failure Analysis
    print("\n=== Failure Analysis ===")

    # Load metadata to get features
    df_val = pd.read_parquet(os.path.join(Config.METADATA_DIR, "val.parquet"))

    # Ensure alignment between errors and dataframe
    # The dataloader preserves order if shuffle=False, but let's be safe and map by ID
    error_map = dict(zip(sample_ids, sample_errors))

    # Create analysis dataframe
    analysis_data = []
    for _, row in df_val.iterrows():
        sid = str(row["id"])
        if sid in error_map:
            err = error_map[sid]

            # Extract features
            seq = row["sequence"]
            struct = row["structure"]

            feats = {
                "error_mse": err,
                "signal_to_noise": row.get("signal_to_noise", 0),
                "SN_filter": int(row.get("SN_filter", 0)),
                "len_A": seq.count("A"),
                "len_G": seq.count("G"),
                "len_C": seq.count("C"),
                "len_U": seq.count("U"),
                "paired_bases": struct.count("(") + struct.count(")"),
            }
            analysis_data.append(feats)

    df_analysis = pd.DataFrame(analysis_data)

    # Calculate correlations
    if not df_analysis.empty:
        print("Correlation between Error Magnitude (MSE) and Features:")
        correlations = (
            df_analysis.corr()["error_mse"]
            .drop("error_mse")
            .sort_values(ascending=False)
        )
        print(correlations)
    else:
        print("Could not perform failure analysis: Data mismatch.")

    # 5. Submission Generation
    # Threshold: 0.6226052641868591
    THRESHOLD = 0.6226052641868591

    if final_mcrmse < THRESHOLD:
        print(
            f"\nMetric ({final_mcrmse}) is better than threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission()
    else:
        print(
            f"\nMetric ({final_mcrmse}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    run_pipeline()
