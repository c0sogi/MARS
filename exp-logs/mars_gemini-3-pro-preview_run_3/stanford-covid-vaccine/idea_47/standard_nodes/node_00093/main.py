import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings
from torch.utils.data import DataLoader

# Import provided library components
from library.config import Config
from library.engine import train_and_evaluate, generate_submission, set_seed
from library.dataset import RNADataset
from library.model import DeepStabilizedBiGRU
from library.metrics import scored_mcrmse

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def main():
    # =========================================================================
    # 1. Configuration & Setup
    # =========================================================================
    # Override Config for Fast Baseline execution
    # 1728 samples is small, so we use the full dataset but limit epochs for speed.
    Config.NUM_EPOCHS = 10
    Config.PATIENCE = 3

    # Set seeds for reproducibility
    set_seed(Config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =========================================================================
    # 2. Training
    # =========================================================================
    # Train the model and save the best checkpoint based on validation score
    # We use max_samples=None to use the full dataset (1728 samples)
    train_and_evaluate(load_cached_data=True, max_samples=None)

    # =========================================================================
    # 3. Validation & Metric Calculation
    # =========================================================================
    # Load the best model
    model = DeepStabilizedBiGRU().to(device)
    if not os.path.exists(Config.MODEL_PATH):
        print(f"Error: Model file not found at {Config.MODEL_PATH}")
        sys.exit(1)

    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.eval()

    # Load Validation Dataset
    val_dataset = RNADataset(split="val", load_cached_data=True)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Run Inference on Validation Set
    all_preds = []
    all_targets = []
    all_ids = []

    with torch.no_grad():
        for batch in val_loader:
            features = batch["features"].to(device)
            adjacency = batch["adjacency"].to(device)
            targets = batch["targets"].to(device)
            ids = batch["id"]

            outputs = model(features, adjacency)

            all_preds.append(outputs.cpu())
            all_targets.append(targets.cpu())
            all_ids.extend(ids)

    # Concatenate results
    global_preds = torch.cat(all_preds, dim=0)
    global_targets = torch.cat(all_targets, dim=0)

    # Calculate Final Metric
    final_metric = scored_mcrmse(global_preds, global_targets)
    print(f"Final Validation Metric: {final_metric}")

    # =========================================================================
    # 4. Failure Analysis
    # =========================================================================
    print("Performing Failure Analysis...")

    # 1. Calculate RMSE per sample for the scored columns
    # Scored columns in Config.TARGET_COLS are at indices 0, 1, 3
    # Config.TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    # Config.SCORED_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
    scored_indices = [0, 1, 3]

    # Slice to scored length (68) and scored columns
    preds_sliced = global_preds[:, : Config.PRED_LEN, scored_indices]
    targets_sliced = global_targets[:, : Config.PRED_LEN, scored_indices]

    # Compute MSE per sample (averaging over sequence and channels)
    # Shape: (N, 68, 3) -> Mean over dims 1,2 -> (N,)
    mse_per_sample = torch.mean((preds_sliced - targets_sliced) ** 2, dim=(1, 2))
    rmse_per_sample = torch.sqrt(mse_per_sample).numpy()

    # 2. Load Metadata to correlate with features
    val_df = pd.read_parquet(Config.VAL_METADATA)

    # Create analysis dataframe
    # Note: val_loader preserves order, and RNADataset loads sequentially from metadata
    analysis_df = pd.DataFrame({"id": all_ids, "error": rmse_per_sample})

    # Merge with metadata
    merged_df = pd.merge(analysis_df, val_df, on="id")

    # 3. Compute Feature Correlations
    # Extract nucleotide content
    merged_df["pct_A"] = merged_df["sequence"].apply(lambda x: x.count("A") / len(x))
    merged_df["pct_G"] = merged_df["sequence"].apply(lambda x: x.count("G") / len(x))
    merged_df["pct_C"] = merged_df["sequence"].apply(lambda x: x.count("C") / len(x))
    merged_df["pct_U"] = merged_df["sequence"].apply(lambda x: x.count("U") / len(x))

    # Select columns for correlation
    corr_cols = [
        "error",
        "signal_to_noise",
        "SN_filter",
        "pct_A",
        "pct_G",
        "pct_C",
        "pct_U",
    ]
    # Ensure columns exist (SN_filter might be int)
    existing_cols = [c for c in corr_cols if c in merged_df.columns]

    correlations = merged_df[existing_cols].corr()["error"].drop("error")
    print("Correlation between Error and Features:")
    print(correlations)

    # =========================================================================
    # 5. Submission Generation
    # =========================================================================
    THRESHOLD = 0.5884495377540588

    if final_metric < THRESHOLD:
        print(f"Metric {final_metric} < {THRESHOLD}. Generating submission...")
        generate_submission(load_cached_data=True)
    else:
        print(f"Metric {final_metric} >= {THRESHOLD}. Submission generation skipped.")


if __name__ == "__main__":
    main()
