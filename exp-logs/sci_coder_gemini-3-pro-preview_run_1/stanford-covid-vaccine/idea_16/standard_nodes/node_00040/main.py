import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

# Add current directory to sys.path to ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.engine import train_model, predict_and_submit, validate
from library.dataset import RNADataset
from library.model import StructureShortcutResBiGRU
from library.utils import seed_everything


def perform_failure_analysis(model, device):
    """
    Performs failure analysis on the validation set.
    Calculates per-sample error and correlates with metadata.
    """
    print("\nPerforming Failure Analysis...")

    # 1. Load Validation Data Metadata
    # We use the parquet file generated in the metadata step
    if not os.path.exists(Config.VAL_PATH):
        print(f"Warning: Validation metadata not found at {Config.VAL_PATH}")
        return

    val_df = pd.read_parquet(Config.VAL_PATH)

    # 2. Load Validation Dataset/Loader for predictions
    # We reuse the dataset class to ensure consistent preprocessing
    val_dataset = RNADataset(split="val", load_cached_data=True)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Get Predictions and Targets
    model.eval()
    all_preds = []
    all_targets = []
    all_ids = []

    with torch.no_grad():
        for batch in val_loader:
            sequence = batch["sequence"].to(device)
            loop_type = batch["loop_type"].to(device)
            pair_index = batch["pair_index"].to(device)
            pair_dist = batch["pair_dist"].to(device)
            targets = batch["targets"].to(device)
            ids = batch["id"]

            outputs = model(sequence, loop_type, pair_index, pair_dist)

            # Slice to scored length (first 68 positions)
            outputs_scored = outputs[:, : Config.PRED_LEN, :]

            all_preds.append(outputs_scored.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_ids.extend(ids)

    y_pred = np.concatenate(all_preds, axis=0)  # Shape: (N, 68, 3)
    y_true = np.concatenate(all_targets, axis=0)  # Shape: (N, 68, 3)

    # 4. Calculate Per-Sample Error (RMSE)
    # We calculate the MSE for each sample across all scored positions and targets, then sqrt
    mse_per_sample = np.mean((y_true - y_pred) ** 2, axis=(1, 2))
    rmse_per_sample = np.sqrt(mse_per_sample)

    # Create a DataFrame for analysis
    error_df = pd.DataFrame({"id": all_ids, "error": rmse_per_sample})

    # Merge with metadata to get features like signal_to_noise
    # We perform a left join on 'id'
    analysis_df = pd.merge(error_df, val_df, on="id", how="left")

    # 5. Feature Engineering for Correlation
    # Calculate GC content and nucleotide counts if not already present or just to be sure
    if "sequence" in analysis_df.columns:
        analysis_df["len_A"] = analysis_df["sequence"].apply(lambda x: x.count("A"))
        analysis_df["len_U"] = analysis_df["sequence"].apply(lambda x: x.count("U"))
        analysis_df["len_G"] = analysis_df["sequence"].apply(lambda x: x.count("G"))
        analysis_df["len_C"] = analysis_df["sequence"].apply(lambda x: x.count("C"))
        analysis_df["gc_content"] = (
            analysis_df["len_G"] + analysis_df["len_C"]
        ) / analysis_df["seq_length"]

    # 6. Calculate Correlations
    features_to_check = ["signal_to_noise", "SN_filter", "len_A", "len_U", "gc_content"]
    print("Correlation between Model Error (RMSE) and Features:")

    for feat in features_to_check:
        if feat in analysis_df.columns:
            # Drop NaNs to avoid errors in pearsonr
            valid_data = analysis_df[[feat, "error"]].dropna()
            if len(valid_data) > 1:
                corr, _ = pearsonr(valid_data[feat], valid_data["error"])
                print(f"  {feat}: {corr:.4f}")
            else:
                print(f"  {feat}: Not enough data")
        else:
            print(f"  {feat}: Feature missing")


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Adjust Config for fast baseline execution
    # We set epochs to 15 to ensure execution within 2 hours while allowing convergence
    Config.EPOCHS = 15
    Config.DEBUG = False  # Ensure we use the full dataset for a valid baseline

    print("Configuration:")
    print(f"  Device: {Config.DEVICE}")
    print(f"  Epochs: {Config.EPOCHS}")
    print(f"  Batch Size: {Config.BATCH_SIZE}")
    print(f"  Model: StructureShortcutResBiGRU")

    # 2. Train Model
    # This function handles the training loop, validation monitoring, and saving the best model
    print("\n=== Starting Training Pipeline ===")
    train_model()

    # 3. Load Best Model for Final Validation
    print("\n=== Validating Best Model ===")
    model = StructureShortcutResBiGRU()
    # Load the best checkpoint saved during training
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        raise FileNotFoundError(f"Model file not found at {Config.MODEL_SAVE_PATH}")

    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.to(device)
    model.eval()

    # Load Validation Loader
    val_dataset = RNADataset(split="val", load_cached_data=True)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Compute Metric
    val_mcrmse = validate(model, val_loader, device)

    # REQUIRED OUTPUT FORMAT
    # Print the full precision metric
    print(f"Final Validation Metric: {val_mcrmse}")

    # 4. Failure Analysis
    perform_failure_analysis(model, device)

    # 5. Conditional Submission
    # Threshold defined in the task
    threshold = 0.6226052641868591

    if val_mcrmse < threshold:
        print(
            f"\nValidation metric ({val_mcrmse}) is better than threshold ({threshold}). Generating submission..."
        )
        # This function loads the best model again internally and generates the CSV
        predict_and_submit()
    else:
        print(
            f"\nValidation metric ({val_mcrmse}) did not beat threshold ({threshold}). Skipping submission."
        )


if __name__ == "__main__":
    main()
