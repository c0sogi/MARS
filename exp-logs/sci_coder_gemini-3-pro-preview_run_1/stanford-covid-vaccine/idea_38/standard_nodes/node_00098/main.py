import os
import sys
import torch
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, calculate_mcrmse
from library.data import get_dataloaders
from library.model import RNAModel
from library.train import Trainer


def run_failure_analysis(model, val_loader, device):
    """
    Performs failure analysis by correlating model error with input features.
    """
    print("\nPerforming Failure Analysis...")

    # Load validation metadata to get features like signal_to_noise
    val_meta_path = Config.VAL_METADATA
    if not os.path.exists(val_meta_path):
        print("Validation metadata not found. Skipping failure analysis.")
        return

    df_val = pd.read_parquet(val_meta_path)

    # Ensure model is in eval mode
    model.eval()

    preds_list = []
    targets_list = []

    # Run inference on validation set
    # Note: val_loader must not be shuffled to match df_val order
    with torch.no_grad():
        for batch in val_loader:
            sequences = batch["sequence"].to(device)
            loop_types = batch["loop_type"].to(device)
            pair_dists = batch["pair_dist"].to(device)
            targets = batch["target"].to(device)

            # Forward pass
            preds = model(sequences, loop_types, pair_dists)

            # Slice to scored positions
            preds_scored = preds[:, : Config.SEQ_SCORED, :]
            targets_scored = targets[:, : Config.SEQ_SCORED, :]

            preds_list.append(preds_scored.cpu().numpy())
            targets_list.append(targets_scored.cpu().numpy())

    # Concatenate all batches
    preds_arr = np.concatenate(preds_list, axis=0)  # (N_samples, 68, 3)
    targets_arr = np.concatenate(targets_list, axis=0)  # (N_samples, 68, 3)

    # Calculate RMSE per sample (averaging over positions and targets)
    # Error = sqrt(mean((y - y_hat)^2))
    squared_diff = (preds_arr - targets_arr) ** 2
    mse_per_sample = np.mean(squared_diff, axis=(1, 2))
    rmse_per_sample = np.sqrt(mse_per_sample)

    # Add error to dataframe
    # We assume the loader preserves order and length matches
    if len(df_val) == len(rmse_per_sample):
        df_val["error_rmse"] = rmse_per_sample

        # Feature Engineering for Analysis
        df_val["len_A"] = df_val["sequence"].apply(lambda x: x.count("A"))
        df_val["len_G"] = df_val["sequence"].apply(lambda x: x.count("G"))
        df_val["len_C"] = df_val["sequence"].apply(lambda x: x.count("C"))
        df_val["len_U"] = df_val["sequence"].apply(lambda x: x.count("U"))

        analysis_features = [
            "signal_to_noise",
            "SN_filter",
            "len_A",
            "len_G",
            "len_C",
            "len_U",
        ]

        print(f"{'Feature':<20} | {'Correlation with Error':<20}")
        print("-" * 45)

        for feat in analysis_features:
            if feat in df_val.columns:
                # Drop NaNs just in case
                subset = df_val[[feat, "error_rmse"]].dropna()
                if len(subset) > 1:
                    corr, _ = pearsonr(subset[feat], subset["error_rmse"])
                    print(f"{feat:<20} | {corr:.4f}")
    else:
        print(
            f"Mismatch in validation set size: DF {len(df_val)} vs Loader {len(rmse_per_sample)}"
        )


def main():
    # 1. Setup
    # Override Config for a fast baseline run
    Config.EPOCHS = 15  # Reduced from 20 to ensure quick execution

    Config.setup()
    set_seed(Config.SEED)

    print(f"Running with Device: {Config.DEVICE}")
    print(f"Epochs: {Config.EPOCHS}")

    # 2. Data Loading
    print("Loading Data...")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Model Initialization
    print("Initializing Model...")
    model = RNAModel(config=Config)
    model.to(Config.DEVICE)

    # 4. Training
    print("Starting Training...")
    trainer = Trainer(model, train_loader, val_loader, test_loader, Config)
    trainer.fit()

    # 5. Final Validation Metric
    print("Computing Final Validation Metric...")
    final_metric = trainer.validate()
    # REQUIRED FORMAT: Do not modify
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    run_failure_analysis(model, val_loader, Config.DEVICE)

    # 7. Submission
    # Threshold check
    THRESHOLD = 0.6199890971183777

    if final_metric < THRESHOLD:
        print(
            f"\nMetric {final_metric} is better than threshold {THRESHOLD}. Generating submission..."
        )
        trainer.generate_submission()
    else:
        print(
            f"\nMetric {final_metric} did not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
