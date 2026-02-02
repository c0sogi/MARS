import sys
import os
import torch
import pandas as pd
import numpy as np
import scipy.stats as stats

# Ensure library is in path
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, mcrmse
from library.data import get_dataloaders
from library.model import StabilizedWideBiLSTM
from library.engine import train_model, generate_submission


def run():
    # 1. Setup
    seed_everything(Config.SEED)
    print(f"Running Project: {Config.PROJECT_NAME}")
    print(f"Device: {Config.DEVICE}")

    # 2. Data Loading
    # Using cached data if available for speed
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Model Initialization
    print("Initializing model...")
    model = StabilizedWideBiLSTM()

    # 4. Training
    # The train_model function handles the loop, validation per epoch, and saving best model
    print("Starting training...")
    train_model(model, train_loader, val_loader)

    # 5. Final Validation Assessment
    print("\nPerforming Final Validation Assessment...")

    # Load the best model weights
    if os.path.exists(Config.BEST_MODEL_PATH):
        model.load_state_dict(
            torch.load(Config.BEST_MODEL_PATH, map_location=Config.DEVICE)
        )
        print("Loaded best model weights.")
    else:
        print("Warning: Best model file not found. Using current model weights.")

    model.to(Config.DEVICE)
    model.eval()

    all_preds = []
    all_targets = []

    # Inference on Validation Set
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(Config.DEVICE)
            # targets: (Batch, Seq_Len, Num_Targets)

            outputs = model(inputs)  # (Batch, Seq_Len, Num_Targets)

            # Slice to scored length (0..67)
            outputs_scored = outputs[:, : Config.SCORED_LEN, :]
            targets_scored = targets[:, : Config.SCORED_LEN, :]

            all_preds.append(outputs_scored.cpu().numpy())
            all_targets.append(targets_scored.cpu().numpy())

    # Concatenate results
    all_preds_np = np.concatenate(all_preds, axis=0)  # (N_samples, 68, 3)
    all_targets_np = np.concatenate(all_targets, axis=0)  # (N_samples, 68, 3)

    # Calculate MCRMSE
    # Flatten: (N_samples * 68, 3)
    flat_preds = all_preds_np.reshape(-1, Config.NUM_TARGETS)
    flat_targets = all_targets_np.reshape(-1, Config.NUM_TARGETS)

    final_metric = mcrmse(flat_targets, flat_preds)

    # REQUIRED OUTPUT
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("\n=== Failure Analysis ===")

    # Calculate error per sample (RMSE)
    # (N_samples, 68, 3) -> (N_samples,)
    squared_diff = (all_preds_np - all_targets_np) ** 2
    mse_per_sample = np.mean(squared_diff, axis=(1, 2))
    rmse_per_sample = np.sqrt(mse_per_sample)

    # Get IDs from the validation dataset
    # val_loader.dataset is an RNADataset which has .ids attribute
    val_ids = val_loader.dataset.ids

    # Load metadata to get features
    if os.path.exists(Config.VAL_METADATA):
        val_meta_df = pd.read_parquet(Config.VAL_METADATA)

        # Create analysis dataframe
        analysis_df = pd.DataFrame({"id": val_ids, "error_rmse": rmse_per_sample})

        # Merge with metadata
        analysis_df = analysis_df.merge(val_meta_df, on="id", how="left")

        # Define features to analyze
        features_to_analyze = ["signal_to_noise", "SN_filter"]

        # Add derived features if sequence is available
        if "sequence" in analysis_df.columns:
            analysis_df["seq_len"] = analysis_df["sequence"].apply(len)
            analysis_df["gc_content"] = analysis_df["sequence"].apply(
                lambda x: (x.count("G") + x.count("C")) / len(x) if len(x) > 0 else 0
            )
            features_to_analyze.extend(["seq_len", "gc_content"])

        print("Correlation between Model Error (RMSE) and Input Features:")
        for feat in features_to_analyze:
            if feat in analysis_df.columns:
                # Ensure numeric
                if pd.api.types.is_numeric_dtype(analysis_df[feat]):
                    # Drop NaNs
                    subset = analysis_df[[feat, "error_rmse"]].dropna()
                    if len(subset) > 1:
                        corr, p_val = stats.pearsonr(subset[feat], subset["error_rmse"])
                        print(
                            f"  {feat}: Correlation = {corr:.4f} (p-value = {p_val:.4e})"
                        )
                else:
                    print(f"  {feat}: Not numeric, skipping correlation.")
    else:
        print(
            "Validation metadata not found, skipping failure analysis feature correlation."
        )

    # 7. Submission Generation
    # Threshold: 0.6176461577
    SUBMISSION_THRESHOLD = 0.6176461577

    if final_metric < SUBMISSION_THRESHOLD:
        print(
            f"\nMetric ({final_metric:.6f}) is better than threshold ({SUBMISSION_THRESHOLD}). Generating submission..."
        )
        generate_submission(model, test_loader)
    else:
        print(
            f"\nMetric ({final_metric:.6f}) did not meet threshold ({SUBMISSION_THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    run()
