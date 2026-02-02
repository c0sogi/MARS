import sys
import os
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Ensure the current directory is in the path for module imports
sys.path.append(os.getcwd())

from library.config import Config
from library.engine import Engine
from library.utils import seed_everything, MCRMSE
from library.model import ADSRN
from library.data import get_loaders


def run():
    # 1. Setup and Configuration
    # Set seed for reproducibility
    seed_everything(Config.SEED)

    # Override Config for a fast baseline run (limit epochs)
    # The dataset is small (~1700 train samples), so 20 epochs is sufficient for a baseline
    # and ensures execution within the 2-hour limit.
    Config.EPOCHS = 20

    print("Starting Fast Baseline Pipeline...")
    print(f"Device: {Config.DEVICE}")

    # 2. Training
    # Engine.run_training handles the loop, validation monitoring, and saving the best model.
    best_model_path = Engine.run_training()

    # 3. Final Validation Assessment
    print("\n" + "=" * 40)
    print("Running Final Validation Assessment")
    print("=" * 40)

    device = torch.device(Config.DEVICE)
    model = ADSRN().to(device)

    # Load the best model weights
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Best model not found at {best_model_path}")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # Get validation loader
    _, val_loader, _ = get_loaders(load_cached_data=True)

    # Initialize metric calculator
    metric = MCRMSE()

    # Storage for failure analysis
    all_preds = []
    all_targets = []

    # Inference Loop
    with torch.no_grad():
        for inputs, targets, partner_indices in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            partner_indices = partner_indices.to(device)

            # Forward pass (get final prediction y_2)
            y_2, _ = model(inputs, partner_indices)

            # Update global metric
            metric.update(y_2, targets)

            # Store data for failure analysis (move to CPU)
            all_preds.append(y_2.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # Compute and print the final metric
    final_score = metric.compute()
    print(f"Final Validation Metric: {final_score}")

    # 4. Failure Analysis
    print("\n" + "=" * 40)
    print("Failure Analysis")
    print("=" * 40)

    # Concatenate all batches
    preds_arr = np.concatenate(all_preds, axis=0)
    targets_arr = np.concatenate(all_targets, axis=0)

    # Get IDs from the dataset to merge with metadata
    val_ids = val_loader.dataset.ids

    # Calculate per-sample MCRMSE for the scored columns
    # Scored columns indices: 0 (reactivity), 1 (deg_Mg_pH10), 3 (deg_Mg_50C)
    scored_indices = [0, 1, 3]
    seq_scored = Config.PRED_LEN  # 68

    per_sample_errors = []

    for i in range(len(val_ids)):
        sample_rmses = []
        for idx in scored_indices:
            # Slice to scored length
            p = preds_arr[i, :seq_scored, idx]
            t = targets_arr[i, :seq_scored, idx]
            # RMSE for this column
            mse = np.mean((p - t) ** 2)
            sample_rmses.append(np.sqrt(mse))
        # MCRMSE is the mean of the column RMSEs
        per_sample_errors.append(np.mean(sample_rmses))

    # Create DataFrame for analysis
    df_errors = pd.DataFrame({"id": val_ids, "error": per_sample_errors})

    # Load validation metadata to get features like Signal-to-Noise
    if os.path.exists(Config.VAL_CSV):
        df_meta = pd.read_csv(Config.VAL_CSV)

        # Merge errors with metadata
        df_analysis = pd.merge(df_errors, df_meta, on="id", how="left")

        # Define features to correlate with error
        features_to_check = ["signal_to_noise", "mean_reactivity"]

        # Add a derived feature: A-content (Adenine count)
        if "sequence" in df_analysis.columns:
            df_analysis["count_A"] = df_analysis["sequence"].apply(
                lambda x: x.count("A")
            )
            features_to_check.append("count_A")

        print(f"{'Feature':<25} | {'Correlation with Error':<25}")
        print("-" * 55)

        for feat in features_to_check:
            if feat in df_analysis.columns:
                # Filter out NaNs
                valid_data = df_analysis[[feat, "error"]].dropna()
                if len(valid_data) > 1 and valid_data[feat].std() > 0:
                    corr, _ = pearsonr(valid_data[feat], valid_data["error"])
                    print(f"{feat:<25} | {corr:.6f}")
                else:
                    print(f"{feat:<25} | N/A (Constant or NaN)")
    else:
        print(
            "Validation metadata CSV not found. Skipping detailed correlation analysis."
        )

    # 5. Submission Generation
    THRESHOLD = 0.47142532743789534

    print("\n" + "=" * 40)
    print("Submission Logic")
    print("=" * 40)

    if final_score < THRESHOLD:
        print(f"Validation Score ({final_score}) < Threshold ({THRESHOLD}).")
        print("Generating submission file...")
        Engine.generate_submission(best_model_path)
    else:
        print(f"Validation Score ({final_score}) >= Threshold ({THRESHOLD}).")
        print("Skipping submission generation.")


if __name__ == "__main__":
    run()
