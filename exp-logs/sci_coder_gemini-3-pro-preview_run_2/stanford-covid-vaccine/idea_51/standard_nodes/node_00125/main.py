import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr

# Import from provided library files
from library.config import Config
from library.data import get_dataloaders
from library.model import DSPFN
from library.train import train_model
from library.utils import set_seed


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = Config.DEVICE
    print(f"Running on device: {device}")

    # 2. Training
    # We limit epochs to 5 for a fast baseline execution as requested.
    print("Starting training...")
    train_model(load_cached_data=True, epochs=5)

    # 3. Validation & Failure Analysis
    print("Starting validation and failure analysis...")

    # Load the best model
    model = DSPFN().to(device)
    model_path = Config.BEST_MODEL_PATH
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Get DataLoaders
    _, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # Containers for metrics and failure analysis
    val_ids = []
    val_errors = []

    # MCRMSE Calculation accumulators
    total_sse = torch.zeros(3, device=device)  # 3 scored columns
    total_count = torch.zeros(3, device=device)

    scored_indices = torch.tensor(Config.SCORED_INDICES, device=device)

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["inputs"].to(device)
            partner_map = batch["partner_map"].to(device)
            targets = batch["targets"].to(device)
            ids = batch["id"]

            # --- Inference (2-Pass Strategy) ---
            z = model.encode(inputs)

            # Pass 1: Zero Init
            batch_size, _, length = inputs.shape
            y_0 = torch.zeros((batch_size, 5, length), device=device)
            y_1 = model.decode(z, y_0, partner_map)

            # Pass 2: Feedback
            y_2 = model.decode(z, y_1, partner_map)

            # --- Metric Calculation Prep ---
            # Select Scored Columns: reactivity, deg_Mg_pH10, deg_Mg_50C
            # targets shape: (N, L, 5) -> permute to (N, 5, L)
            targets_p = targets.permute(0, 2, 1)

            preds_scored = torch.index_select(y_2, 1, scored_indices)
            targets_scored = torch.index_select(targets_p, 1, scored_indices)

            # Slice to scored length (68)
            eff_len = Config.SEQ_SCORED
            preds_scored = preds_scored[:, :, :eff_len]
            targets_scored = targets_scored[:, :, :eff_len]

            # Squared Error
            se = (preds_scored - targets_scored) ** 2

            # Accumulate for Global MCRMSE
            total_sse += torch.sum(se, dim=(0, 2))
            total_count += se.shape[0] * se.shape[2]

            # --- Failure Analysis Prep ---
            # Calculate RMSE per sample (averaged over scored columns and positions)
            # Shape: (N, 3, 68) -> mean over (1, 2) -> (N,)
            mse_per_sample = torch.mean(se, dim=(1, 2))
            rmse_per_sample = torch.sqrt(mse_per_sample)

            val_ids.extend(ids)
            val_errors.extend(rmse_per_sample.cpu().numpy())

    # Compute Final Metric
    col_mse = total_sse / total_count
    col_rmse = torch.sqrt(col_mse)
    final_metric = torch.mean(col_rmse).item()

    print(f"Final Validation Metric: {final_metric}")

    # --- Failure Analysis ---
    # Load metadata to correlate errors with features
    val_df = pd.read_csv(Config.VAL_CSV)

    # Create a DataFrame of errors
    error_df = pd.DataFrame({"id": val_ids, "rmse": val_errors})

    # Merge with metadata
    analysis_df = pd.merge(val_df, error_df, on="id", how="inner")

    # Calculate correlations
    # We look at signal_to_noise and mean_reactivity
    features_to_check = ["signal_to_noise", "mean_reactivity"]
    print("\n--- Failure Analysis: Error Correlations ---")
    for feat in features_to_check:
        if feat in analysis_df.columns:
            # Drop NaNs if any
            valid_data = analysis_df[[feat, "rmse"]].dropna()
            if len(valid_data) > 1:
                corr, _ = pearsonr(valid_data[feat], valid_data["rmse"])
                print(f"Correlation between Error (RMSE) and {feat}: {corr:.4f}")
            else:
                print(f"Not enough data to correlate {feat}")
        else:
            print(f"Feature {feat} not found in metadata.")

    # 4. Submission
    THRESHOLD = 0.47142532743789534

    if final_metric < THRESHOLD:
        print("\nMetric threshold met. Generating submission...")

        submission_dir = "./submission"
        os.makedirs(submission_dir, exist_ok=True)

        preds_list = []
        ids_list = []

        with torch.no_grad():
            for batch in test_loader:
                inputs = batch["inputs"].to(device)
                partner_map = batch["partner_map"].to(device)
                ids = batch["id"]

                # Inference
                z = model.encode(inputs)

                batch_size, _, length = inputs.shape
                y_0 = torch.zeros((batch_size, 5, length), device=device)
                y_1 = model.decode(z, y_0, partner_map)
                y_2 = model.decode(z, y_1, partner_map)

                # Store predictions: (N, 5, 107) -> (N, 107, 5)
                # We need all 107 positions for the submission format
                preds_np = y_2.permute(0, 2, 1).cpu().numpy()

                preds_list.append(preds_np)
                ids_list.extend(ids)

        # Concatenate all predictions
        all_preds = np.concatenate(preds_list, axis=0)  # (Total_Test, 107, 5)

        # Flatten for CSV format: One row per id_seqpos
        # Target columns in order: reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
        # This matches Config.TARGET_COLS order exactly.

        flat_preds = all_preds.reshape(-1, 5)

        # Generate ID column
        flat_ids = []
        for sample_id in ids_list:
            for i in range(Config.SEQ_LEN):
                flat_ids.append(f"{sample_id}_{i}")

        submission_df = pd.DataFrame(flat_preds, columns=Config.TARGET_COLS)
        submission_df.insert(0, "id_seqpos", flat_ids)

        save_path = os.path.join(submission_dir, "submission.csv")
        submission_df.to_csv(save_path, index=False)
        print(f"Submission saved to {save_path}")

    else:
        print(
            f"\nMetric {final_metric} did not meet threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
