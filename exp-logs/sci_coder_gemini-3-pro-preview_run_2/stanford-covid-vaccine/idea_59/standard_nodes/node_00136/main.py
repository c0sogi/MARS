import os
import sys
import torch
import numpy as np
import pandas as pd
import scipy.stats as stats

# Import library modules
import library.config

# Patch epochs for fast baseline execution
library.config.EPOCHS = 10

from library.config import (
    WORKING_DIR,
    SEED,
    set_seed,
    NUM_TARGETS,
    SCORED_COLS_INDICES,
    PRED_LEN,
    BATCH_SIZE,
)
from library.train import train_model, generate_submission
from library.data import get_loaders, Preprocessor
from library.model import DSRDN


def calculate_per_sample_mcrmse(preds, targets):
    """
    Calculates MCRMSE for each sample in the batch.
    preds, targets: [B, 5, L]
    Returns: numpy array of shape [B]
    """
    # Select scored columns: indices 0, 1, 3
    scored_indices = [0, 1, 3]

    # Slice data
    preds_scored = preds[:, scored_indices, :PRED_LEN]
    targets_scored = targets[:, scored_indices, :PRED_LEN]

    # Squared Error: [B, 3, 68]
    se = (preds_scored - targets_scored) ** 2

    # Mean Squared Error per column per sample: [B, 3]
    # Average over length (dim 2)
    mse_per_col = se.mean(axis=2)

    # RMSE per column: [B, 3]
    rmse_per_col = np.sqrt(mse_per_col)

    # Mean over columns: [B]
    mcrmse_per_sample = rmse_per_col.mean(axis=1)

    return mcrmse_per_sample


def run_failure_analysis(device):
    print("\nRunning Failure Analysis...")

    # 1. Load Best Model
    model_path = os.path.join(WORKING_DIR, "best_model.pth")
    if not os.path.exists(model_path):
        print("Model not found for failure analysis.")
        return

    model = DSRDN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 2. Load Validation Data
    preprocessor = Preprocessor()
    data = preprocessor.process(load_cached_data=True)
    _, val_loader, _ = get_loaders(data, batch_size=BATCH_SIZE, debug=False)

    # Load Metadata to correlate
    val_csv_path = "./metadata/val.csv"
    if not os.path.exists(val_csv_path):
        print("Validation metadata not found.")
        return
    val_df = pd.read_csv(val_csv_path)

    # 3. Inference and Error Calculation
    all_errors = []

    with torch.no_grad():
        for inputs, targets, pairs in val_loader:
            inputs = inputs.to(device)
            targets = targets.cpu().numpy()  # Move targets to CPU for calc
            pairs = pairs.to(device)
            B, _, L = inputs.shape

            # Inference with Recycling
            z = model.forward_static(inputs)

            # Pass 1
            y_prev_0 = torch.zeros((B, NUM_TARGETS, L), device=device)
            e_fb_1 = model.forward_feedback(y_prev_0)
            preds_1 = model.forward_interaction(z, e_fb_1, pairs)

            # Pass 2
            e_fb_2 = model.forward_feedback(preds_1)
            preds_2 = model.forward_interaction(z, e_fb_2, pairs)

            preds_np = preds_2.cpu().numpy()

            # Calculate error
            batch_errors = calculate_per_sample_mcrmse(preds_np, targets)
            all_errors.append(batch_errors)

    all_errors = np.concatenate(all_errors)

    # Ensure lengths match (loader might drop last batch if configured, but here drop_last=False default)
    if len(all_errors) != len(val_df):
        print(
            f"Warning: Mismatch in validation set size. Errors: {len(all_errors)}, DF: {len(val_df)}"
        )
        # Truncate to min length to proceed
        min_len = min(len(all_errors), len(val_df))
        all_errors = all_errors[:min_len]
        val_df = val_df.iloc[:min_len]

    val_df["mcrmse"] = all_errors

    # 4. Correlation Analysis
    # Features to analyze
    features = ["signal_to_noise", "seq_length", "mean_reactivity", "SN_filter"]
    # Check which exist
    available_features = [f for f in features if f in val_df.columns]

    print(f"{'Feature':<20} | {'Correlation':<12} | {'P-Value':<12}")
    print("-" * 50)

    for feat in available_features:
        # Drop NaNs for correlation
        subset = val_df[[feat, "mcrmse"]].dropna()
        if len(subset) > 1:
            corr, p_val = stats.pearsonr(subset[feat], subset["mcrmse"])
            print(f"{feat:<20} | {corr:<12.4f} | {p_val:<12.4e}")
        else:
            print(f"{feat:<20} | N/A          | N/A")

    # Additional check: Error by SN_filter
    if "SN_filter" in val_df.columns:
        passed = val_df[val_df["SN_filter"] == 1]["mcrmse"].mean()
        failed = val_df[val_df["SN_filter"] == 0]["mcrmse"].mean()
        print(f"\nMean Error - Passed SN Filter: {passed:.4f}")
        print(f"Mean Error - Failed SN Filter: {failed:.4f}")


def main():
    # 1. Setup
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Train Model
    # Using debug=False to use the full dataset, but relying on reduced EPOCHS=10 for speed.
    val_score = train_model(debug=False)

    # 3. Print Final Metric
    print(f"Final Validation Metric: {val_score}")

    # 4. Failure Analysis
    run_failure_analysis(device)

    # 5. Submission
    THRESHOLD = 0.47142532743789534
    if val_score < THRESHOLD:
        generate_submission(debug=False)
    else:
        print(
            f"Validation score {val_score} did not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
