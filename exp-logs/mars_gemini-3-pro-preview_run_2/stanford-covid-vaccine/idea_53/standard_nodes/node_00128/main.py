import os
import sys
import numpy as np
import pandas as pd
import torch
import scipy.stats as stats
from torch.utils.data import DataLoader

# Ensure local library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed, mcrmse_loss
from library.model import SS_DFRN, process_data, RNADataset
from library.train import train_model, predict_and_submit


def main():
    # =========================================================================
    # 1. Configuration & Setup
    # =========================================================================
    # Override Config for Fast Baseline requirements
    Config.EPOCHS = 20  # Sufficient for convergence on this dataset size
    Config.SUBMISSION_PATH = "./submission/submission.csv"

    # Ensure output directories exist
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    print(f"Configuration:")
    print(f"  Device: {Config.DEVICE}")
    print(f"  Epochs: {Config.EPOCHS}")
    print(f"  Submission Path: {Config.SUBMISSION_PATH}")

    # =========================================================================
    # 2. Training
    # =========================================================================
    print("\n=== Starting Training ===")
    # Executes the training loop defined in library.train
    # Saves the best model to Config.BEST_MODEL_PATH
    train_model()

    # =========================================================================
    # 3. Validation Evaluation
    # =========================================================================
    print("\n=== Starting Validation Evaluation ===")

    device = Config.DEVICE

    # Load the best model
    if not os.path.exists(Config.BEST_MODEL_PATH):
        raise FileNotFoundError(f"Best model not found at {Config.BEST_MODEL_PATH}")

    model = SS_DFRN().to(device)
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    model.eval()

    # Load Validation Data
    val_features, val_pidx, val_targets = process_data(
        Config.VAL_CSV, Config.VAL_CACHE, load_cached_data=True, is_test=False
    )

    val_dataset = RNADataset(val_features, val_pidx, val_targets)
    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2
    )

    # Validation Inference
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, p_idx, y in val_loader:
            x = x.to(device)
            p_idx = p_idx.to(device)
            y = y.to(device)

            # Pass 1: Zero Feedback
            pred1 = model(x, p_idx, feedback=None)

            # Pass 2: Feedback from Pass 1 (Final Prediction)
            pred2 = model(x, p_idx, feedback=pred1)

            all_preds.append(pred2.cpu())
            all_targets.append(y.cpu())

    # Concatenate results
    y_pred_tensor = torch.cat(all_preds, dim=0)
    y_true_tensor = torch.cat(all_targets, dim=0)

    # Compute Metric (MCRMSE)
    final_metric = mcrmse_loss(y_pred_tensor, y_true_tensor).item()

    # REQUIRED OUTPUT
    print(f"Final Validation Metric: {final_metric}")

    # =========================================================================
    # 4. Failure Analysis
    # =========================================================================
    print("\n=== Performing Failure Analysis ===")

    # Load validation metadata
    val_df = pd.read_csv(Config.VAL_CSV)

    # Calculate RMSE per sample (averaged over the 3 scored columns)
    # Scored columns indices: 0 (reactivity), 1 (deg_Mg_pH10), 3 (deg_Mg_50C)
    scored_indices = [0, 1, 3]
    seq_scored = Config.PRED_LEN

    y_p_numpy = y_pred_tensor.numpy()[:, :seq_scored, scored_indices]
    y_t_numpy = y_true_tensor.numpy()[:, :seq_scored, scored_indices]

    # Mean Squared Error per sample (averaged over sequence length and columns)
    mse_per_sample = np.mean((y_p_numpy - y_t_numpy) ** 2, axis=(1, 2))
    rmse_per_sample = np.sqrt(mse_per_sample)

    val_df["rmse_error"] = rmse_per_sample

    # Add GC Content feature for analysis
    val_df["gc_content"] = val_df["sequence"].apply(
        lambda s: (s.count("G") + s.count("C")) / len(s)
    )

    features_to_analyze = [
        "signal_to_noise",
        "mean_reactivity",
        "gc_content",
        "seq_length",
    ]

    print("Correlation between Error (RMSE) and Features:")
    for col in features_to_analyze:
        if col in val_df.columns:
            # Filter valid data
            subset = val_df[[col, "rmse_error"]].dropna()
            if len(subset) > 1:
                corr, _ = stats.pearsonr(subset[col], subset["rmse_error"])
                print(f"  {col}: {corr:.4f}")
            else:
                print(f"  {col}: Not enough data")
        else:
            print(f"  {col}: Column not found in metadata")

    # =========================================================================
    # 5. Submission
    # =========================================================================
    THRESHOLD = 0.47142532743789534

    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric}) is below threshold ({THRESHOLD}). Generating submission..."
        )
        # predict_and_submit uses the updated Config.SUBMISSION_PATH
        predict_and_submit()
    else:
        print(
            f"\nMetric ({final_metric}) is NOT below threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    set_seed(Config.SEED)
    main()
