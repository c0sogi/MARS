import os
import sys
import numpy as np
import pandas as pd
import torch

# Import from provided libraries
from library.config import Config
from library.utils import seed_everything, mcrmse_numpy
from library.engine import train_pipeline, predict_and_submit
from library.data import get_dataloaders
from library.model import SDBR_BiGRU


def run():
    # 1. Setup Environment
    seed_everything(Config.SEED)
    Config.setup_directories()

    # 2. Train Model
    # Using 10 epochs for a fast baseline as requested.
    # The dataset size (1728 samples) allows this to run very quickly.
    print("Starting training pipeline...")
    train_pipeline(epochs=10, batch_size=64, debug=False)

    # 3. Validation & Metric Calculation
    print("Evaluating best model on validation set...")
    device = torch.device(Config.DEVICE)

    # Load the best model saved during training
    model = SDBR_BiGRU().to(device)
    if not os.path.exists(Config.BEST_MODEL_PATH):
        raise FileNotFoundError(f"Best model not found at {Config.BEST_MODEL_PATH}")

    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    model.eval()

    # Get Validation Data
    # load_cached_data=True ensures we use preprocessed data if available
    _, val_loader, _ = get_dataloaders(load_cached_data=True, batch_size=64)

    all_preds = []
    all_targets = []

    # Inference loop (No Grad for memory/speed optimization)
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["inputs"].to(device)
            pair_idx = batch["pair_index"].to(device)
            pair_mask = batch["pair_mask"].to(device)
            targets = batch["targets"].numpy()  # Keep targets on CPU

            preds = model(inputs, pair_idx, pair_mask)
            preds = preds.cpu().numpy()

            all_preds.append(preds)
            all_targets.append(targets)

    # Concatenate results
    y_pred = np.concatenate(all_preds, axis=0)  # (N, 107, 5)
    y_true = np.concatenate(all_targets, axis=0)  # (N, 68, 5)

    # Compute Metric
    # mcrmse_numpy handles slicing y_pred to 68 and selecting scored columns
    final_metric = mcrmse_numpy(y_true, y_pred)

    # REQUIRED PRINT
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    print("\nRunning Failure Analysis...")

    # Calculate per-sample error
    # Slice predictions to match scored length (68)
    y_pred_sliced = y_pred[:, : Config.SEQ_SCORED, :]

    # Identify indices of scored columns
    scored_indices = [Config.TARGET_COLS.index(col) for col in Config.SCORED_TARGETS]

    # Filter to scored columns
    y_true_scored = y_true[:, :, scored_indices]
    y_pred_scored = y_pred_sliced[:, :, scored_indices]

    # Compute MCRMSE per sample:
    # 1. MSE per sample per column (mean over sequence length)
    mse_per_sample_col = np.mean(
        (y_true_scored - y_pred_scored) ** 2, axis=1
    )  # Shape: (N, 3)
    # 2. RMSE per sample per column
    rmse_per_sample_col = np.sqrt(mse_per_sample_col)
    # 3. Mean over columns -> Scalar error per sample
    error_per_sample = np.mean(rmse_per_sample_col, axis=1)  # Shape: (N,)

    # Load Validation Metadata
    val_df = pd.read_parquet(Config.VAL_METADATA)

    # Ensure alignment (val_loader is not shuffled, so order matches)
    if len(val_df) != len(error_per_sample):
        print(
            f"Warning: Metadata length {len(val_df)} != Predictions length {len(error_per_sample)}"
        )

    val_df["error"] = error_per_sample

    # Feature Engineering for Correlation
    val_df["pct_A"] = val_df["sequence"].apply(lambda x: x.count("A") / len(x))
    val_df["pct_G"] = val_df["sequence"].apply(lambda x: x.count("G") / len(x))
    val_df["pct_U"] = val_df["sequence"].apply(lambda x: x.count("U") / len(x))
    val_df["pct_C"] = val_df["sequence"].apply(lambda x: x.count("C") / len(x))

    # Calculate Correlations
    corr_features = ["signal_to_noise", "SN_filter", "pct_A", "pct_G", "pct_U", "pct_C"]
    correlations = val_df[corr_features].corrwith(val_df["error"])

    print("Correlation between Error and Features:")
    print(correlations)

    # 5. Submission Logic
    THRESHOLD = 0.5884495377540588

    if final_metric < THRESHOLD:
        print(
            f"\nMetric {final_metric} meets threshold ({THRESHOLD}). Generating submission..."
        )
        predict_and_submit(batch_size=64)
    else:
        print(
            f"\nMetric {final_metric} does not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    run()
