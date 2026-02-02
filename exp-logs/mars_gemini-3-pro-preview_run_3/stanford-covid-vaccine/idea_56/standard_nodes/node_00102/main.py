import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import seed_everything, calculate_metric
from library.data import get_loaders, load_data
from library.train import run_training, predict_and_submit
from library.model import SDBR_BiGRU


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Override Config for fast baseline execution
    # The dataset is small (1728 samples), so we use full data but fewer epochs.
    # 10 epochs should be sufficient for a baseline and run very quickly.
    FAST_EPOCHS = 10

    print("Initializing Fast Baseline Run...")

    # 2. Training
    # We use the library function which handles data loading, model init, and training loop.
    best_model_path = run_training(
        epochs=FAST_EPOCHS,
        debug=False,  # Use full dataset
        load_cached_data=True,
        patience=5,
    )

    # 3. Validation Assessment
    print("\nPerforming Validation Assessment...")

    # Load validation data explicitly for analysis
    _, val_df, _ = load_data(debug=False, load_cached_data=True)
    _, val_loader, _ = get_loaders(debug=False, load_cached_data=True)

    # Load best model
    model = SDBR_BiGRU().to(device)
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Model file not found at {best_model_path}")

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    val_preds_list = []
    val_targets_list = []

    with torch.no_grad():
        for batch in val_loader:
            features = batch["features"].to(device)
            pair_index = batch["pair_index"].to(device)
            targets = batch["targets"].to(device)

            preds = model(features, pair_index)

            val_preds_list.append(preds.cpu())
            val_targets_list.append(targets.cpu())

    val_preds = torch.cat(val_preds_list, dim=0)
    val_targets = torch.cat(val_targets_list, dim=0)

    # Calculate Final Metric
    # calculate_metric handles the slicing to seq_scored (68) and filtering columns
    final_metric = calculate_metric(val_preds, val_targets)

    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    print("\nPerforming Failure Analysis...")

    # Calculate sample-wise MCRMSE for the scored columns
    seq_scored = Config.SEQ_SCORED
    scored_indices = [0, 1, 3]  # reactivity, deg_Mg_pH10, deg_Mg_50C

    # Slice sequence if necessary
    if val_preds.size(1) > seq_scored:
        preds_sliced = val_preds[:, :seq_scored, :]
    else:
        preds_sliced = val_preds

    preds_scored = preds_sliced[:, :, scored_indices]
    targets_scored = val_targets[:, :, scored_indices]

    # MSE per sample: (B, L, C) -> (B, C) -> (B,)
    # We compute RMSE per column per sample, then mean over columns
    mse_per_sample_col = torch.mean(
        (preds_scored - targets_scored) ** 2, dim=1
    )  # (B, 3)
    rmse_per_sample_col = torch.sqrt(mse_per_sample_col)
    mcrmse_per_sample = torch.mean(rmse_per_sample_col, dim=1).numpy()  # (B,)

    # Add error to dataframe
    # Ensure indices align (val_loader is not shuffled)
    val_df = val_df.copy()
    val_df["error"] = mcrmse_per_sample

    # Extract analysis features
    # Signal to Noise
    if "signal_to_noise" not in val_df.columns:
        val_df["signal_to_noise"] = 0.0

    # Nucleotide content
    val_df["pct_A"] = val_df["sequence"].apply(lambda s: s.count("A") / len(s))
    val_df["pct_G"] = val_df["sequence"].apply(lambda s: s.count("G") / len(s))
    val_df["pct_U"] = val_df["sequence"].apply(lambda s: s.count("U") / len(s))
    val_df["pct_C"] = val_df["sequence"].apply(lambda s: s.count("C") / len(s))

    # Correlations
    analysis_cols = ["signal_to_noise", "pct_A", "pct_G", "pct_U", "pct_C"]
    available_cols = [c for c in analysis_cols if c in val_df.columns]

    if available_cols:
        correlations = val_df[available_cols].corrwith(val_df["error"])
        print("Correlation between Error and Features:")
        print(correlations)
    else:
        print("No metadata features available for correlation analysis.")

    # 5. Submission
    THRESHOLD = 0.5884495377540588

    if final_metric < THRESHOLD:
        print(f"\nMetric {final_metric} < {THRESHOLD}. Generating submission...")
        predict_and_submit(best_model_path, debug=False, load_cached_data=True)
    else:
        print(f"\nMetric {final_metric} >= {THRESHOLD}. Skipping submission.")


if __name__ == "__main__":
    main()
