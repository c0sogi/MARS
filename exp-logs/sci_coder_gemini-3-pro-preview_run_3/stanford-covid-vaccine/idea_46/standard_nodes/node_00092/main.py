import sys
import os
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Add current directory to path to ensure imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed, compute_mcrmse
from library.data import get_dataloaders
from library.model import DeepStabilizedBiGRU
from library.train import run_training


def main():
    # 1. Configuration Overrides for Fast Baseline
    # We reduce epochs to ensure execution within time limits while maintaining performance.
    # The dataset (1.7k samples) is small enough that we don't need to subsample data
    # to be "fast", but we limit epochs to a reasonable number for convergence.
    Config.NUM_EPOCHS = 15
    Config.EARLY_STOPPING_PATIENCE = 5

    # Ensure directories exist
    Config.create_directories()

    # Set seed for reproducibility
    set_seed(Config.SEED)

    print("==== Starting Training Pipeline ====")
    # 2. Run Training
    # This handles data loading, model init, training loop, and saving best_model.pth
    run_training()

    print("\n==== Starting Validation & Failure Analysis ====")
    # 3. Load Best Model
    device = Config.DEVICE
    model = DeepStabilizedBiGRU().to(device)
    # Load the best model saved during training
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    model.eval()

    # 4. Validation Inference
    # We need the validation loader again.
    _, val_loader, test_loader = get_dataloaders(debug=Config.DEBUG)

    all_preds = []
    all_targets = []
    all_ids = []

    # Inference loop (No gradients for speed/memory)
    with torch.no_grad():
        for batch in val_loader:
            features = batch["features"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            pair_mask = batch["pair_mask"].to(device)
            targets = batch["targets"]  # CPU
            ids = batch["ids"]

            outputs = model(features, pair_indices, pair_mask)

            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.numpy())
            all_ids.extend(ids)

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # 5. Compute Metric
    # compute_mcrmse handles slicing to SEQ_SCORED (68) and filtering columns
    val_metric = compute_mcrmse(all_preds, all_targets)
    print(f"Final Validation Metric: {val_metric}")

    # 6. Failure Analysis
    # Load metadata to get features
    val_df = pd.read_parquet(Config.VAL_METADATA_PATH)

    # Ensure alignment: Map errors to IDs
    # Calculate error per sample to correlate with metadata
    # Slicing to scored length
    preds_sliced = all_preds[:, : Config.SEQ_SCORED, :]
    targets_sliced = all_targets[:, : Config.SEQ_SCORED, :]

    # Filter columns: reactivity(0), deg_Mg_pH10(1), deg_Mg_50C(3)
    scored_indices = [0, 1, 3]
    p_filt = preds_sliced[:, :, scored_indices]
    t_filt = targets_sliced[:, :, scored_indices]

    # MSE per sample: Mean over Sequence(1) and Channels(2)
    # Shape: (N,)
    mse_per_sample = np.mean((p_filt - t_filt) ** 2, axis=(1, 2))
    rmse_per_sample = np.sqrt(mse_per_sample)

    # Map RMSE to ID in dataframe
    error_map = {id_: err for id_, err in zip(all_ids, rmse_per_sample)}
    val_df["rmse_error"] = val_df["id"].map(error_map)

    # Feature Engineering for Analysis
    val_df["len"] = val_df["sequence"].apply(len)
    val_df["pct_G"] = val_df["sequence"].apply(lambda s: s.count("G") / len(s))
    val_df["pct_A"] = val_df["sequence"].apply(lambda s: s.count("A") / len(s))
    val_df["pct_U"] = val_df["sequence"].apply(lambda s: s.count("U") / len(s))
    val_df["pct_C"] = val_df["sequence"].apply(lambda s: s.count("C") / len(s))

    analysis_cols = ["signal_to_noise", "SN_filter", "pct_G", "pct_A", "pct_U", "pct_C"]

    print("\nFailure Analysis (Correlation with Error):")
    for col in analysis_cols:
        if col in val_df.columns:
            # Drop NaNs if any (e.g. if error mapping failed for some reason)
            subset = val_df[[col, "rmse_error"]].dropna()
            if len(subset) > 1:
                corr, _ = pearsonr(subset[col], subset["rmse_error"])
                print(f"  {col}: {corr:.4f}")

    # 7. Submission
    THRESHOLD = 0.5884495377540588

    if val_metric < THRESHOLD:
        print(
            f"\nMetric ({val_metric}) < Threshold ({THRESHOLD}). Generating Submission..."
        )

        test_preds = []
        test_ids = []

        with torch.no_grad():
            for batch in test_loader:
                features = batch["features"].to(device)
                pair_indices = batch["pair_indices"].to(device)
                pair_mask = batch["pair_mask"].to(device)
                ids = batch["ids"]

                outputs = model(features, pair_indices, pair_mask)

                test_preds.append(outputs.cpu().numpy())
                test_ids.extend(ids)

        test_preds = np.concatenate(test_preds, axis=0)  # (N, 107, 5)

        # Prepare submission dataframe
        # Columns: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
        # The model outputs 5 targets in the order defined in library/data.py
        cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

        submission_rows = []

        # Flatten predictions: (N_samples * Seq_Len) rows
        for i, sample_id in enumerate(test_ids):
            sample_pred = test_preds[i]  # (107, 5)
            for seqpos in range(Config.SEQ_LEN):
                row_id = f"{sample_id}_{seqpos}"
                row_vals = sample_pred[seqpos]

                row_dict = {"id_seqpos": row_id}
                for j, col_name in enumerate(cols):
                    row_dict[col_name] = float(row_vals[j])

                submission_rows.append(row_dict)

        submission_df = pd.DataFrame(submission_rows)
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
        print(f"Submission shape: {submission_df.shape}")

    else:
        print(
            f"\nMetric ({val_metric}) >= Threshold ({THRESHOLD}). Skipping Submission."
        )


if __name__ == "__main__":
    main()
