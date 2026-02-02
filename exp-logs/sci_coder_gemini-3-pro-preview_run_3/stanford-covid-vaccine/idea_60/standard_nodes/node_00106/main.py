import os
import sys
import torch
import numpy as np
import pandas as pd

# Ensure library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed, scored_mcrmse
from library.data import get_dataloaders
from library.model import HCSDBR_BiGRU
from library.train import run_training


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = Config.DEVICE
    print(f"Running on device: {device}")

    # 2. Training
    # Run training for a limited number of epochs to create a fast baseline
    # 20 epochs is sufficient for this dataset size (~1.7k samples) to show convergence
    print("Starting training...")
    run_training(load_cached_data=True, debug=False, epochs=20)

    # 3. Validation Inference
    print("Loading best model for validation...")
    model = HCSDBR_BiGRU()
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.to(device)
    model.eval()

    # Get dataloaders (uses cache)
    _, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=True,
        debug=False,
    )

    all_preds = []
    all_targets = []
    all_ids = []

    print("Running validation inference...")
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["inputs"].to(device)
            adj = batch["adjacency"].to(device)
            mask = batch["mask"].to(device)
            targets = batch["targets"]  # Keep on CPU
            ids = batch["ids"]

            outputs = model(inputs, adj, mask)

            all_preds.append(outputs.cpu())
            all_targets.append(targets)
            all_ids.extend(ids)

    # Concatenate
    y_pred = torch.cat(all_preds, dim=0)
    y_true = torch.cat(all_targets, dim=0)

    # Compute Final Metric
    val_metric = scored_mcrmse(y_true, y_pred)
    print(f"Final Validation Metric: {val_metric}")

    # 4. Failure Analysis
    print("Performing failure analysis...")
    # Calculate RMSE per sample (on scored parts only)
    seq_scored = Config.SEQ_SCORED
    scored_cols_indices = [Config.TARGET_COLS.index(c) for c in Config.SCORED_TARGETS]

    # Slice to scored region and columns
    y_pred_scored = y_pred[:, :seq_scored, scored_cols_indices]
    y_true_scored = y_true[:, :seq_scored, scored_cols_indices]

    # MSE per sample: Average over sequence (dim 1) and columns (dim 2)
    mse_per_sample = torch.mean((y_pred_scored - y_true_scored) ** 2, dim=(1, 2))
    rmse_per_sample = torch.sqrt(mse_per_sample).numpy()

    # Load metadata to correlate with features
    val_df = pd.read_parquet(Config.VAL_PATH)

    # Create analysis dataframe
    error_df = pd.DataFrame({"id": all_ids, "rmse": rmse_per_sample})
    analysis_df = val_df.merge(error_df, on="id")

    # Correlation with Signal to Noise
    if "signal_to_noise" in analysis_df.columns:
        corr_snr = analysis_df["rmse"].corr(analysis_df["signal_to_noise"])
        print(f"Correlation between Error (RMSE) and Signal-to-Noise: {corr_snr:.8f}")

    # Correlation with SN_filter
    if "SN_filter" in analysis_df.columns:
        corr_filter = analysis_df["rmse"].corr(analysis_df["SN_filter"])
        print(f"Correlation between Error (RMSE) and SN_filter: {corr_filter:.8f}")

    # 5. Submission
    THRESHOLD = 0.5884495377540588
    if val_metric < THRESHOLD:
        print(f"Metric {val_metric} < {THRESHOLD}. Generating submission...")

        test_preds = []
        test_ids = []

        with torch.no_grad():
            for batch in test_loader:
                inputs = batch["inputs"].to(device)
                adj = batch["adjacency"].to(device)
                mask = batch["mask"].to(device)
                ids = batch["ids"]

                outputs = model(inputs, adj, mask)
                test_preds.append(outputs.cpu().numpy())
                test_ids.extend(ids)

        # Concatenate predictions: (N_samples, 107, 5)
        test_preds_arr = np.concatenate(test_preds, axis=0)

        # Format submission
        submission_rows = []
        target_cols = (
            Config.TARGET_COLS
        )  # ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

        for i, sample_id in enumerate(test_ids):
            pred_sample = test_preds_arr[i]  # (107, 5)
            for seqpos in range(Config.SEQ_LEN):
                # Row ID format: id_{sample_id}_{seqpos}
                # But based on sample_submission, 'id' in json is like 'id_00073f8be'
                # and submission id_seqpos is 'id_00073f8be_0'
                row_id = f"{sample_id}_{seqpos}"

                row_data = {"id_seqpos": row_id}
                for t_idx, col_name in enumerate(target_cols):
                    row_data[col_name] = float(pred_sample[seqpos, t_idx])

                submission_rows.append(row_data)

        submission_df = pd.DataFrame(submission_rows)

        # Ensure submission directory exists
        os.makedirs("./submission", exist_ok=True)
        sub_path = "./submission/submission.csv"

        # Reorder columns to match sample submission just in case, though dict order is usually preserved
        cols_order = [
            "id_seqpos",
            "reactivity",
            "deg_Mg_pH10",
            "deg_pH10",
            "deg_Mg_50C",
            "deg_50C",
        ]
        submission_df = submission_df[cols_order]

        submission_df.to_csv(sub_path, index=False)
        print(f"Submission saved to {sub_path}")

    else:
        print(f"Metric {val_metric} >= {THRESHOLD}. Skipping submission.")


if __name__ == "__main__":
    main()
