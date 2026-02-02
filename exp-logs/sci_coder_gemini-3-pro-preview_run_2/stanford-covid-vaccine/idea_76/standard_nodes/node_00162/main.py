import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

# Import from provided library
from library.config import Config
from library.utils import set_seed, get_global_rmse
from library.dataset import RNADataset
from library.model import AHCHIDN
from library.trainer import Trainer


def main():
    # 1. Setup and Overrides for Fast Baseline
    print("Initializing Fast Baseline Run...")

    # Override Config for speed
    Config.EPOCHS = 10
    Config.SUBSET_SIZE = 1000  # Limit training data for speed

    # Ensure submission directory exists
    os.makedirs("./submission", exist_ok=True)

    # Set seeds for reproducibility
    set_seed(Config.SEED)

    # 2. Training
    print("Starting Training...")
    trainer = Trainer()
    trainer.fit()

    # 3. Validation & Metric Calculation
    print("\nRunning Final Validation...")

    # Load best model
    device = torch.device(Config.DEVICE)
    model = AHCHIDN().to(device)
    model.load_state_dict(torch.load(trainer.best_model_path, map_location=device))
    model.eval()

    # Get Validation Loader
    val_dataset = RNADataset(mode="val", load_cached_data=True)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    all_preds = []
    all_targets = []
    all_ids = []

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["inputs"].to(device)
            partner_indices = batch["partner_indices"].to(device)
            targets = batch["targets"].to(device)
            ids = batch["id"]

            # Inference (Pass 1 -> Pass 2)
            preds_1 = model(inputs, partner_indices, prev_preds=None)
            preds_2 = model(inputs, partner_indices, prev_preds=preds_1)

            all_preds.append(preds_2.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_ids.extend(ids)

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Compute Metric
    final_metric = get_global_rmse(all_preds, all_targets)
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    print("\nPerforming Failure Analysis...")

    # Calculate RMSE per sample (averaged over scored columns and positions)
    # Filter to scored columns/positions first
    seq_scored = Config.SEQ_SCORED
    scored_indices = Config.SCORED_COLS_INDICES

    preds_scored = all_preds[:, :seq_scored, :][:, :, scored_indices]
    targets_scored = all_targets[:, :seq_scored, :][:, :, scored_indices]

    # Mean Squared Error per sample
    mse_per_sample = np.mean((preds_scored - targets_scored) ** 2, axis=(1, 2))
    rmse_per_sample = np.sqrt(mse_per_sample)

    # Load Metadata to get Signal to Noise
    val_meta_path = os.path.join(Config.METADATA_DIR, "val.csv")
    val_df = pd.read_csv(val_meta_path)

    # Map RMSE to Metadata using IDs
    # Create a dict for fast lookup
    sn_map = dict(zip(val_df["id"], val_df["signal_to_noise"]))

    # Align SN with the order of predictions
    sn_values = np.array([sn_map.get(sid, 0.0) for sid in all_ids])

    # Calculate Correlation
    # Handle potential NaNs if any (though data should be clean)
    valid_mask = ~np.isnan(sn_values)
    if np.sum(valid_mask) > 1:
        corr, _ = pearsonr(rmse_per_sample[valid_mask], sn_values[valid_mask])
        print(f"Correlation between Error (RMSE) and Signal_to_Noise: {corr:.4f}")
    else:
        print("Insufficient data for correlation analysis.")

    # 5. Submission Generation
    threshold = 0.47142532743789534
    if final_metric < threshold:
        print(
            f"\nMetric ({final_metric}) is better than threshold ({threshold}). Generating Submission..."
        )

        test_dataset = RNADataset(mode="test", load_cached_data=True)
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        )

        test_preds = []
        test_ids = []

        with torch.no_grad():
            for batch in test_loader:
                inputs = batch["inputs"].to(device)
                partner_indices = batch["partner_indices"].to(device)
                ids = batch["id"]

                # Inference
                preds_1 = model(inputs, partner_indices, prev_preds=None)
                preds_2 = model(inputs, partner_indices, prev_preds=preds_1)

                test_preds.append(preds_2.cpu().numpy())
                test_ids.extend(ids)

        test_preds = np.concatenate(test_preds, axis=0)  # (N, 107, 5)

        # Format for Submission
        # We need to flatten: id_seqpos, val1, val2, val3, val4, val5
        submission_rows = []
        target_cols = (
            Config.TARGET_COLS
        )  # ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

        for i, sample_id in enumerate(test_ids):
            sample_pred = test_preds[i]  # (107, 5)
            for seqpos in range(Config.SEQ_LENGTH):
                row_id = f"{sample_id}_{seqpos}"
                row_values = sample_pred[seqpos].tolist()

                # Create dict
                row_dict = {"id_seqpos": row_id}
                for col_name, val in zip(target_cols, row_values):
                    row_dict[col_name] = val
                submission_rows.append(row_dict)

        submission_df = pd.DataFrame(submission_rows)

        # Reorder columns to match sample submission requirement
        # id_seqpos,reactivity,deg_Mg_pH10,deg_pH10,deg_Mg_50C,deg_50C
        cols_order = ["id_seqpos"] + target_cols
        submission_df = submission_df[cols_order]

        save_path = "./submission/submission.csv"
        submission_df.to_csv(save_path, index=False)
        print(f"Submission saved to {save_path}")

    else:
        print(
            f"\nMetric ({final_metric}) did not meet threshold ({threshold}). Skipping submission."
        )


if __name__ == "__main__":
    main()
