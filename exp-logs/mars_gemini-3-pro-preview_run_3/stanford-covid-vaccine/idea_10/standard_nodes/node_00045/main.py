import os
import sys
import torch
import pandas as pd
import numpy as np
import library.config
from library.utils import seed_everything, mcrmse_loss
from library.data import get_dataloaders
from library.model import RNARegressor
from library.engine import run_training


def main():
    # 1. Setup
    seed_everything(library.config.Config.SEED)
    device = torch.device(library.config.Config.DEVICE)

    # Ensure submission directory exists as per prompt requirement
    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)
    submission_file_path = os.path.join(submission_dir, "submission.csv")

    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        train_batch_size=library.config.Config.BATCH_SIZE,
        val_batch_size=library.config.Config.BATCH_SIZE,
        load_cached_data=True,
    )

    # 2. Training
    # run_training handles initialization, training loop, early stopping, and saving best model
    print("Starting Training...")
    best_mcrmse_from_training = run_training(train_loader, val_loader)

    # 3. Validation & Failure Analysis
    print("\nRunning Validation and Failure Analysis...")

    # Load Best Model
    model = RNARegressor()
    model.load_state_dict(
        torch.load(library.config.Config.MODEL_SAVE_PATH, map_location=device)
    )
    model.to(device)
    model.eval()

    val_preds = []
    val_targets = []
    val_inputs = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Inference
            reg_out = model(inputs)

            # Store data for analysis (move to CPU to save GPU memory)
            val_preds.append(reg_out.cpu())
            val_targets.append(targets.cpu())
            val_inputs.append(inputs.cpu())

    # Concatenate all batches
    val_preds = torch.cat(val_preds, dim=0)  # (N, 107, 5)
    val_targets = torch.cat(val_targets, dim=0)  # (N, 107, 5)
    val_inputs = torch.cat(val_inputs, dim=0)  # (N, 107, 14)

    # Slice to scored positions
    val_preds_scored = val_preds[:, : library.config.Config.SEQ_SCORED, :]
    val_targets_scored = val_targets[:, : library.config.Config.SEQ_SCORED, :]

    # Filter for scored columns ONLY (reactivity, deg_Mg_pH10, deg_Mg_50C)
    val_preds_scored = val_preds_scored[:, :, library.config.Config.SCORED_COLS_INDICES]
    val_targets_scored = val_targets_scored[
        :, :, library.config.Config.SCORED_COLS_INDICES
    ]

    # Compute Final Metric
    final_metric = mcrmse_loss(val_preds_scored, val_targets_scored).item()
    print(f"Final Validation Metric: {final_metric}")

    # --- Failure Analysis ---
    # Calculate error per sample (mean across scored positions and targets)
    # Squared error: (N, 68, 5)
    squared_errors = (val_preds_scored - val_targets_scored) ** 2
    # Mean over positions and targets -> (N,)
    sample_mse = torch.mean(squared_errors, dim=(1, 2))
    sample_rmse = torch.sqrt(sample_mse).numpy()

    # Extract features from inputs (N, 107, 14)
    # Channels: 0-3 (A,G,C,U), 4-6 (., (, )), 7-13 (Loops)
    # We calculate the percentage of each feature in the sequence
    feature_counts = torch.mean(val_inputs, dim=1).numpy()  # (N, 14)

    feature_names = [
        "pct_A",
        "pct_G",
        "pct_C",
        "pct_U",
        "pct_unpaired",
        "pct_open_paren",
        "pct_close_paren",
        "pct_S",
        "pct_M",
        "pct_I",
        "pct_B",
        "pct_H",
        "pct_E",
        "pct_X",
    ]

    print("\nFailure Analysis (Correlation with Error Magnitude):")
    correlations = {}
    for i, name in enumerate(feature_names):
        corr = np.corrcoef(sample_rmse, feature_counts[:, i])[0, 1]
        correlations[name] = corr

    # Sort and print
    sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    for name, corr in sorted_corrs[:5]:
        print(f"  {name}: {corr:.4f}")

    # 4. Submission
    THRESHOLD = 0.7247761841173526

    if final_metric < THRESHOLD:
        print(
            f"\nMetric meets threshold ({final_metric} < {THRESHOLD}). Generating submission..."
        )

        # Load Test IDs from parquet
        test_df = pd.read_parquet(library.config.Config.TEST_DATA_PATH)
        test_ids = test_df["id"].values

        test_preds = []

        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(device)
                reg_out = model(inputs)
                test_preds.append(reg_out.cpu())

        test_preds = torch.cat(test_preds, dim=0)  # (N_test, 107, 5)

        # Prepare submission data
        # We need to flatten: id_seqpos, and the 5 targets
        submission_rows = []

        target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

        # Convert to numpy for faster indexing
        test_preds_np = test_preds.numpy()

        for i, sample_id in enumerate(test_ids):
            # Get predictions for this sample
            # We iterate over full length 107 as per submission format
            sample_preds = test_preds_np[i]  # (107, 5)

            for seqpos in range(library.config.Config.SEQ_LENGTH):
                row_id = f"{sample_id}_{seqpos}"
                preds = sample_preds[seqpos]

                row_dict = {"id_seqpos": row_id}
                for t_idx, col in enumerate(target_cols):
                    row_dict[col] = preds[t_idx]

                submission_rows.append(row_dict)

        # Create DataFrame
        submission_df = pd.DataFrame(submission_rows)

        # Save
        submission_df.to_csv(submission_file_path, index=False)
        print(f"Submission saved to {submission_file_path}")

    else:
        print(
            f"\nMetric {final_metric} did not meet threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
