import os
import sys
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, calculate_mcrmse
from library.data import get_dataloaders, get_test_dataloader
from library.model import RISRBiGRU
from library.train import Trainer


def main():
    # =========================================================================
    # 1. Configuration & Setup
    # =========================================================================
    # Initialize configuration
    config = Config()

    # Adjust for fast baseline execution
    # We limit epochs to ensure the run completes quickly within the time limit
    # while still allowing for convergence on the small dataset.
    config.epochs = 10
    config.debug = False  # Use full dataset, but limited epochs

    # Ensure reproducibility
    seed_everything(config.seed)

    print("Configuration:")
    print(f"  Device: {config.device}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Batch Size: {config.batch_size}")

    # =========================================================================
    # 2. Training
    # =========================================================================
    print("\n[Step 1/4] Training Model...")

    # Get DataLoaders
    # load_cached_data=True allows using preprocessed .npz files from ./working
    train_loader, val_loader = get_dataloaders(config, load_cached_data=True)

    # Initialize Trainer
    trainer = Trainer(config, train_loader, val_loader)

    # Run Training
    trainer.fit()

    # =========================================================================
    # 3. Validation & Metric Calculation
    # =========================================================================
    print("\n[Step 2/4] Validating Best Model...")

    # Load the best model
    model = RISRBiGRU(config).to(config.device)
    model.load_state_dict(
        torch.load(config.best_model_path, map_location=config.device)
    )
    model.eval()

    # Run Inference on Validation Set
    val_preds = []
    val_targets = []

    # We need to ensure we process the validation set in the same order as the metadata
    # The val_loader from get_dataloaders(shuffle=False) ensures this.
    with torch.no_grad():
        for inputs, adjacency, targets in val_loader:
            inputs = inputs.to(config.device)
            adjacency = adjacency.to(config.device)

            # Forward pass
            preds = model(inputs, adjacency)

            # Slice predictions to match target length (68)
            # Targets in val_loader are already sliced to config.pred_len (68)
            if preds.shape[1] > targets.shape[1]:
                preds = preds[:, : targets.shape[1], :]

            val_preds.append(preds.cpu().numpy())
            val_targets.append(targets.numpy())

    val_preds = np.concatenate(val_preds, axis=0)
    val_targets = np.concatenate(val_targets, axis=0)

    # Calculate Metric on Scored Columns
    # indices: reactivity(0), deg_Mg_pH10(1), deg_Mg_50C(3)
    scored_indices = [0, 1, 3]

    final_metric = calculate_mcrmse(
        val_preds, val_targets, scored_indices=scored_indices
    )

    # PRINT REQUIRED METRIC
    print(f"Final Validation Metric: {final_metric}")

    # =========================================================================
    # 4. Failure Analysis
    # =========================================================================
    print("\n[Step 3/4] Performing Failure Analysis...")

    # Load Validation Metadata to correlate errors with features
    val_meta_path = config.val_metadata_path
    val_df = pd.read_parquet(val_meta_path)

    # Calculate Sample-wise Error (MCRMSE per sample)
    # val_preds: (N, 68, 5), val_targets: (N, 68, 5)
    # We focus on the scored columns for error analysis
    diff = val_preds[:, :, scored_indices] - val_targets[:, :, scored_indices]
    mse_per_sample = np.mean(diff**2, axis=(1, 2))  # Mean over length and channels
    rmse_per_sample = np.sqrt(mse_per_sample)

    val_df["error_rmse"] = rmse_per_sample

    # Features to correlate
    analysis_cols = ["signal_to_noise", "seq_length", "SN_filter"]

    # Add sequence content features
    val_df["pct_A"] = val_df["sequence"].apply(lambda s: s.count("A") / len(s))
    val_df["pct_G"] = val_df["sequence"].apply(lambda s: s.count("G") / len(s))
    val_df["pct_U"] = val_df["sequence"].apply(lambda s: s.count("U") / len(s))
    val_df["pct_C"] = val_df["sequence"].apply(lambda s: s.count("C") / len(s))

    analysis_cols.extend(["pct_A", "pct_G", "pct_U", "pct_C"])

    print("Correlation between Error (RMSE) and Features:")
    correlations = val_df[analysis_cols].corrwith(val_df["error_rmse"])
    print(correlations)

    # =========================================================================
    # 5. Submission Generation
    # =========================================================================
    threshold = 0.5978901386

    if final_metric < threshold:
        print(
            f"\n[Step 4/4] Metric ({final_metric}) < Threshold ({threshold}). Generating Submission..."
        )

        # Get Test DataLoader
        test_loader = get_test_dataloader(config, load_cached_data=True)

        test_preds = []
        test_ids = []

        with torch.no_grad():
            for inputs, adjacency, sample_ids in test_loader:
                inputs = inputs.to(config.device)
                adjacency = adjacency.to(config.device)

                # Forward pass
                # Model outputs (B, 107, 5)
                preds = model(inputs, adjacency)

                test_preds.append(preds.cpu().numpy())
                test_ids.extend(sample_ids)

        # Concatenate
        # shape: (N_test, 107, 5)
        test_preds_arr = np.concatenate(test_preds, axis=0)

        # Prepare Submission DataFrame
        # Format: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
        submission_rows = []
        target_cols = (
            config.target_cols
        )  # ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

        seq_len = config.seq_len  # 107

        for i, sample_id in enumerate(test_ids):
            for pos in range(seq_len):
                row_id = f"{sample_id}_{pos}"
                preds_values = test_preds_arr[i, pos, :]

                row_dict = {"id_seqpos": row_id}
                for col_idx, col_name in enumerate(target_cols):
                    row_dict[col_name] = preds_values[col_idx]

                submission_rows.append(row_dict)

        submission_df = pd.DataFrame(submission_rows)

        # Save
        submission_path = config.submission_path
        submission_df.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")
        print(f"Submission shape: {submission_df.shape}")

    else:
        print(
            f"\n[Step 4/4] Metric ({final_metric}) >= Threshold ({threshold}). Skipping Submission."
        )


if __name__ == "__main__":
    main()
