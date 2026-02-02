import os
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

# Import from provided library files
from library.config import Config
from library.trainer import Trainer, set_seed
from library.dataset import RNADataset
from library.model import LatentSpatialBiGRU


def main():
    # 1. Setup and Configuration
    # ----------------------------------------------------------------
    # Initialize Config
    config = Config()

    # Modify config for a fast baseline execution
    config.epochs = 20  # Reduced from 50 to ensure quick runtime
    config.batch_size = 32  # Adjusted for safety
    config.num_workers = 2

    # Ensure submission directory exists
    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)

    # Set seeds
    set_seed(config.seed)

    print("Configuration configured for fast baseline.")
    print(f"Device: {config.device}")

    # 2. Training
    # ----------------------------------------------------------------
    # Initialize Trainer
    trainer = Trainer(config)

    # Run training
    print("\nStarting Training...")
    trainer.fit(load_cached_data=True)

    # 3. Validation & Metric Calculation
    # ----------------------------------------------------------------
    print("\nStarting Validation...")

    # Load best model
    best_model_path = config.model_save_path
    if not os.path.exists(best_model_path):
        print("Error: Best model file not found.")
        return

    model = LatentSpatialBiGRU(config).to(config.device)
    model.load_state_dict(torch.load(best_model_path, map_location=config.device))
    model.eval()

    # Load Validation Data
    val_dataset = RNADataset(split="val", load_cached_data=True)
    val_loader = DataLoader(
        val_dataset, batch_size=64, shuffle=False, num_workers=config.num_workers
    )

    # Inference on Validation
    all_preds = []
    all_targets = []
    all_ids = []

    # Scored columns indices
    scored_indices = [
        i for i, col in enumerate(config.target_cols) if col in config.scored_cols
    ]

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["input"].to(config.device)
            pair_indices = batch["pair_index"].to(config.device)
            targets = batch["target"].to(config.device)
            ids = batch["id"]

            outputs = model(inputs, pair_indices)

            # Slice outputs to match target length (seq_scored=68)
            seq_len_target = targets.shape[1]
            if outputs.shape[1] > seq_len_target:
                outputs = outputs[:, :seq_len_target, :]

            all_preds.append(outputs.cpu())
            all_targets.append(targets.cpu())
            all_ids.extend(ids)

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Calculate Metric (MCRMSE on scored columns)
    scored_preds = all_preds[:, :, scored_indices]
    scored_targets = all_targets[:, :, scored_indices]

    mse_scored = torch.mean((scored_preds - scored_targets) ** 2, dim=(0, 1))
    rmse_scored = torch.sqrt(mse_scored)
    final_metric = torch.mean(rmse_scored).item()

    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    # ----------------------------------------------------------------
    print("\nPerforming Failure Analysis...")

    # Calculate MCRMSE per sample
    # Shape: (N, 68, 3) -> Mean over (68, 3) -> Scalar per sample
    # Note: MCRMSE is mean of RMSEs. Per sample, we can just take sqrt(MSE)
    # Let's compute the error magnitude per sample as the mean RMSE across the 3 scored columns

    # (N, 68, 3)
    squared_diff = (scored_preds - scored_targets) ** 2
    # MSE per sample per column: (N, 3)
    mse_per_sample_col = torch.mean(squared_diff, dim=1)
    # RMSE per sample per column: (N, 3)
    rmse_per_sample_col = torch.sqrt(mse_per_sample_col)
    # Mean RMSE per sample: (N,)
    error_per_sample = torch.mean(rmse_per_sample_col, dim=1).numpy()

    # Load Metadata to get features
    val_meta_path = config.val_path
    val_df = pd.read_parquet(val_meta_path)

    # Ensure alignment (dataset loader preserves order, but let's be safe)
    # The dataset class loads ids from the processed file.
    # We create a mapping from ID to error
    id_to_error = dict(zip(all_ids, error_per_sample))

    # Map errors to dataframe
    val_df["model_error"] = val_df["id"].map(id_to_error)

    # Features to analyze
    features_to_check = ["signal_to_noise", "SN_filter"]

    # Add nucleotide content features
    val_df["pct_A"] = val_df["sequence"].apply(lambda s: s.count("A") / len(s))
    val_df["pct_U"] = val_df["sequence"].apply(lambda s: s.count("U") / len(s))
    val_df["pct_G"] = val_df["sequence"].apply(lambda s: s.count("G") / len(s))
    val_df["pct_C"] = val_df["sequence"].apply(lambda s: s.count("C") / len(s))

    features_to_check.extend(["pct_A", "pct_U", "pct_G", "pct_C"])

    print("Correlation between Model Error and Features:")
    for feat in features_to_check:
        if feat in val_df.columns:
            # Drop NaNs just in case
            valid_data = val_df[[feat, "model_error"]].dropna()
            if len(valid_data) > 1:
                corr, _ = pearsonr(valid_data[feat], valid_data["model_error"])
                print(f"  {feat}: {corr:.4f}")

    # 5. Submission Generation
    # ----------------------------------------------------------------
    THRESHOLD = 0.7247761841173526

    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric}) is better than threshold ({THRESHOLD}). Generating submission..."
        )

        # Load Test Data
        test_dataset = RNADataset(split="test", load_cached_data=True)
        test_loader = DataLoader(
            test_dataset, batch_size=64, shuffle=False, num_workers=config.num_workers
        )

        test_preds = []
        test_ids = []

        with torch.no_grad():
            for batch in test_loader:
                inputs = batch["input"].to(config.device)
                pair_indices = batch["pair_index"].to(config.device)
                ids = batch["id"]

                # Forward pass
                # Output shape: (Batch, 107, 5)
                outputs = model(inputs, pair_indices)

                test_preds.append(outputs.cpu().numpy())
                test_ids.extend(ids)

        # Concatenate predictions: (N_test, 107, 5)
        test_preds = np.concatenate(test_preds, axis=0)

        # Prepare Submission DataFrame
        # Format: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
        submission_rows = []
        target_cols = (
            config.target_cols
        )  # ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

        for i, sample_id in enumerate(test_ids):
            # Get predictions for this sample: (107, 5)
            sample_pred = test_preds[i]

            for seqpos in range(sample_pred.shape[0]):
                row_id = f"{sample_id}_{seqpos}"
                row_values = sample_pred[seqpos]

                row_dict = {"id_seqpos": row_id}
                for col_idx, col_name in enumerate(target_cols):
                    row_dict[col_name] = float(row_values[col_idx])

                submission_rows.append(row_dict)

        submission_df = pd.DataFrame(submission_rows)

        # Save
        save_path = os.path.join(submission_dir, "submission.csv")
        submission_df.to_csv(save_path, index=False)
        print(f"Submission saved to {save_path}. Shape: {submission_df.shape}")

    else:
        print(
            f"\nMetric ({final_metric}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
