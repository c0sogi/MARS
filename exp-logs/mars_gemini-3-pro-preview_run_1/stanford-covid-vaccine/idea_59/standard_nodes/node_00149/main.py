import os
import sys
import numpy as np
import pandas as pd
import torch

# Ensure library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed, mcrmse_metric
from library.train import Trainer
from library.dataset import get_dataloaders


def calculate_correlation(x, y):
    """Calculates Pearson correlation coefficient between two arrays."""
    if len(x) != len(y) or len(x) == 0:
        return 0.0
    # Use numpy for correlation to ensure compatibility
    matrix = np.corrcoef(x, y)
    if matrix.size > 1:
        return matrix[0, 1]
    return 0.0


def main():
    # 1. Setup and Configuration
    # Set seed for reproducibility
    set_seed(42)

    # Modify Config for a fast baseline execution
    # Using 20 epochs as per optimization plan (Cite 00131)
    Config.EPOCHS = 20

    print(f"Configuration: EPOCHS={Config.EPOCHS}, BATCH_SIZE={Config.BATCH_SIZE}")

    # 2. Training
    # Initialize Trainer and fit the model
    trainer = Trainer(debug=False)
    trainer.fit(load_cached_data=True)

    # 3. Validation & Metric Calculation
    print("\nPerforming Final Validation...")

    # Load the best model weights
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    if not os.path.exists(best_model_path):
        print("Error: Best model not found.")
        return

    model = trainer.model
    model.load_state_dict(torch.load(best_model_path, map_location=trainer.device))
    model.eval()

    # Get dataloaders (using cached data)
    _, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE, load_cached_data=True
    )

    # Collect predictions and targets for the entire validation set
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            seq = batch["seq"].to(trainer.device)
            loop = batch["loop"].to(trainer.device)
            dist = batch["dist"].to(trainer.device)
            targets = batch["targets"].to(trainer.device)

            preds = model(seq, loop, dist)

            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Calculate Final Validation Metric (MCRMSE)
    # The metric function handles slicing to the scored length (Config.PRED_LEN)
    final_metric = mcrmse_metric(all_targets, all_preds, pred_len=Config.PRED_LEN)
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    print("\nFailure Analysis:")

    # Load validation metadata to retrieve features
    val_parquet_path = os.path.join(Config.METADATA_DIR, "val.parquet")
    if os.path.exists(val_parquet_path):
        val_df = pd.read_parquet(val_parquet_path)

        # Calculate per-sample error
        # We use the mean of the RMSEs of the 3 scored columns for each sample
        # Slice to scored length
        p_np = all_preds.numpy()[:, : Config.PRED_LEN, :]
        t_np = all_targets.numpy()[:, : Config.PRED_LEN, :]

        # Squared differences: (N, 68, 3)
        sq_diff = (p_np - t_np) ** 2
        # MSE per column per sample: (N, 3)
        mse_per_col = np.mean(sq_diff, axis=1)
        # RMSE per column per sample: (N, 3)
        rmse_per_col = np.sqrt(mse_per_col)
        # Average RMSE across the 3 columns for each sample: (N,)
        sample_errors = np.mean(rmse_per_col, axis=1)

        # Add error to dataframe
        # Ensure alignment (dataloaders preserve order if shuffle=False, which is true for val_loader)
        if len(val_df) == len(sample_errors):
            val_df["error_metric"] = sample_errors

            # Extract sequence features
            val_df["len_A"] = val_df["sequence"].apply(lambda x: x.count("A"))
            val_df["len_G"] = val_df["sequence"].apply(lambda x: x.count("G"))
            val_df["len_C"] = val_df["sequence"].apply(lambda x: x.count("C"))
            val_df["len_U"] = val_df["sequence"].apply(lambda x: x.count("U"))

            # Features to analyze
            features_to_check = [
                "signal_to_noise",
                "SN_filter",
                "len_A",
                "len_G",
                "len_C",
                "len_U",
            ]

            for feat in features_to_check:
                if feat in val_df.columns:
                    # Filter out NaNs if any
                    subset = val_df[[feat, "error_metric"]].dropna()
                    if len(subset) > 0:
                        corr = calculate_correlation(
                            subset[feat].values, subset["error_metric"].values
                        )
                        print(f"Correlation between Error and {feat}: {corr:.4f}")
        else:
            print(
                "Warning: Validation DataFrame length does not match predictions length. Skipping detailed correlation analysis."
            )
    else:
        print("Warning: Validation metadata file not found. Skipping failure analysis.")

    # 5. Submission Generation
    # Threshold defined in task
    THRESHOLD = 0.6176461577

    if final_metric < THRESHOLD:
        print(
            f"\nMetric {final_metric} is better than threshold {THRESHOLD}. Generating submission..."
        )
        trainer.generate_submission(test_loader)
    else:
        print(
            f"\nMetric {final_metric} did not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
