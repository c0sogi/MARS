import sys
import os
import pandas as pd
import numpy as np
import torch
import scipy.stats as stats

# Ensure library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.train_eval import Trainer
from library.dataset import get_dataloader
from library.model_components import HSDARNModel


def main():
    # 1. Initialization and Reproducibility
    Config.set_seed(Config.SEED)

    # 2. Training
    # We use the provided Trainer class which handles the training loop,
    # validation logging, and model checkpointing.
    trainer = Trainer()
    trainer.fit()

    # 3. Validation Reporting
    # The trainer stores the best validation score (MCRMSE)
    final_metric = trainer.best_score
    # Print with full precision as required
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    print("\n==== Failure Analysis ====")

    # We need to compute predictions on the validation set using the best model
    # to analyze where the model fails.

    # Load the best model
    model = HSDARNModel().to(Config.DEVICE)
    if os.path.exists(trainer.best_model_path):
        model.load_state_dict(
            torch.load(trainer.best_model_path, map_location=Config.DEVICE)
        )
    else:
        print("Warning: Best model not found. Using initialized model for analysis.")

    model.eval()

    # Get validation loader
    val_loader = get_dataloader("val", batch_size=Config.BATCH_SIZE, shuffle=False)

    all_preds = []
    all_targets = []
    all_ids = []

    # Inference loop on validation set
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["inputs"].to(Config.DEVICE)
            partner_map = batch["partner_map"].to(Config.DEVICE)
            targets = batch["targets"].to(Config.DEVICE)

            # Forward pass - use the refined prediction (y2)
            _, y2 = model(inputs, partner_map)

            all_preds.append(y2.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

            if "id" in batch:
                all_ids.extend(batch["id"])

    # Concatenate results
    if len(all_preds) > 0:
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        # Calculate RMSE per sample for the scored columns
        # Scored columns: reactivity, deg_Mg_pH10, deg_Mg_50C
        scored_indices = [
            i for i, col in enumerate(Config.TARGET_COLS) if col in Config.SCORED_COLS
        ]

        # Slice to scored length (68) and scored columns
        # Shape: (N, 68, 3)
        preds_scored = all_preds[:, : Config.PRED_LEN, scored_indices]
        targets_scored = all_targets[:, : Config.PRED_LEN, scored_indices]

        # Compute MSE per sample (average over length and channels)
        mse_per_sample = np.mean((preds_scored - targets_scored) ** 2, axis=(1, 2))
        rmse_per_sample = np.sqrt(mse_per_sample)

        # Load validation metadata to get features like signal_to_noise
        val_df = pd.read_csv(Config.VAL_CSV)

        # Create analysis dataframe
        # We assume the order in val_loader matches val_df because shuffle=False
        analysis_df = val_df.copy()

        # If IDs were collected, we can merge safely to ensure alignment
        if len(all_ids) == len(analysis_df):
            error_mapping = pd.DataFrame({"id": all_ids, "rmse_error": rmse_per_sample})
            analysis_df = analysis_df.merge(error_mapping, on="id")
        else:
            # Fallback to direct assignment if lengths match
            if len(rmse_per_sample) == len(analysis_df):
                analysis_df["rmse_error"] = rmse_per_sample
            else:
                print("Error: Mismatch in validation set size for analysis.")
                analysis_df = None

        if analysis_df is not None:
            # Correlation with Signal to Noise
            if "signal_to_noise" in analysis_df.columns:
                corr_sn, _ = stats.pearsonr(
                    analysis_df["signal_to_noise"], analysis_df["rmse_error"]
                )
                print(
                    f"Correlation between Error (RMSE) and signal_to_noise: {corr_sn}"
                )

            # Correlation with Mean Reactivity (if available in metadata)
            if "mean_reactivity" in analysis_df.columns:
                corr_mr, _ = stats.pearsonr(
                    analysis_df["mean_reactivity"], analysis_df["rmse_error"]
                )
                print(
                    f"Correlation between Error (RMSE) and mean_reactivity: {corr_mr}"
                )

    else:
        print("No validation predictions generated for analysis.")

    # 5. Submission
    # Generate submission only if metric meets the threshold
    THRESHOLD = 0.47142532743789534

    if final_metric < THRESHOLD:
        print(
            f"\nValidation Metric ({final_metric}) is better than threshold ({THRESHOLD})."
        )
        print("Generating submission file...")
        trainer.predict()
    else:
        print(
            f"\nValidation Metric ({final_metric}) did not meet threshold ({THRESHOLD})."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
