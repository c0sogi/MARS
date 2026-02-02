import os
import torch
import pandas as pd
import numpy as np
import warnings

# Import from the provided library files
from library.config import Config
from library.train import run_training, generate_submission
from library.utils import seed_everything, mcrmse_loss, load_checkpoint
from library.data import get_dataloaders
from library.model import BondAwareModel


def main():
    # 1. Setup and Configuration
    warnings.filterwarnings("ignore")
    seed_everything(Config.seed)

    # Ensure the submission directory exists and update the config path
    os.makedirs("./submission", exist_ok=True)
    Config.submission_path = "./submission/submission.csv"

    # 2. Run Training
    # This executes the training loop defined in library/train.py
    # It saves the best model to Config.model_path based on validation MCRMSE
    print("--- Starting Training ---")
    run_training()

    # 3. Evaluation & Failure Analysis
    print("\n--- Starting Evaluation & Failure Analysis ---")

    # Load the best model for analysis
    device = torch.device(Config.device)
    model = BondAwareModel().to(device)
    epoch, loss = load_checkpoint(model, None, Config.model_path, device=device)
    print(f"Loaded best model from epoch {epoch} (Loss: {loss:.6f})")
    model.eval()

    # Get DataLoaders (using cached data if available)
    _, val_loader, _ = get_dataloaders(debug=Config.debug)

    # Containers for analysis
    all_preds = []
    all_targets = []
    all_ids = []

    # Inference loop on Validation Set
    with torch.no_grad():
        for batch in val_loader:
            # Move inputs to device
            seq = batch["seq"].to(device)
            loop = batch["loop"].to(device)
            dist = batch["dist"].to(device)
            targets = batch["targets"].to(device)
            batch_ids = batch["id"]

            # Forward pass
            preds = model(seq, loop, dist)

            # Slice to scored length (first 68 positions)
            preds_scored = preds[:, : Config.pred_len, :]
            targets_scored = targets[:, : Config.pred_len, :]

            # Store results on CPU
            all_preds.append(preds_scored.cpu())
            all_targets.append(targets_scored.cpu())
            all_ids.extend(batch_ids)

    # Concatenate all batches
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Compute and Print Final Metric
    final_metric = mcrmse_loss(all_targets, all_preds).item()
    print(f"Final Validation Metric: {final_metric}")

    # --- Failure Analysis ---
    # Calculate RMSE per sample: Average MSE over positions and targets, then sqrt
    # Shape: (N, 68, 3) -> (N,)
    mse_per_sample = torch.mean((all_targets - all_preds) ** 2, dim=(1, 2))
    rmse_per_sample = torch.sqrt(mse_per_sample).numpy()

    # Load validation metadata to get features for correlation
    val_df = pd.read_parquet(Config.val_file)

    # Create a DataFrame linking IDs to their error
    error_df = pd.DataFrame({"id": all_ids, "error": rmse_per_sample})

    # Merge with metadata on 'id' to ensure correct alignment
    analysis_df = pd.merge(error_df, val_df, on="id", how="inner")

    print("\nFailure Analysis (Correlation with Error):")
    features_to_check = ["signal_to_noise", "SN_filter", "seq_length"]

    for feat in features_to_check:
        if feat in analysis_df.columns:
            # Ensure the column is numeric for correlation calculation
            if pd.api.types.is_numeric_dtype(analysis_df[feat]):
                corr = analysis_df["error"].corr(analysis_df[feat])
                print(f"  Correlation with {feat}: {corr:.6f}")
            else:
                print(f"  {feat} is not numeric, skipping.")
        else:
            print(f"  Feature {feat} not found in metadata.")

    # 4. Submission Generation
    # Only generate if the metric is better (lower) than the threshold
    THRESHOLD = 0.6199890971183777

    if final_metric < THRESHOLD:
        print(f"\nMetric {final_metric} < {THRESHOLD}. Generating submission...")
        generate_submission()
    else:
        print(f"\nMetric {final_metric} >= {THRESHOLD}. Skipping submission.")


if __name__ == "__main__":
    main()
