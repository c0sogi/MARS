import os
import sys
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Import from the provided library files
from library.config import Config
from library.utils import set_seed
from library.data import get_dataloaders
from library.model import ConvBiGRU
from library.train import Trainer, generate_submission


def main():
    # 1. Configuration and Setup
    # Cite solution_lesson_node_00031: "Capacity is useless without convergence."
    # Increasing epochs to 50 to allow Cosine Annealing to fully converge.
    config = Config(epochs=50, batch_size=64)

    # Override submission path as per task requirement
    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)
    config.SUBMISSION_PATH = os.path.join(submission_dir, "submission.csv")

    # Set seeds for reproducibility
    set_seed(config.SEED)

    print(f"Device: {config.DEVICE}")
    print("Initializing DataLoaders...")

    # 2. Data Loading
    # We use cached data if available to speed up the process
    train_loader, val_loader, test_loader = get_dataloaders(
        config, load_cached_data=True
    )

    # 3. Model Initialization
    model = ConvBiGRU(config)
    model = model.to(config.DEVICE)

    # 4. Training
    print("Starting Training...")
    trainer = Trainer(config, model, train_loader, val_loader)
    trainer.fit()

    # 5. Load Best Model for Evaluation
    print("Loading best model for evaluation...")
    if not os.path.exists(config.MODEL_PATH):
        print(f"Error: Model path {config.MODEL_PATH} does not exist.")
        return

    checkpoint = torch.load(
        config.MODEL_PATH, map_location=config.DEVICE, weights_only=False
    )
    # Handle state dict loading robustly
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    # 6. Validation Assessment
    print("Performing Validation Inference...")
    all_preds = []
    all_targets = []

    # Scored columns indices: reactivity (0), deg_Mg_pH10 (1), deg_Mg_50C (3)
    scored_indices = [0, 1, 3]

    with torch.no_grad():
        for features, targets in val_loader:
            features = features.to(config.DEVICE)
            targets = targets.to(config.DEVICE)

            # Forward pass
            preds = model(features)

            # Slice to prediction length (68)
            preds_sliced = preds[:, : config.PRED_LEN, :]

            all_preds.append(preds_sliced.cpu())
            all_targets.append(targets.cpu())

    # Concatenate
    val_preds = torch.cat(all_preds, dim=0)  # (N, 68, 5)
    val_targets = torch.cat(all_targets, dim=0)  # (N, 68, 5)

    # Calculate MCRMSE on Scored Columns
    # MSE per column
    mse_per_col = torch.mean(
        (val_preds - val_targets) ** 2, dim=0
    )  # (68, 5) -> mean over N -> (5,) if we flatten?
    # Actually, MCRMSE definition:
    # 1. RMSE for each column (averaging over N samples and N_t positions)
    # The metric is columnwise.
    # Let's flatten samples and sequence length first to be safe and precise with the formula.
    # Shape: (N * 68, 5)
    flat_preds = val_preds.reshape(-1, 5)
    flat_targets = val_targets.reshape(-1, 5)

    mse_cols = torch.mean((flat_preds - flat_targets) ** 2, dim=0)
    rmse_cols = torch.sqrt(mse_cols)

    # Select scored columns
    scored_rmse = rmse_cols[scored_indices]
    final_metric = torch.mean(scored_rmse).item()

    print(f"Final Validation Metric: {final_metric:.15f}")

    # 7. Failure Analysis
    print("\n==== Failure Analysis ====")
    # Calculate error per sample to correlate with metadata
    # Error per sample = Mean RMSE of the scored columns for that sample
    # val_preds: (N, 68, 5)
    diff = val_preds - val_targets
    squared_diff = diff**2
    # Mean over sequence length (dim 1)
    mse_per_sample_col = torch.mean(squared_diff, dim=1)  # (N, 5)
    rmse_per_sample_col = torch.sqrt(mse_per_sample_col)  # (N, 5)

    # Average over scored columns
    sample_errors = torch.mean(rmse_per_sample_col[:, scored_indices], dim=1).numpy()

    # Load Validation Metadata
    df_val = pd.read_parquet(config.VAL_PATH)

    # Ensure alignment (DataLoader is not shuffled for val)
    if len(df_val) != len(sample_errors):
        print(
            f"Warning: Metadata length {len(df_val)} != Predictions length {len(sample_errors)}"
        )
    else:
        # Add error to dataframe
        df_val["model_error"] = sample_errors

        # Analyze correlations
        analysis_cols = ["signal_to_noise", "SN_filter"]
        print("Correlation between Model Error and Metadata:")
        for col in analysis_cols:
            if col in df_val.columns:
                # Drop NaNs if any
                valid_data = df_val[[col, "model_error"]].dropna()
                if len(valid_data) > 1:
                    corr, _ = pearsonr(valid_data[col], valid_data["model_error"])
                    print(f"  {col}: {corr:.4f}")

        # Check error by SN_filter group
        if "SN_filter" in df_val.columns:
            print("\nAverage Error by SN_filter:")
            print(df_val.groupby("SN_filter")["model_error"].mean())

    # 8. Submission Generation
    # Threshold: 0.7247761841173526
    THRESHOLD = 0.7247761841173526

    if final_metric < THRESHOLD:
        print(
            f"\nValidation metric ({final_metric:.6f}) meets threshold ({THRESHOLD:.6f}). Generating submission..."
        )
        generate_submission(config, model, test_loader)
    else:
        print(
            f"\nValidation metric ({final_metric:.6f}) does NOT meet threshold ({THRESHOLD:.6f}). Submission skipped."
        )


if __name__ == "__main__":
    main()
