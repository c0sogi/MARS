import os
import time
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import functions and classes from the provided library
from library.config import Config
from library.utils import seed_everything, get_device
from library.dataset import get_dataloaders, get_test_loader
from library.model import RNAModel
from library.loss import MCRMSELoss
from library.train import train_one_epoch, validate, inference_and_submission
from library.metrics import calculate_competition_metric


def run_failure_analysis(model, val_loader, device, config):
    """
    Performs failure analysis on the validation set.
    Calculates error magnitude per sample and correlates it with metadata features.
    """
    print("\n==== Failure Analysis ====")
    model.eval()

    all_preds = []
    all_targets = []

    # 1. Collect Predictions and Targets
    with torch.no_grad():
        for inputs, bpp_indices, bpp_masks, targets in val_loader:
            inputs = inputs.to(device)
            bpp_indices = bpp_indices.to(device)
            bpp_masks = bpp_masks.to(device)
            targets = targets.to(device)

            preds = model(inputs, bpp_indices, bpp_masks)

            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

    all_preds = torch.cat(all_preds, dim=0)  # (N, 107, 5)
    all_targets = torch.cat(all_targets, dim=0)  # (N, 107, 5)

    # 2. Calculate Error Magnitude per Sample
    # We focus on the scored columns and positions for the error metric
    # Slice to scored length
    preds_sliced = all_preds[:, : config.pred_len, :]
    targets_sliced = all_targets[:, : config.pred_len, :]

    # Identify scored indices
    scored_indices = [
        i for i, col in enumerate(config.target_cols) if col in config.scored_cols
    ]

    # Filter columns
    preds_filtered = preds_sliced[:, :, scored_indices]
    targets_filtered = targets_sliced[:, :, scored_indices]

    # Calculate MSE per sample (averaged over positions and scored columns)
    # Shape: (N, 68, 3) -> (N,)
    mse_per_sample = torch.mean((preds_filtered - targets_filtered) ** 2, dim=(1, 2))
    rmse_per_sample = torch.sqrt(mse_per_sample).numpy()

    # 3. Load Metadata
    # We assume the validation loader preserves the order of the validation parquet file
    val_df = pd.read_parquet(config.val_file)

    # If config used max_samples, slice the df
    if config.max_val_samples is not None:
        val_df = val_df.iloc[: config.max_val_samples].reset_index(drop=True)

    # Ensure lengths match
    if len(val_df) != len(rmse_per_sample):
        print(
            f"Warning: Metadata length ({len(val_df)}) does not match predictions ({len(rmse_per_sample)}). Skipping correlation analysis."
        )
        return

    # Add error to dataframe
    val_df["model_error"] = rmse_per_sample

    # 4. Feature Engineering for Correlation
    # Extract sequence properties
    val_df["pct_A"] = val_df["sequence"].apply(lambda s: s.count("A") / len(s))
    val_df["pct_G"] = val_df["sequence"].apply(lambda s: s.count("G") / len(s))
    val_df["pct_U"] = val_df["sequence"].apply(lambda s: s.count("U") / len(s))
    val_df["pct_C"] = val_df["sequence"].apply(lambda s: s.count("C") / len(s))
    val_df["pct_unpaired"] = val_df["structure"].apply(lambda s: s.count(".") / len(s))

    # Select features for correlation
    analysis_cols = [
        "signal_to_noise",
        "SN_filter",
        "pct_A",
        "pct_G",
        "pct_U",
        "pct_C",
        "pct_unpaired",
    ]
    # Filter only existing columns
    analysis_cols = [c for c in analysis_cols if c in val_df.columns]

    # Compute correlations
    correlations = (
        val_df[analysis_cols]
        .corrwith(val_df["model_error"])
        .sort_values(ascending=False)
    )

    print("Correlation between Model Error (RMSE) and Input Features:")
    print(correlations)
    print("==========================\n")


def main():
    # 1. Setup and Configuration
    seed_everything(42)

    # Initialize Config and override for Fast Baseline
    config = Config()
    config.num_epochs = 5  # Limit epochs for speed
    config.max_train_samples = (
        None  # Use full training set (1728 samples is small enough)
    )
    config.max_val_samples = None  # Use full validation set

    # Update submission path to match requirements
    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)
    config.submission_file = os.path.join(submission_dir, "submission.csv")

    device = get_device()
    print(f"Running on device: {device}")

    # 2. Data Loading
    # Uses caching implicitly via library functions
    print("Loading data...")
    train_loader, val_loader = get_dataloaders(config, load_cached_data=True)

    # 3. Model Initialization
    model = RNAModel(config).to(device)

    # 4. Training Setup
    optimizer = optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.num_epochs)
    criterion = MCRMSELoss()

    best_score = float("inf")

    # 5. Training Loop
    print(f"Starting training for {config.num_epochs} epochs...")
    for epoch in range(config.num_epochs):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, config
        )

        # Validate
        val_score = validate(model, val_loader, device, config)

        # Scheduler Step
        scheduler.step()

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch+1}/{config.num_epochs} | Train Loss: {train_loss:.6f} | Val MCRMSE: {val_score:.6f} | Time: {elapsed:.2f}s"
        )

        # Save Best Model
        if val_score < best_score:
            best_score = val_score
            torch.save(model.state_dict(), config.model_save_path)

    # 6. Final Evaluation
    print("Loading best model for final evaluation...")
    model.load_state_dict(torch.load(config.model_save_path, map_location=device))

    # Re-calculate metric on full validation set to be precise
    final_val_metric = validate(model, val_loader, device, config)
    print(f"Final Validation Metric: {final_val_metric}")

    # 7. Failure Analysis
    run_failure_analysis(model, val_loader, device, config)

    # 8. Conditional Submission
    # Threshold from task description: 0.5884495377540588
    THRESHOLD = 0.5884495377540588

    if final_val_metric < THRESHOLD:
        print(
            f"Validation metric ({final_val_metric}) is better than threshold ({THRESHOLD}). Generating submission..."
        )
        inference_and_submission(model, device, config)
    else:
        print(
            f"Validation metric ({final_val_metric}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
