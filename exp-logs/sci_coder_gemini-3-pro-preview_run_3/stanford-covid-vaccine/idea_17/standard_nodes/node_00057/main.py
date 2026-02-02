import os
import sys
import torch
import pandas as pd
import numpy as np

# Ensure library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.train import run_training, validate_global
from library.model import generate_submission, DISR_BiGRU
from library.data import get_dataloaders
from library.utils import set_seed


def perform_failure_analysis(model, val_loader, config):
    """
    Calculates per-sample errors on the validation set and correlates them
    with input features to identify systematic weaknesses.
    """
    print("\nPerforming Failure Analysis...")

    # Load validation metadata
    val_df = pd.read_parquet(config.val_metadata_path)

    # Collect predictions and targets
    all_preds = []
    all_targets = []
    all_masks = []

    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            features = batch["features"].to(config.device)
            pair_indices = batch["pair_indices"].to(config.device)
            targets = batch["targets"].to(config.device)
            mask = batch["mask"].to(config.device)

            preds = model(features, pair_indices)

            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())
            all_masks.append(mask.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_masks = torch.cat(all_masks, dim=0)

    # Filter for Scored Columns: reactivity(0), deg_Mg_pH10(1), deg_Mg_50C(3)
    scored_indices = [0, 1, 3]

    preds_scored = all_preds[:, :, scored_indices]
    targets_scored = all_targets[:, :, scored_indices]

    # Expand mask for broadcasting: (N, L) -> (N, L, 1)
    masks_expanded = all_masks.unsqueeze(-1)

    # Calculate Squared Error per element: (N, L, 3)
    squared_errors = (preds_scored - targets_scored) ** 2
    squared_errors = squared_errors * masks_expanded

    # Calculate Mean Squared Error per sample per target
    # Sum over length (L), divide by number of scored positions
    seq_scored_counts = all_masks.sum(dim=1).unsqueeze(-1)  # (N, 1)
    # Avoid division by zero
    seq_scored_counts = torch.clamp(seq_scored_counts, min=1.0)

    mse_per_sample_target = squared_errors.sum(dim=1) / seq_scored_counts  # (N, 3)

    # RMSE per sample per target
    rmse_per_sample_target = torch.sqrt(mse_per_sample_target)

    # MCRMSE per sample: Mean over the 3 targets
    mcrmse_per_sample = rmse_per_sample_target.mean(dim=1).numpy()

    # Align with DataFrame
    # Note: DataLoader is sequential for validation, so order is preserved
    if len(val_df) != len(mcrmse_per_sample):
        # In case of debug subsetting
        val_df = val_df.iloc[: len(mcrmse_per_sample)].copy()

    val_df["error"] = mcrmse_per_sample

    # Feature Engineering
    # Calculate composition percentages
    val_df["pct_A"] = val_df["sequence"].apply(lambda x: x.count("A") / len(x))
    val_df["pct_G"] = val_df["sequence"].apply(lambda x: x.count("G") / len(x))
    val_df["pct_C"] = val_df["sequence"].apply(lambda x: x.count("C") / len(x))
    val_df["pct_U"] = val_df["sequence"].apply(lambda x: x.count("U") / len(x))
    # Calculate paired percentage (structure density)
    val_df["pct_paired"] = val_df["structure"].apply(
        lambda x: (x.count("(") + x.count(")")) / len(x)
    )

    # Calculate Correlations
    features_to_check = [
        "signal_to_noise",
        "pct_A",
        "pct_G",
        "pct_C",
        "pct_U",
        "pct_paired",
    ]
    # Ensure signal_to_noise is float
    val_df["signal_to_noise"] = val_df["signal_to_noise"].astype(float)

    correlations = val_df[features_to_check].corrwith(val_df["error"])
    print("Correlation between Error and Features:")
    print(correlations)


def main():
    # 1. Setup Configuration
    config = Config()

    # Override for Fast Baseline
    config.epochs = 15

    # Ensure submission directory exists and set path
    os.makedirs("./submission", exist_ok=True)
    config.submission_path = "./submission/submission.csv"

    # Set seeds for reproducibility
    set_seed(config.seed)

    # 2. Run Training
    run_training(config)

    # 3. Validation Assessment
    print("\nLoading best model for validation...")
    model = DISR_BiGRU(config).to(config.device)
    model.load_state_dict(
        torch.load(config.model_save_path, map_location=config.device)
    )

    _, val_loader, _ = get_dataloaders(config)

    # Calculate and Print Final Metric
    val_metric = validate_global(model, val_loader, config)
    print(f"Final Validation Metric: {val_metric}")

    # 4. Failure Analysis
    perform_failure_analysis(model, val_loader, config)

    # 5. Conditional Submission
    THRESHOLD = 0.7247761841173526

    if val_metric < THRESHOLD:
        print(f"\nValidation metric {val_metric} is better than threshold {THRESHOLD}.")
        print("Generating submission...")
        generate_submission(config)
    else:
        print(f"\nValidation metric {val_metric} did not meet threshold {THRESHOLD}.")
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
