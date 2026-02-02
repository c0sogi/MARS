import os
import torch
import pandas as pd
import numpy as np
import scipy.stats as stats
from library.config import Config
from library.utils import set_seed, MCRMSELoss
from library.data import get_dataloaders
from library.model import CascadedDenseNet
from library.train import train_model, generate_submission


def run_failure_analysis(val_loader, device):
    """
    Performs failure analysis on the validation set to identify error patterns.
    """
    print("\n==== Starting Failure Analysis ====")

    # Load the best model
    model = CascadedDenseNet().to(device)
    if not os.path.exists(Config.MODEL_PATH):
        print("Model file not found. Skipping failure analysis.")
        return

    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.eval()

    # Store predictions and targets
    all_preds = []
    all_targets = []

    # Inference loop
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["inputs"].to(device)
            partners = batch["partner_indices"].to(device)
            targets = batch["targets"].to(device)

            outputs = model(inputs, partners)

            # Slice predictions to match target length (107 -> 68)
            if outputs.shape[1] > targets.shape[1]:
                outputs = outputs[:, : targets.shape[1], :]

            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Identify scored columns indices
    scored_indices = [
        i for i, col in enumerate(Config.TARGET_COLS) if col in Config.SCORED_COLS
    ]

    # Filter for scored columns
    preds_scored = all_preds[:, :, scored_indices]
    targets_scored = all_targets[:, :, scored_indices]

    # Calculate MCRMSE per sample
    # Shape: (N, 68, 3) -> Mean over (68, 3) is not quite right for MCRMSE definition
    # MCRMSE is mean of RMSEs of columns.
    # For failure analysis, we want a scalar "error score" per sample.
    # We will use the mean squared error per sample averaged over columns and positions.
    mse_per_sample = np.mean((preds_scored - targets_scored) ** 2, axis=(1, 2))
    rmse_per_sample = np.sqrt(mse_per_sample)

    # Load Validation Metadata to get features
    val_df = pd.read_csv(Config.VAL_CSV)

    # Ensure alignment (loader is sequential, csv is sequential)
    if len(val_df) != len(rmse_per_sample):
        print(
            f"Warning: Metadata length ({len(val_df)}) != Predictions ({len(rmse_per_sample)})"
        )
        # Truncate to match (in case of debug subset)
        min_len = min(len(val_df), len(rmse_per_sample))
        val_df = val_df.iloc[:min_len]
        rmse_per_sample = rmse_per_sample[:min_len]

    # Feature Engineering for Analysis
    val_df["error_rmse"] = rmse_per_sample

    # 1. Signal to Noise
    val_df["signal_to_noise"] = pd.to_numeric(
        val_df["signal_to_noise"], errors="coerce"
    ).fillna(0)

    # 2. Sequence Composition (GC Content)
    val_df["gc_content"] = val_df["sequence"].apply(
        lambda x: (x.count("G") + x.count("C")) / len(x)
    )

    # 3. Structure Composition (Paired Percentage)
    val_df["paired_pct"] = val_df["structure"].apply(
        lambda x: (x.count("(") + x.count(")")) / len(x)
    )

    # Calculate Correlations
    features_to_check = [
        "signal_to_noise",
        "gc_content",
        "paired_pct",
        "mean_reactivity",
    ]

    print(f"{'Feature':<20} | {'Correlation with Error (RMSE)':<30}")
    print("-" * 60)

    for feat in features_to_check:
        if feat in val_df.columns:
            corr, _ = stats.pearsonr(val_df[feat], val_df["error_rmse"])
            print(f"{feat:<20} | {corr:.4f}")

    print("\nAnalysis Interpretation:")
    print(" - Negative correlation with S/N: Model performs better on cleaner data.")
    print(
        " - Positive correlation with GC/Paired: Model struggles with complex/stable structures."
    )


def main():
    # 1. Configuration & Setup
    # Override Config for a fast baseline execution while retaining sufficient capacity
    Config.EPOCHS = 20  # Reduced from 50 to ensure < 2 hours runtime
    Config.BATCH_SIZE = 32  # Increase batch size slightly for A100 speedup

    print("Initializing Fast Baseline Run...")
    set_seed(Config.SEED)

    # 2. Training
    # train_model handles data loading, training loop, and saving the best model
    best_val_score = train_model(debug=False)

    # 3. Validation Reporting
    # The task requires printing the metric in a specific format with full precision
    print(f"Final Validation Metric: {best_val_score}")

    # 4. Failure Analysis
    # Get validation loader again for analysis
    _, val_loader, _ = get_dataloaders(debug=False, load_cached_data=True)
    device = torch.device(Config.DEVICE)
    run_failure_analysis(val_loader, device)

    # 5. Submission
    # Threshold defined in task: 0.5421870350837708
    THRESHOLD = 0.5421870350837708

    if best_val_score < THRESHOLD:
        print(f"\nValidation score ({best_val_score}) meets threshold ({THRESHOLD}).")
        generate_submission(debug=False)
    else:
        print(
            f"\nValidation score ({best_val_score}) did not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
