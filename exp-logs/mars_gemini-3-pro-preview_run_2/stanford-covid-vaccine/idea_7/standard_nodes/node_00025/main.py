import os
import sys
import numpy as np
import pandas as pd
import torch

# Ensure the current directory is in the python path to locate the library
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed
from library.train import run_training, validate
from library.inference import run_inference
from library.data import get_dataloaders
from library.loss import MaskedMCRMSELoss
from library.model import HybridNet


def pearson_corr(x, y):
    """Calculates Pearson correlation coefficient between two numpy arrays."""
    if len(x) < 2:
        return 0.0
    # Handle potential constant arrays to avoid division by zero
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return np.corrcoef(x, y)[0, 1]


def failure_analysis(model, val_loader, device):
    """
    Performs failure analysis on the validation set.
    Calculates per-sample error and correlates with metadata features.
    """
    print("\nPerforming Failure Analysis...")
    model.eval()

    all_ids = []
    all_preds = []
    all_targets = []

    # 1. Collect Predictions and Targets
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["inputs"].to(device)
            partner_indices = batch["partner_indices"].to(device)
            targets = batch["targets"].to(device)
            ids = batch["id"]

            preds = model(inputs, partner_indices)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_ids.extend(ids)

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # 2. Calculate Per-Sample Error (MCRMSE)
    # Align shapes first (slice preds to target length)
    seq_len_target = all_targets.shape[1]
    if all_preds.shape[1] > seq_len_target:
        all_preds = all_preds[:, :seq_len_target, :]

    # Identify scored columns
    scored_cols = Config.SCORED_COLS
    target_cols = Config.TARGET_COLS
    scored_indices = [i for i, col in enumerate(target_cols) if col in scored_cols]

    # Filter columns
    preds_scored = all_preds[:, :, scored_indices]
    targets_scored = all_targets[:, :, scored_indices]

    # Calculate RMSE per column per sample
    # Squared Error: (N, L, C)
    sq_error = (preds_scored - targets_scored) ** 2

    # Mean Squared Error per sample per column (averaging over sequence length L)
    # Note: We assume dense targets as per dataset description.
    # If using nanmean, it handles NaNs if any.
    mse_per_sample_col = np.nanmean(sq_error, axis=1)  # Shape: (N, C)

    # RMSE per sample per column
    rmse_per_sample_col = np.sqrt(mse_per_sample_col)

    # MCRMSE per sample (averaging over columns C)
    sample_errors = np.mean(rmse_per_sample_col, axis=1)  # Shape: (N,)

    # 3. Merge with Metadata
    val_csv_path = Config.VAL_CSV
    if not os.path.exists(val_csv_path):
        print(
            f"Warning: Metadata file {val_csv_path} not found. Skipping correlation analysis."
        )
        return

    val_df = pd.read_csv(val_csv_path)

    # Create error dataframe
    error_df = pd.DataFrame({"id": all_ids, "model_error": sample_errors})

    # Merge
    analysis_df = pd.merge(error_df, val_df, on="id", how="left")

    # 4. Correlate
    # Features to check: signal_to_noise, mean_reactivity
    features = ["signal_to_noise", "mean_reactivity"]

    print("-" * 50)
    print(f"{'Feature':<20} | {'Correlation':<15}")
    print("-" * 50)

    for feat in features:
        if feat in analysis_df.columns:
            # Drop NaNs
            valid_data = analysis_df[[feat, "model_error"]].dropna()
            if len(valid_data) > 1:
                corr = pearson_corr(
                    valid_data[feat].values, valid_data["model_error"].values
                )
                print(f"{feat:<20} | {corr:.4f}")
            else:
                print(f"{feat:<20} | Not enough data")
        else:
            print(f"{feat:<20} | Not found in metadata")
    print("-" * 50)


def main():
    # 1. Configure for Fast Baseline
    # Limit epochs to ensure completion within time limits.
    # We do not limit samples via Config.DEBUG because it would truncate the test set,
    # preventing valid submission generation.
    Config.EPOCHS = 15

    # Ensure we use GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Starting Runfile Execution...")
    print(f"Device: {device}")

    # 2. Run Training
    # run_training handles data loading, model init, training loop, and saving best model
    model = run_training(epochs=Config.EPOCHS, load_cached_data=True)

    # 3. Validation Assessment
    print("\nRunning Final Validation...")
    _, val_loader, _ = get_dataloaders(load_cached_data=True)
    criterion = MaskedMCRMSELoss().to(device)

    # Ensure model is in eval mode and on correct device
    model.to(device)
    model.eval()

    val_score, _ = validate(model, val_loader, criterion, device)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {val_score}")

    # 4. Failure Analysis
    failure_analysis(model, val_loader, device)

    # 5. Submission
    THRESHOLD = 0.6477736930052439
    if val_score < THRESHOLD:
        print(
            f"\nValidation score ({val_score}) meets threshold ({THRESHOLD}). Generating submission..."
        )
        submission_dir = "./submission"
        os.makedirs(submission_dir, exist_ok=True)
        submission_path = os.path.join(submission_dir, "submission.csv")

        run_inference(
            model_path=Config.MODEL_CHECKPOINT,
            output_path=submission_path,
            load_cached_data=True,
        )
    else:
        print(
            f"\nValidation score ({val_score}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
