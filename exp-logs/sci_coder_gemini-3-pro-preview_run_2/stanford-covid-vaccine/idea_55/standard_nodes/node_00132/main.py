import os
import pandas as pd
import numpy as np
import torch
import scipy.stats as stats
import warnings

# Import from provided libraries
from library.config import Config
from library.train import run_training
from library.model import DDARN
from library.data import get_dataloaders, get_data
from library.utils import set_seed, get_device
from library.loss import MaskedMCRMSELoss

# Suppress warnings
warnings.filterwarnings("ignore")


def perform_failure_analysis(model, device):
    """
    Analyzes model performance on the validation set and correlates errors with metadata.
    Returns the global MCRMSE metric.
    """
    print("\nPerforming Failure Analysis...")

    # Load validation metadata to get signal_to_noise
    val_meta_path = os.path.join(Config.METADATA_DIR, "val.csv")
    if not os.path.exists(val_meta_path):
        print("Validation metadata not found. Skipping failure analysis.")
        return float("inf")

    df_val = pd.read_csv(val_meta_path)

    # Load validation data via loader to ensure consistent processing
    val_loader = get_dataloaders(
        mode="val", load_cached=True, batch_size=Config.BATCH_SIZE, num_workers=2
    )

    # Get raw IDs from the data source to ensure alignment
    # library.data.get_dataloaders uses library.config.get_data internally
    _, _, _, val_ids_ordered = get_data("val", load_cached=True)

    model.eval()

    # Accumulate predictions and targets
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X, partners, y in val_loader:
            X, partners, y = X.to(device), partners.to(device), y.to(device)

            # Inference (Pass 2 logic as per DDARN strategy)
            pred_1 = model(X, partners, prev_pred=None)
            fb_input = pred_1.clone()
            fb_input[:, :, Config.PRED_LEN :] = 0.0
            pred_2 = model(X, partners, prev_pred=fb_input)

            all_preds.append(pred_2.cpu())
            all_targets.append(y.cpu())

    # Concatenate
    preds_tensor = torch.cat(all_preds, dim=0)  # (N, 5, L)
    targets_tensor = torch.cat(all_targets, dim=0)  # (N, L, 5)

    # Align targets if necessary (N, L, 5) -> (N, 5, L)
    if targets_tensor.shape[1] != 5 and targets_tensor.shape[2] == 5:
        targets_tensor = targets_tensor.permute(0, 2, 1)

    # Select scored columns and positions
    preds_scored = preds_tensor[:, Config.SCORED_TARGETS, : Config.PRED_LEN]
    targets_scored = targets_tensor[:, Config.SCORED_TARGETS, : Config.PRED_LEN]

    # --- Per Sample Analysis ---
    # MSE per sample: (N, 3, 68)
    mse = (preds_scored - targets_scored) ** 2
    # RMSE per column per sample: (N, 3)
    rmse_per_col = torch.sqrt(torch.mean(mse, dim=2))
    # MCRMSE per sample: Mean over columns (N,)
    mcrmse_per_sample = torch.mean(rmse_per_col, dim=1).numpy()

    # Create DataFrame for analysis
    analysis_df = pd.DataFrame({"id": val_ids_ordered, "error": mcrmse_per_sample})

    # Merge with metadata
    merged_df = pd.merge(analysis_df, df_val, on="id", how="left")

    # Correlation with Signal to Noise
    if "signal_to_noise" in merged_df.columns:
        clean_df = merged_df.dropna(subset=["error", "signal_to_noise"])
        if len(clean_df) > 0:
            corr, _ = stats.pearsonr(clean_df["error"], clean_df["signal_to_noise"])
            print(f"Correlation between Error and Signal_to_Noise: {corr:.4f}")
        else:
            print("Not enough data for correlation analysis.")
    else:
        print("signal_to_noise column not found in metadata.")

    # --- Global Metric Calculation ---
    # Global MCRMSE: Sqrt of (Sum of Squared Errors / Total Count) per column, then Mean over columns
    total_sse = torch.sum((preds_scored - targets_scored) ** 2, dim=(0, 2))  # (3,)
    total_count = preds_scored.shape[0] * preds_scored.shape[2]
    col_rmse = torch.sqrt(total_sse / total_count)
    global_mcrmse = torch.mean(col_rmse).item()

    print(f"Final Validation Metric: {global_mcrmse}")
    return global_mcrmse


def generate_submission(model, device):
    """
    Generates submission file for the test set.
    """
    print("Generating submission...")

    # Get test IDs
    _, _, _, test_ids = get_data("test", load_cached=True)

    loader = get_dataloaders(
        mode="test", load_cached=True, batch_size=Config.BATCH_SIZE, num_workers=2
    )

    model.eval()
    preds_list = []

    with torch.no_grad():
        for X, partners in loader:
            X, partners = X.to(device), partners.to(device)

            # Inference (Pass 2 logic)
            pred_1 = model(X, partners, prev_pred=None)
            fb_input = pred_1.clone()
            fb_input[:, :, Config.PRED_LEN :] = 0.0
            pred_2 = model(X, partners, prev_pred=fb_input)

            # Output is (N, 5, L). Transpose to (N, L, 5) for CSV format
            preds_list.append(pred_2.permute(0, 2, 1).cpu().numpy())

    preds_array = np.concatenate(preds_list, axis=0)  # (N_samples, 107, 5)

    # Prepare submission dataframe
    submission_rows = []
    target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    for i, sample_id in enumerate(test_ids):
        for seq_pos in range(Config.SEQ_LEN):
            row_id = f"{sample_id}_{seq_pos}"
            row_vals = preds_array[i, seq_pos, :]

            # Clip negative values
            row_vals = np.clip(row_vals, 0, None)

            row_dict = {"id_seqpos": row_id}
            for idx, col in enumerate(target_cols):
                row_dict[col] = float(row_vals[idx])
            submission_rows.append(row_dict)

    submission_df = pd.DataFrame(submission_rows)

    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def main():
    # 1. Setup
    set_seed(42)
    device = get_device()

    # 2. Training
    # Executes the training loop defined in library.train
    # Saves best model to working/idea_55/best_model.pth
    run_training(num_epochs=Config.NUM_EPOCHS, load_cached=True)

    # 3. Load Best Model
    model = DDARN().to(device)
    best_model_path = os.path.join(Config.CACHE_DIR, "best_model.pth")

    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print("Warning: Best model not found. Using random weights.")

    # 4. Validation & Failure Analysis
    val_metric = perform_failure_analysis(model, device)

    # 5. Submission
    threshold = 0.47142532743789534
    if val_metric < threshold:
        generate_submission(model, device)
    else:
        print(
            f"Validation metric {val_metric} is not lower than threshold {threshold}. Skipping submission."
        )


if __name__ == "__main__":
    main()
