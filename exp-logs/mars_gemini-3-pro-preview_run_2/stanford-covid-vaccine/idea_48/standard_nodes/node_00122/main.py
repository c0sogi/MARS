import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import pearsonr, spearmanr

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, get_device
from library.data import get_loaders
from library.model import EIPFN
from library.engine import train_model, validate
from library.loss_metric import MCRMSELoss


def run_failure_analysis(model, val_loader, device):
    """
    Performs failure analysis on the validation set.
    Calculates per-sample error and correlates it with metadata features.
    """
    print("\n==== Failure Analysis ====")
    model.eval()

    # 1. Collect Predictions and Targets
    all_preds = []
    all_targets = []

    # We need to ensure we match these to the metadata.
    # The val_loader is created from metadata/val.csv with shuffle=False.
    # So the order should match the CSV rows.

    with torch.no_grad():
        for inputs, partner_map, targets in val_loader:
            inputs = inputs.to(device)
            partner_map = partner_map.to(device)

            # Use Pass 2 (Refined) predictions
            _, y_pred = model(inputs, partner_map)

            all_preds.append(y_pred.cpu().numpy())
            all_targets.append(targets.numpy())

    y_pred_all = np.concatenate(all_preds, axis=0)  # (N, 5, 107)
    y_true_all = np.concatenate(all_targets, axis=0)  # (N, 5, 107)

    # 2. Calculate Per-Sample RMSE (on scored columns and positions)
    # Scored columns: reactivity(0), deg_Mg_pH10(1), deg_Mg_50C(3)
    scored_indices = [0, 1, 3]
    pred_len = Config.PRED_LEN

    # Slice to scored region
    y_pred_scored = y_pred_all[:, scored_indices, :pred_len]
    y_true_scored = y_true_all[:, scored_indices, :pred_len]

    # MSE per sample: mean over (channels, length)
    mse_per_sample = np.mean((y_pred_scored - y_true_scored) ** 2, axis=(1, 2))
    rmse_per_sample = np.sqrt(mse_per_sample)

    # 3. Load Metadata
    val_csv_path = os.path.join(Config.METADATA_DIR, "val.csv")
    if not os.path.exists(val_csv_path):
        print("Validation metadata not found. Skipping detailed correlation analysis.")
        return

    val_df = pd.read_csv(val_csv_path)

    # Ensure lengths match
    if len(val_df) != len(rmse_per_sample):
        print(
            f"Warning: Metadata length ({len(val_df)}) does not match prediction length ({len(rmse_per_sample)})."
        )
        return

    val_df["model_rmse"] = rmse_per_sample

    # 4. Compute Correlations
    features_to_check = ["signal_to_noise", "mean_reactivity", "seq_length"]
    # Note: seq_length is constant (107), so correlation will be NaN, but we check anyway.

    print(f"{'Feature':<20} | {'Spearman R':<12} | {'Pearson R':<12}")
    print("-" * 50)

    for feat in features_to_check:
        if feat in val_df.columns:
            # Handle NaNs if any
            valid_mask = val_df[feat].notna() & val_df["model_rmse"].notna()
            if valid_mask.sum() > 1:
                vals = val_df.loc[valid_mask, feat]
                errs = val_df.loc[valid_mask, "model_rmse"]

                # Check for constant values to avoid warnings
                if vals.std() == 0:
                    spear_r, pear_r = 0.0, 0.0
                else:
                    spear_r, _ = spearmanr(vals, errs)
                    pear_r, _ = pearsonr(vals, errs)

                print(f"{feat:<20} | {spear_r:.4f}       | {pear_r:.4f}")


def generate_submission(model, test_loader, device, threshold_met):
    """
    Generates submission file if threshold is met.
    """
    if not threshold_met:
        print("Validation metric threshold not met. Skipping submission generation.")
        return

    print("\nGenerating submission...")
    model.eval()

    ids_list = []
    preds_list = []

    with torch.no_grad():
        for inputs, partner_map, sample_ids in test_loader:
            inputs = inputs.to(device)
            partner_map = partner_map.to(device)

            # Inference (Pass 2)
            _, y_pred = model(inputs, partner_map)

            # y_pred: (B, 5, 107)
            preds_list.append(y_pred.cpu().numpy())
            ids_list.extend(sample_ids)

    all_preds = np.concatenate(preds_list, axis=0)  # (N_test, 5, 107)

    # Reshape for submission
    # Format: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
    # Columns in output tensor are ordered as in Config.TARGET_COLS:
    # ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']

    submission_rows = []
    target_cols = Config.TARGET_COLS

    for i, sample_id in enumerate(ids_list):
        sample_pred = all_preds[i]  # (5, 107)

        for pos in range(Config.SEQ_LEN):
            row_id = f"{sample_id}_{pos}"
            row_values = sample_pred[:, pos]

            row_dict = {"id_seqpos": row_id}
            for idx, col_name in enumerate(target_cols):
                row_dict[col_name] = float(row_values[idx])

            submission_rows.append(row_dict)

    submission_df = pd.DataFrame(submission_rows)

    # Save
    sub_path = "./submission/submission.csv"
    os.makedirs(os.path.dirname(sub_path), exist_ok=True)
    submission_df.to_csv(sub_path, index=False)
    print(f"Submission saved to {sub_path}")


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = get_device()

    # Override Config for Fast Baseline
    Config.EPOCHS = 15  # Reduced from 50 for speed

    print(f"Configuration:")
    print(f"  Device: {device}")
    print(f"  Epochs: {Config.EPOCHS}")
    print(f"  Batch Size: {Config.BATCH_SIZE}")
    print(f"  Working Dir: {Config.WORKING_DIR}")

    # 2. Data Loading
    print("\nLoading Data...")
    train_loader, val_loader, test_loader = get_loaders(load_cached_data=True)

    # 3. Model Initialization
    print("\nInitializing Model...")
    model = EIPFN().to(device)

    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    # 4. Training
    print("\nStarting Training...")
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=Config.EPOCHS,
        device=device,
        patience=Config.PATIENCE,
    )

    # 5. Final Validation & Analysis
    print("\nLoading best model for validation...")
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))

    final_metric = validate(model, val_loader, device)
    print(f"Final Validation Metric: {final_metric:.15f}")

    run_failure_analysis(model, val_loader, device)

    # 6. Submission
    # Threshold from prompt: 0.47142532743789534
    threshold = 0.47142532743789534
    threshold_met = final_metric < threshold

    # Ensure output directory exists
    os.makedirs("submission", exist_ok=True)

    # Generate submission if threshold is met (or just generate it as per standard practice to have artifacts)
    # The prompt says "If and only if", so we respect the logic strictly.
    if threshold_met:
        generate_submission(model, test_loader, device, threshold_met=True)
    else:
        print(f"Metric {final_metric} >= {threshold}. Submission not generated.")


if __name__ == "__main__":
    main()
