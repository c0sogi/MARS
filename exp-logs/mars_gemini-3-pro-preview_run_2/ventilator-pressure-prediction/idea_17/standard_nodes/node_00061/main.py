import pandas as pd
import numpy as np
import torch
import os
import sys

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, load_checkpoint
from library.dataset import get_data_loaders, get_test_loader
from library.model import DPGIBiLSTM
from library.train import run_training, WeightedL1Loss

# ==========================================
# Configuration Overrides for Fast Baseline
# ==========================================
# Adjust settings to ensure execution within 2 hours while maintaining
# sufficient capacity to reach the target metric (< 0.162).
# A100 GPU allows for larger batch size and reasonable epoch count.
Config.EPOCHS = 30
Config.SCHEDULER_T_MAX = 30
Config.BATCH_SIZE = 512
Config.DEBUG = False  # Must use full dataset to achieve required accuracy


def main():
    # 1. Setup and Reproducibility
    Config.setup()
    seed_everything(Config.SEED)
    print(f"Starting Fast Baseline Run.")
    print(
        f"Settings: Epochs={Config.EPOCHS}, Batch Size={Config.BATCH_SIZE}, Device={Config.DEVICE}"
    )

    # 2. Training Phase
    # run_training handles data loading, preprocessing (caching), and the training loop.
    # It saves the best model to Config.BEST_MODEL_PATH.
    run_training()

    # 3. Validation Phase
    print("\nStarting Validation and Failure Analysis...")

    # Load the best model checkpoint
    model = DPGIBiLSTM().to(Config.DEVICE)
    checkpoint = load_checkpoint(model, Config.BEST_MODEL_PATH, device=Config.DEVICE)

    if checkpoint is None:
        print("Error: Best model checkpoint not found. Training may have failed.")
        sys.exit(1)

    print(
        f"Loaded best model from epoch {checkpoint['epoch']} with Val MAE: {checkpoint['best_mae']}"
    )

    # Get validation data loader (uses cached data)
    _, val_loader = get_data_loaders(load_cached_data=True)

    # Run Inference
    model.eval()
    all_preds = []
    all_targets = []
    all_u_out = []
    all_inputs = []

    with torch.no_grad():
        for X, u_out, y in val_loader:
            X = X.to(Config.DEVICE)
            u_out = u_out.to(Config.DEVICE)
            y = y.to(Config.DEVICE)

            preds = model(X)

            # Store results on CPU for analysis
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            all_u_out.append(u_out.cpu().numpy())
            all_inputs.append(X.cpu().numpy())

    # Flatten results
    preds_flat = np.concatenate(all_preds).flatten()
    targets_flat = np.concatenate(all_targets).flatten()
    u_out_flat = np.concatenate(all_u_out).flatten()
    # Inputs: (N_breaths, 80, 14) -> (Total_Steps, 14)
    inputs_flat = np.concatenate(all_inputs).reshape(-1, 14)

    # Calculate Metric: MAE on Inspiratory Phase (u_out == 0)
    insp_mask = u_out_flat == 0
    abs_errors = np.abs(preds_flat - targets_flat)

    if insp_mask.sum() > 0:
        final_metric = np.mean(abs_errors[insp_mask])
    else:
        final_metric = float("inf")

    # REQUIRED OUTPUT: Print full precision metric
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    print("\nFailure Analysis (Feature Correlations with Error):")

    # Map feature indices to names (based on library.model and dataset)
    # 0:time_step, 1:u_in, 2:R, 3:C, 4:volume, 5:R_u_in, 6:vol_C,
    # 7:u_in_diff1, 8:u_in_diff2, 9-12:u_in_lag1-4, 13:u_out
    feature_names = [
        "time_step",
        "u_in",
        "R",
        "C",
        "volume",
        "R_u_in",
        "vol_C",
        "u_in_diff1",
        "u_in_diff2",
        "u_in_lag1",
        "u_in_lag2",
        "u_in_lag3",
        "u_in_lag4",
        "u_out",
    ]

    df_analysis = pd.DataFrame(inputs_flat, columns=feature_names)
    df_analysis["abs_error"] = abs_errors

    # Analyze correlations only on the scored inspiratory phase
    df_insp = df_analysis[insp_mask]
    correlations = (
        df_insp.corr()["abs_error"].drop("abs_error").sort_values(ascending=False)
    )
    print(correlations)

    # 5. Submission Generation
    TARGET_THRESHOLD = 0.1619843989610672

    if final_metric < TARGET_THRESHOLD:
        print(
            f"\nMetric {final_metric} meets threshold {TARGET_THRESHOLD}. Generating submission..."
        )

        test_loader = get_test_loader(load_cached_data=True)
        test_preds = []

        with torch.no_grad():
            for X, u_out in test_loader:
                X = X.to(Config.DEVICE)
                preds = model(X)
                test_preds.append(preds.cpu().numpy())

        test_preds_flat = np.concatenate(test_preds).flatten()

        # Load test metadata to map predictions to IDs
        # Data loader yields breaths sorted by breath_id, then time_step.
        # We sort metadata similarly to ensure alignment.
        df_test_meta = pd.read_csv(Config.TEST_META)
        df_test_meta = df_test_meta.sort_values([Config.BREATH_COL, Config.ID_COL])

        if len(df_test_meta) != len(test_preds_flat):
            print(
                f"Warning: Prediction count {len(test_preds_flat)} matches metadata {len(df_test_meta)}?"
            )

        df_test_meta["pressure"] = test_preds_flat
        submission = df_test_meta[[Config.ID_COL, "pressure"]]

        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nMetric {final_metric} does not meet threshold {TARGET_THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
