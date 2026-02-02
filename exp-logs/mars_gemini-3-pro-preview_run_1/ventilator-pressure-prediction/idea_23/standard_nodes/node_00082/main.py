import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Import library components
from library.config import Config
from library.train import Trainer
from library.utils import seed_everything, get_device
from library.dataset import VentilatorDataset
from library.model import GraduatedCapacityNetwork
from library.features import engineer_features

# =========================================================================
# Configuration Overrides for Fast Baseline
# =========================================================================
# Reduce epochs to ensure completion within 2 hours while allowing convergence
Config.EPOCHS = 6
# Ensure we use the full dataset to meet the strict performance threshold
Config.DEBUG = False


def get_feature_names():
    """
    Helper to reconstruct feature names by running the engineering pipeline
    on a dummy dataframe. This is needed for failure analysis interpretation.
    """
    # Create a dummy dataframe with the initial columns
    dummy_df = pd.DataFrame(
        {
            "id": [1, 2],
            "breath_id": [1, 1],
            "R": [20, 20],
            "C": [50, 50],
            "time_step": [0.0, 0.1],
            "u_in": [0.0, 10.0],
            "u_out": [0, 0],
            "pressure": [5.0, 6.0],
        }
    )

    # Run engineering
    df_eng = engineer_features(dummy_df)

    # Apply the same exclusion logic as in library/features.py
    exclude_cols = [
        Config.ID_COL,
        Config.BREATH_ID_COL,
        Config.TARGET_COL,
        Config.U_OUT_COL,
    ]
    feature_cols = [c for c in df_eng.columns if c not in exclude_cols]

    # Add u_out which is appended as the last feature in prepare_dataset
    feature_cols.append("u_out")
    return feature_cols


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = get_device()
    print(f"Running on device: {device}")

    # 2. Training
    print("\n=== Starting Training ===")
    trainer = Trainer()
    trainer.fit()

    # 3. Validation & Metric Calculation
    print("\n=== Validation & Failure Analysis ===")

    # Load best model
    input_dim = trainer.train_dataset.X.shape[-1]
    model = GraduatedCapacityNetwork(input_dim=input_dim)
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    val_loader = trainer.val_loader

    all_preds = []
    all_targets = []
    all_u_out = []
    all_inputs = []

    with torch.no_grad():
        for batch in val_loader:
            X, u_out, y, _ = batch
            X = X.to(device)

            # Inference
            final_pred, _ = model(X)

            # Store data for analysis (move to CPU to save GPU memory)
            all_preds.append(final_pred.cpu().numpy())
            all_targets.append(y.numpy())
            all_u_out.append(u_out.numpy())
            all_inputs.append(X.cpu().numpy())

    # Concatenate
    preds_flat = np.concatenate(all_preds).flatten()
    targets_flat = np.concatenate(all_targets).flatten()
    u_out_flat = np.concatenate(all_u_out).flatten()

    # Calculate Metric: MAE on inspiratory phase (u_out == 0)
    # Mask is 1 where u_out is 0
    mask = 1.0 - u_out_flat

    abs_error = np.abs(preds_flat - targets_flat)
    masked_error = abs_error * mask

    # Avoid division by zero
    if mask.sum() > 0:
        final_metric = masked_error.sum() / mask.sum()
    else:
        final_metric = float("inf")

    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    print("\n--- Failure Analysis (Correlation with Error Magnitude) ---")

    # Flatten inputs: (N_breaths * 80, N_features)
    inputs_flat = np.concatenate(all_inputs).reshape(-1, input_dim)

    # Get feature names
    feature_names = get_feature_names()

    # Ensure dimensions match (sanity check)
    if len(feature_names) != input_dim:
        print(
            f"Warning: Feature name count ({len(feature_names)}) != Input dim ({input_dim}). Using generic names."
        )
        feature_names = [f"feat_{i}" for i in range(input_dim)]

    # Calculate correlation between feature values and absolute error
    # We use the full validation set for this analysis
    error_series = pd.Series(abs_error, name="abs_error")

    correlations = {}
    for i, feat_name in enumerate(feature_names):
        feat_vals = inputs_flat[:, i]
        corr = np.corrcoef(feat_vals, abs_error)[0, 1]
        correlations[feat_name] = corr

    # Sort and print
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    print("Top 10 Features correlated with Error Magnitude:")
    for name, corr in sorted_corr[:10]:
        print(f"  {name}: {corr:.4f}")

    # 5. Submission
    THRESHOLD = 0.2164510190486908

    if final_metric < THRESHOLD:
        print(f"\nMetric {final_metric:.6f} < {THRESHOLD}. Generating submission...")

        # Load Test Data
        test_dataset = VentilatorDataset(split="test", load_cached_data=True)
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        test_preds = []
        test_ids = []

        with torch.no_grad():
            for batch in test_loader:
                X, u_out, _, ids = batch
                X = X.to(device)

                # Inference
                pred, _ = model(X)

                test_preds.append(pred.cpu().numpy().flatten())
                test_ids.append(ids.numpy().flatten())

        # Create Submission DataFrame
        submission_df = pd.DataFrame(
            {"id": np.concatenate(test_ids), "pressure": np.concatenate(test_preds)}
        )

        # Sort by ID to be safe (though dataset should be ordered)
        submission_df.sort_values("id", inplace=True)

        # Save
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        submission_df.to_csv(Config.SUBMISSION_FILE, index=False)
        print(f"Submission saved to {Config.SUBMISSION_FILE}")

    else:
        print(
            f"\nMetric {final_metric:.6f} >= {THRESHOLD}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
