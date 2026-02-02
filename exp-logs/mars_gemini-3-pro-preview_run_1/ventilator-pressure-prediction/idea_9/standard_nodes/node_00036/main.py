import os
import sys
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import set_seed, get_device
from library.data_loader import get_data_loaders
from library.train import Trainer
from library.model import PhysicsInjectedNet
from library.features import get_datasets

# ==========================================
# 1. Configuration Overrides for Fast Baseline
# ==========================================
# Reduce epochs to ensure execution completes within the time limit (Fast Baseline)
Config.EPOCHS = 15
# Ensure we use the full dataset for valid metric calculation, so DEBUG must be False
Config.DEBUG = False
# Ensure working directory is consistent
os.makedirs(Config.WORKING_DIR, exist_ok=True)


def main():
    # ==========================================
    # 2. Setup & Data Loading
    # ==========================================
    set_seed(Config.SEED)
    device = get_device()
    print(f"Running on device: {device}")

    print("Loading data and initializing DataLoaders...")
    # load_cached=True attempts to use existing .npy files in ./working/idea_9
    train_loader, val_loader, test_loader = get_data_loaders(load_cached=True)

    # ==========================================
    # 3. Training
    # ==========================================
    print("Initializing Trainer...")
    trainer = Trainer(train_loader, val_loader)

    print(f"Starting training for {Config.EPOCHS} epochs...")
    # Fit the model (OneCycleLR will adapt to the new Config.EPOCHS)
    trainer.fit(patience=5)

    # ==========================================
    # 4. Validation & Metric Calculation
    # ==========================================
    print("Loading best model for validation...")
    model = PhysicsInjectedNet().to(device)
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.eval()

    print("Running inference on full validation set...")
    val_preds = []
    val_targets = []
    val_u_out = []
    val_inputs = []

    # Identify index of u_out for masking
    u_out_idx = Config.FEATURE_COLS.index("u_out")

    with torch.no_grad():
        for X, y in val_loader:
            X = X.to(device)
            y = y.to(device)

            # Forward pass
            preds = model(X)

            # Collect data for metric and analysis
            val_preds.append(preds.cpu().numpy())
            val_targets.append(y.cpu().numpy())
            val_u_out.append(X[:, :, u_out_idx].cpu().numpy())
            val_inputs.append(X.cpu().numpy())

    # Concatenate all batches
    val_preds = np.concatenate(val_preds)  # (N, 80)
    val_targets = np.concatenate(val_targets)  # (N, 80)
    val_u_out = np.concatenate(val_u_out)  # (N, 80)
    val_inputs = np.concatenate(val_inputs)  # (N, 80, F)

    # Compute Masked MAE
    # Mask: 1 for inspiratory (u_out == 0), 0 for expiratory
    mask = 1 - val_u_out
    abs_error = np.abs(val_preds - val_targets)
    masked_error = abs_error * mask

    # Avoid division by zero
    final_metric = masked_error.sum() / (mask.sum() + 1e-8)

    print(f"Final Validation Metric: {final_metric}")

    # ==========================================
    # 5. Failure Analysis
    # ==========================================
    print("\n=== Failure Analysis ===")
    # Flatten arrays to element-wise level, filtering only inspiratory phase
    mask_bool = mask.astype(bool).flatten()
    errors_flat = abs_error.flatten()[mask_bool]

    # Reshape inputs to (Total_Steps, Features) and filter
    n_features = len(Config.FEATURE_COLS)
    inputs_flat = val_inputs.reshape(-1, n_features)
    inputs_masked = inputs_flat[mask_bool]

    correlations = {}
    print("Correlation of Error Magnitude with Input Features:")
    for i, feature_name in enumerate(Config.FEATURE_COLS):
        feat_values = inputs_masked[:, i]

        # Calculate Pearson correlation
        if np.std(feat_values) > 1e-9:
            corr = np.corrcoef(errors_flat, feat_values)[0, 1]
            correlations[feature_name] = corr
        else:
            correlations[feature_name] = 0.0

    # Print sorted correlations
    for name, corr in sorted(
        correlations.items(), key=lambda x: abs(x[1]), reverse=True
    ):
        print(f"  {name}: {corr:.4f}")

    # ==========================================
    # 6. Submission Generation
    # ==========================================
    THRESHOLD = 0.3096454441547394

    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric:.6f}) < Threshold ({THRESHOLD}). Generating submission..."
        )

        test_preds = []
        with torch.no_grad():
            for X in test_loader:
                X = X.to(device)
                preds = model(X)
                test_preds.append(preds.cpu().numpy())

        # Flatten predictions to match sample_submission format
        test_preds_flat = np.concatenate(test_preds).flatten()

        # Retrieve test_ids from processed datasets
        datasets = get_datasets(load_cached=True)
        test_ids = datasets["test_ids"]

        # Ensure alignment
        if len(test_preds_flat) != len(test_ids):
            print(
                f"Warning: Prediction length {len(test_preds_flat)} != ID length {len(test_ids)}"
            )
            # Truncate or pad if necessary (though strictly they should match)
            min_len = min(len(test_preds_flat), len(test_ids))
            test_preds_flat = test_preds_flat[:min_len]
            test_ids = test_ids[:min_len]

        # Create DataFrame
        submission_df = pd.DataFrame({"id": test_ids, "pressure": test_preds_flat})

        # Save
        sub_dir = "./submission"
        os.makedirs(sub_dir, exist_ok=True)
        sub_path = os.path.join(sub_dir, "submission.csv")
        submission_df.to_csv(sub_path, index=False)
        print(f"Submission saved to {sub_path}")

    else:
        print(
            f"\nMetric ({final_metric:.6f}) >= Threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
