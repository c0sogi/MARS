import sys
import os
import torch
import numpy as np
import pandas as pd

from library import config
from library import data
from library import model as model_lib
from library import train as train_lib
from library import utils


def get_feature_names():
    """
    Helper to retrieve feature names by processing a minimal chunk of data.
    This ensures failure analysis reports meaningful feature names.
    """
    try:
        # Load a tiny sample
        df = pd.read_csv(config.TRAIN_PATH, nrows=200)
        # Apply engineering
        df = data.engineer_features(df)
        # Identify feature columns as done in data.py
        exclude_cols = [
            config.ID_COL,
            config.BREATH_ID_COL,
            config.TARGET_COL,
            config.U_OUT_COL,
        ]
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        return feature_cols
    except Exception as e:
        print(f"Warning: Could not retrieve feature names ({e}). Using indices.")
        return None


def main():
    # 1. Setup
    utils.seed_everything(config.SEED)
    device = config.DEVICE

    # 2. Training
    # We use a reduced number of epochs for a fast baseline execution.
    FAST_EPOCHS = 6

    print(f"Starting fast baseline training for {FAST_EPOCHS} epochs...")
    trained_model = train_lib.run_training(
        epochs=FAST_EPOCHS, batch_size=config.BATCH_SIZE, load_cached_data=True
    )

    # 3. Validation & Metric Calculation
    print("Performing validation...")
    # Retrieve dataloaders (leveraging cache)
    _, val_loader, test_loader = data.get_dataloaders(
        batch_size=config.BATCH_SIZE, load_cached_data=True
    )

    trained_model.eval()
    val_preds = []
    val_targets = []
    val_uouts = []
    val_inputs = []

    with torch.no_grad():
        for x, u_out, y in val_loader:
            x = x.to(device)
            u_out = u_out.to(device)
            y = y.to(device)

            # Forward pass
            pred, _ = trained_model(x, u_out)

            val_preds.append(pred.cpu().numpy())
            val_targets.append(y.cpu().numpy())
            val_uouts.append(u_out.cpu().numpy())
            val_inputs.append(x.cpu().numpy())

    val_preds = np.concatenate(val_preds)
    val_targets = np.concatenate(val_targets)
    val_uouts = np.concatenate(val_uouts)
    val_inputs = np.concatenate(val_inputs)

    # Compute MAE using the library utility
    mae = utils.compute_mae(val_preds, val_targets, val_uouts)
    print(f"Final Validation Metric: {mae}")

    # 4. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Flatten arrays for correlation analysis
    preds_flat = val_preds.flatten()
    targets_flat = val_targets.flatten()
    uouts_flat = val_uouts.flatten()
    # Reshape inputs to (Total_Steps, N_Features)
    inputs_flat = val_inputs.reshape(-1, val_inputs.shape[-1])

    # Mask for inspiratory phase (u_out == 0)
    insp_mask = uouts_flat < 0.5

    if np.sum(insp_mask) > 0:
        # Calculate absolute error
        errors = np.abs(preds_flat[insp_mask] - targets_flat[insp_mask])
        feat_subset = inputs_flat[insp_mask]

        # Create DataFrame for correlation
        feature_names = get_feature_names()
        if feature_names is None or len(feature_names) != feat_subset.shape[1]:
            feature_names = [f"Feature_{i}" for i in range(feat_subset.shape[1])]

        analysis_df = pd.DataFrame(feat_subset, columns=feature_names)
        analysis_df["error_magnitude"] = errors

        # Compute correlations
        correlations = analysis_df.corr()["error_magnitude"].drop("error_magnitude")
        top_correlations = correlations.abs().sort_values(ascending=False).head(5)

        print("Correlation between Error Magnitude and Input Features (Top 5):")
        print(top_correlations)
    else:
        print("No inspiratory phase data found for failure analysis.")

    # 5. Submission
    THRESHOLD = 0.2164510190486908
    if mae < THRESHOLD:
        print(
            f"\nValidation metric {mae} is better than threshold {THRESHOLD}. Generating submission..."
        )
        # Pass the existing test_loader to avoid reloading
        model_lib.predict(dataloaders=(None, None, test_loader))
    else:
        print(
            f"\nValidation metric {mae} did not meet threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
