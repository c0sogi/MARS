import torch
import numpy as np
import pandas as pd
import os
import sys

# 1. Configuration Setup
from library.config import Config

# Modify Config for fast baseline execution
Config.NUM_EPOCHS = 100
Config.DEBUG_SAMPLE_SIZE = None  # Use full dataset

# 2. Import Library Modules
from library.data import get_dataloaders
from library.model import CR_WDS
from library.train import Trainer, set_seed
from library.utils import inverse_log_transform, compute_rmsle


def main():
    # Set reproducibility
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    print(f"Running on device: {device}")

    # 3. Data Loading
    # Uses cached data if available to save time
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        load_cached_data=True,
        debug_size=Config.DEBUG_SAMPLE_SIZE,
    )

    # 4. Model Initialization
    model = CR_WDS()

    # 5. Training
    trainer = Trainer(model, device)
    trainer.fit(
        train_loader,
        val_loader,
        epochs=Config.NUM_EPOCHS,
        patience=Config.EARLY_STOPPING_PATIENCE,
    )

    # 6. Validation Assessment
    print("\n--- Validation Assessment ---")
    # Load best model weights
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    all_preds = []
    all_targets = []
    all_global_feats = []

    # Inference loop without gradient calculation
    with torch.no_grad():
        for batch in val_loader:
            atomic_features = batch["atomic_features"].to(device)
            mask = batch["mask"].to(device)
            global_features = batch["global_features"].to(device)
            targets = batch["targets"].to(device)

            batch_dict = {
                "atomic_features": atomic_features,
                "mask": mask,
                "global_features": global_features,
            }
            outputs = model(batch_dict)

            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_global_feats.append(global_features.cpu().numpy())

    # Concatenate batches
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_global_feats = np.concatenate(all_global_feats, axis=0)

    # Inverse transform targets (log1p -> expm1)
    pred_orig = inverse_log_transform(all_preds)
    target_orig = inverse_log_transform(all_targets)

    # Clip negative predictions
    pred_orig = np.maximum(pred_orig, 0.0)

    # Compute Final Metric
    final_metric = compute_rmsle(target_orig, pred_orig)
    print(f"Final Validation Metric: {final_metric}")

    # 7. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate Mean Absolute Error in log space per sample as a proxy for error magnitude
    # We use log space error because the metric is RMSLE
    log_pred = np.log1p(pred_orig)
    log_true = np.log1p(target_orig)
    error_magnitude = np.mean(np.abs(log_pred - log_true), axis=1)

    feature_names = [
        "Lattice_a",
        "Lattice_b",
        "Lattice_c",
        "Angle_alpha",
        "Angle_beta",
        "Angle_gamma",
        "Cell_Volume",
        "Atomic_Density",
        "Stoich_Al",
        "Stoich_Ga",
        "Stoich_In",
        "Total_Atoms",
    ]

    print("Correlation between Error Magnitude and Global Features:")
    for i, name in enumerate(feature_names):
        if i < all_global_feats.shape[1]:
            feat_vals = all_global_feats[:, i]
            # Avoid correlation with constant features
            if np.std(feat_vals) > 1e-6:
                corr = np.corrcoef(feat_vals, error_magnitude)[0, 1]
                print(f"  {name:<15}: {corr:.4f}")
            else:
                print(f"  {name:<15}: NaN (Constant)")

    # 8. Submission Generation
    threshold = 0.05479004207787702
    if final_metric < threshold:
        print(f"\nMetric {final_metric} < {threshold}. Generating submission...")

        # Predict on Test Set
        ids, test_preds = trainer.predict(test_loader)

        # Create DataFrame
        submission_df = pd.DataFrame(
            {
                "id": ids,
                "formation_energy_ev_natom": test_preds[:, 0],
                "bandgap_energy_ev": test_preds[:, 1],
            }
        )

        # Sort by ID
        submission_df.sort_values("id", inplace=True)

        # Save
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(f"\nMetric {final_metric} >= {threshold}. Submission skipped.")


if __name__ == "__main__":
    main()
