import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import warnings

# Import from the provided library
from library.config import DEVICE, MODEL_SAVE_PATH, SUBMISSION_FILE_PATH, EPOCHS
from library.utils import seed_everything, load_checkpoint, save_submission
from library.dataset import get_dataloaders, TargetTransformer
from library.model import CSNWDS
from library.engine import Trainer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    # 1. Setup
    print("Initializing CSN-WDS Pipeline...")
    seed_everything(42)

    # 2. Data Loading
    # Uses the library function to load metadata, process geometry (with caching),
    # fit/load scalers, and return DataLoaders.
    print("Loading and processing data...")
    train_loader, val_loader, test_loader, scaler = get_dataloaders(
        debug_sample_size=None
    )

    # 3. Model Initialization
    print("Initializing model...")
    model = CSNWDS()

    # 4. Training
    print("Starting training...")
    trainer = Trainer(model, train_loader, val_loader, device=DEVICE)
    trainer.fit(epochs=EPOCHS)

    # 5. Validation & Metric Calculation
    print("\nPerforming final validation...")
    # Load the best model checkpoint saved during training
    model = load_checkpoint(model, MODEL_SAVE_PATH, device=DEVICE)
    model.eval()

    val_preds_list = []
    val_targets_list = []
    val_globals_list = []

    # Inference on validation set (no gradient needed)
    with torch.no_grad():
        for batch in val_loader:
            # Unpack batch: (atomic, batch_idx, global, target, ids)
            atomic_features = batch[0].to(DEVICE)
            batch_index = batch[1].to(DEVICE)
            global_features = batch[2].to(DEVICE)
            targets = batch[3].to(DEVICE)

            # Forward pass
            outputs = model(atomic_features, batch_index, global_features)

            # Store results on CPU
            val_preds_list.append(outputs.cpu())
            val_targets_list.append(targets.cpu())
            val_globals_list.append(global_features.cpu())

    # Concatenate all batches
    val_preds = torch.cat(val_preds_list, dim=0)
    val_targets = torch.cat(val_targets_list, dim=0)
    val_globals = torch.cat(val_globals_list, dim=0)

    # Calculate Column-wise RMSLE
    # Note: Targets and Preds are already in log(1+y) space due to TargetTransformer in dataset.
    # Therefore, RMSE on these values IS the RMSLE.
    mse_per_col = torch.mean((val_preds - val_targets) ** 2, dim=0)
    rmsle_per_col = torch.sqrt(mse_per_col)

    # The metric is the mean of the column-wise RMSLEs
    final_metric = torch.mean(rmsle_per_col).item()

    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("\nPerforming failure analysis...")
    # Calculate Mean Absolute Error per sample in log space as a proxy for "error magnitude"
    # shape: (N,)
    errors = torch.mean(torch.abs(val_preds - val_targets), dim=1).numpy()

    # Global feature names corresponding to the 12 dimensions in process_file
    feature_names = [
        "lattice_len_a",
        "lattice_len_b",
        "lattice_len_c",
        "lattice_angle_alpha",
        "lattice_angle_beta",
        "lattice_angle_gamma",
        "cell_volume",
        "atomic_density",
        "total_atoms",
        "frac_Al",
        "frac_Ga",
        "frac_In",
    ]

    # Calculate correlation between each global feature and the error
    global_feats_np = val_globals.numpy()
    correlations = {}

    for i, name in enumerate(feature_names):
        if i < global_feats_np.shape[1]:
            feat_values = global_feats_np[:, i]
            # Handle potential constant features (std=0) which produce nan correlation
            if np.std(feat_values) > 1e-9:
                corr = np.corrcoef(feat_values, errors)[0, 1]
                correlations[name] = corr
            else:
                correlations[name] = 0.0

    # Print correlations sorted by magnitude
    print("Correlation between Input Features and Error Magnitude:")
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    for name, corr in sorted_corr:
        print(f"  {name:<20}: {corr:.4f}")

    # 7. Submission Generation
    # Threshold defined in requirements
    THRESHOLD = 0.05479004207787702

    if final_metric < THRESHOLD:
        print(
            f"\nMetric {final_metric} meets threshold ({THRESHOLD}). Generating submission..."
        )

        ids_all = []
        preds_all = []
        target_transformer = TargetTransformer()

        with torch.no_grad():
            for batch in test_loader:
                # Unpack batch
                atomic_features = batch[0].to(DEVICE)
                batch_index = batch[1].to(DEVICE)
                global_features = batch[2].to(DEVICE)
                ids = batch[4]  # IDs are the 5th element

                # Forward pass
                outputs = model(atomic_features, batch_index, global_features)

                # Inverse transform: log(1+y) -> y
                # We must predict original scale values
                preds_original_scale = target_transformer.inverse_transform(outputs)

                ids_all.extend(ids)
                preds_all.append(preds_original_scale.cpu().numpy())

        # Concatenate predictions
        preds_all = np.concatenate(preds_all, axis=0)

        # Save submission using library utility
        # preds_all is (N, 2) -> [formation_energy, bandgap_energy]
        save_submission(ids_all, preds_all[:, 0], preds_all[:, 1], SUBMISSION_FILE_PATH)
        print(f"Submission saved to {SUBMISSION_FILE_PATH}")

    else:
        print(
            f"\nMetric {final_metric} does not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
