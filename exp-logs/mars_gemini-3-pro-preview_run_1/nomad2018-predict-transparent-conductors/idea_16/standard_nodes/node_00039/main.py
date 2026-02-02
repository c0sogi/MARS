import torch
import numpy as np
import pandas as pd
import os
import sys

# Import from the provided library
from library import config
from library import trainer
from library import dataset
from library import model


def main():
    # Set up execution parameters
    # Reducing epochs to 100 to ensure fast baseline execution within time limits
    # while maintaining enough iterations for convergence.
    n_epochs = 100

    print(f"Starting pipeline...")
    print(f"Training for {n_epochs} epochs...")

    # 1. Run Training
    # This will train the HCPDS model and save the best checkpoint.
    trainer.run_training(epochs=n_epochs, batch_size=config.BATCH_SIZE)

    # 2. Validation & Metric Calculation
    print("\nPerforming Validation Assessment...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load validation data
    val_loader = dataset.get_dataloader(
        "val", batch_size=config.BATCH_SIZE, shuffle=False, load_cached_data=True
    )

    # Load best model
    if not os.path.exists(config.MODEL_CHECKPOINT):
        print("Error: Model checkpoint not found.")
        sys.exit(1)

    net = model.HCPDS().to(device)
    net.load_state_dict(torch.load(config.MODEL_CHECKPOINT, map_location=device))
    net.eval()

    all_preds_log = []
    all_targets_log = []
    all_global_feats = []

    # Inference loop (no gradients)
    with torch.no_grad():
        for batch in val_loader:
            atomic_feat = batch["atomic_features"].to(device)
            global_feat = batch["global_features"].to(device)
            mask = batch["mask"].to(device)
            targets = batch["target"].to(device)

            outputs = net(atomic_feat, global_feat, mask)

            all_preds_log.append(outputs.cpu().numpy())
            all_targets_log.append(targets.cpu().numpy())
            # Keep global features for failure analysis
            all_global_feats.append(global_feat.cpu().numpy())

    # Concatenate results
    all_preds_log = np.concatenate(all_preds_log, axis=0)
    all_targets_log = np.concatenate(all_targets_log, axis=0)
    all_global_feats = np.concatenate(all_global_feats, axis=0)

    # Compute Column-wise RMSLE
    # Since targets and preds are already log1p transformed, RMSLE is simply RMSE of these values.
    # MSE per column
    mse_per_col = np.mean((all_preds_log - all_targets_log) ** 2, axis=0)
    # RMSE per column
    rmsle_per_col = np.sqrt(mse_per_col)
    # Mean of column-wise RMSLE
    final_metric = np.mean(rmsle_per_col)

    print(f"Final Validation Metric: {final_metric}")

    # 3. Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Calculate error magnitude per sample (Mean Absolute Error in log space)
    # We average the error across the two targets for a single error score per sample
    sample_errors = np.mean(np.abs(all_preds_log - all_targets_log), axis=1)

    # Define feature names corresponding to global_features construction in features.py
    # [Lens(3), Angs(3), Vol(1), Dens(1), Stoich(3), N_Atoms(1)]
    feature_names = [
        "Lattice_Len_A",
        "Lattice_Len_B",
        "Lattice_Len_C",
        "Lattice_Ang_Alpha",
        "Lattice_Ang_Beta",
        "Lattice_Ang_Gamma",
        "Cell_Volume",
        "Atomic_Density",
        "Stoich_Al",
        "Stoich_Ga",
        "Stoich_In",
        "Total_Atoms",
    ]

    # Compute correlations
    if all_global_feats.shape[1] == len(feature_names):
        print("Correlation between Input Features and Error Magnitude:")
        correlations = []
        for i, name in enumerate(feature_names):
            feat_values = all_global_feats[:, i]
            # Handle potential constant features (std=0) which cause NaN correlation
            if np.std(feat_values) > 1e-9:
                corr = np.corrcoef(feat_values, sample_errors)[0, 1]
                correlations.append((name, corr))
            else:
                correlations.append((name, 0.0))

        # Sort by absolute correlation strength
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)

        for name, corr in correlations:
            print(f"  {name:<20}: {corr:.4f}")
    else:
        print("Feature dimension mismatch. Skipping detailed feature correlation.")

    # 4. Submission Generation
    THRESHOLD = 0.05479004207787702
    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric}) is below threshold ({THRESHOLD}). Generating submission..."
        )
        trainer.generate_submission(batch_size=config.BATCH_SIZE, load_cached_data=True)
    else:
        print(
            f"\nMetric ({final_metric}) is NOT below threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
