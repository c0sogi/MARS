import os
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import set_seed, compute_column_wise_rmsle
from library.data import get_dataloaders
from library.train import run_training, generate_submission


def evaluate_and_analyze(model, val_loader, device):
    """
    Performs validation inference, computes metrics, and runs failure analysis.
    """
    model.eval()

    all_preds = []
    all_targets = []
    all_global_feats = []

    # Disable gradient calculation for inference
    with torch.no_grad():
        for batch in val_loader:
            # Move data to device
            batch_data = {
                "atomic_features": batch["atomic_features"].to(device),
                "batch_indices": batch["batch_indices"].to(device),
                "global_features": batch["global_features"].to(device),
            }
            targets = batch["targets"].to(device)

            # Forward pass
            outputs = model(batch_data)

            all_preds.append(outputs.cpu())
            all_targets.append(targets.cpu())
            all_global_feats.append(batch["global_features"].cpu())

    # Concatenate all batches
    preds = torch.cat(all_preds, dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy()
    global_feats = torch.cat(all_global_feats, dim=0).numpy()

    # Compute Metric (RMSLE)
    # Since predictions and targets are already in log1p space, RMSE here is RMSLE of original
    rmsle = compute_column_wise_rmsle(preds, targets)
    print(f"Final Validation Metric: {rmsle}")

    # --- Failure Analysis ---
    print("\n" + "=" * 40)
    print(" FAILURE ANALYSIS ")
    print("=" * 40)

    # Calculate error magnitude (Mean Absolute Error in log space per sample)
    # We average the error across the two targets (formation energy and bandgap) for a single scalar error score
    errors = np.mean(np.abs(preds - targets), axis=1)

    # Feature names for global features (indices 0-11)
    # 0-2: Lattice Lengths, 3-5: Lattice Angles, 6: Volume, 7: Density, 8-10: Stoich, 11: Num Atoms
    feature_names = [
        "Lattice_Len_a",
        "Lattice_Len_b",
        "Lattice_Len_c",
        "Lattice_Ang_alpha",
        "Lattice_Ang_beta",
        "Lattice_Ang_gamma",
        "Volume",
        "Density",
        "Stoich_Al",
        "Stoich_Ga",
        "Stoich_In",
        "Num_Atoms",
    ]

    # Create DataFrame for correlation analysis
    df_analysis = pd.DataFrame(global_feats, columns=feature_names)
    df_analysis["Error_Magnitude"] = errors

    # Compute correlations
    correlations = df_analysis.corr()["Error_Magnitude"].drop("Error_Magnitude")

    # Sort by absolute correlation
    sorted_corr = correlations.abs().sort_values(ascending=False)

    print("Correlation between Error Magnitude and Input Features:")
    for feat in sorted_corr.index:
        print(f"  {feat:<20}: {correlations[feat]:.4f}")

    return rmsle


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # 2. Train Model
    # Using 50 epochs for a fast baseline as requested
    print("\nStarting Model Training...")
    model = run_training(epochs=50, load_cached_data=True)

    # 3. Load Validation Data
    print("\nLoading Validation Data...")
    _, val_loader, _ = get_dataloaders(
        batch_size=Config.BATCH_SIZE, load_cached_data=True
    )

    # 4. Evaluate and Analyze
    print("\nRunning Validation and Failure Analysis...")
    val_metric = evaluate_and_analyze(model, val_loader, device)

    # 5. Submission Logic
    # Threshold defined in requirements
    THRESHOLD = 0.05479004207787702

    if val_metric < THRESHOLD:
        print(
            f"\nValidation metric ({val_metric}) meets threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission(model, batch_size=Config.BATCH_SIZE, load_cached_data=True)
    else:
        print(
            f"\nValidation metric ({val_metric}) did NOT meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
