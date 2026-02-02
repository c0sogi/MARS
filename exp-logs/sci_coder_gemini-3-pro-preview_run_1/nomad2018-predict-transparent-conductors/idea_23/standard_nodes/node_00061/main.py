import torch
import numpy as np
import pandas as pd
import os
import sys

# Import from provided libraries
import library.config as config
import library.training as training
import library.data_loader as data_loader
import library.model as model_lib


def main():
    # Set random seeds for reproducibility
    config.set_seed()

    print("Starting Chemically-Contextualized Wide Deep Sets (CC-WDS) Pipeline...")

    # 1. Train the model
    # We use the default configuration (200 epochs) which is sufficient for this dataset size.
    # The training function handles data loading, model initialization, and the training loop.
    trained_model, test_loader = training.train_model(
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        max_samples=None,  # Use full dataset
    )

    # 2. Validation Assessment & Failure Analysis
    print("\nRunning Validation Assessment...")

    # Get validation loader (index 1 from get_dataloaders)
    _, val_loader, _ = data_loader.get_dataloaders(
        batch_size=config.BATCH_SIZE, load_cached_data=True
    )

    trained_model.eval()
    all_preds = []
    all_targets = []
    all_global_feats = []

    # Inference on validation set (no gradients needed)
    with torch.no_grad():
        for batch in val_loader:
            atomic, indices, global_f, targets, ids = batch

            # Move to device
            atomic = atomic.to(config.DEVICE)
            indices = indices.to(config.DEVICE)
            global_f = global_f.to(config.DEVICE)

            # Forward pass
            outputs = trained_model(atomic, indices, global_f)

            # Collect results (keep on CPU for numpy)
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.numpy())
            all_global_feats.append(global_f.cpu().numpy())

    # Concatenate batches
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_global_feats = np.concatenate(all_global_feats, axis=0)

    # Compute Metric: Column-wise Root Mean Squared Logarithmic Error
    # Note: The model outputs and targets are already log1p transformed.
    # Therefore, RMSE on these values is equivalent to RMSLE on original values.
    mse_per_col = np.mean((all_preds - all_targets) ** 2, axis=0)
    rmsle_per_col = np.sqrt(mse_per_col)
    final_metric = np.mean(rmsle_per_col)

    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    # Calculate error magnitude per sample (mean absolute error across the two targets)
    errors = np.mean(np.abs(all_preds - all_targets), axis=1)

    # Map global feature indices to names based on feature_extractor.py
    # 0-2: Lattice lengths, 3-5: Lattice angles, 6: Volume, 7: Density,
    # 8-10: Stoichiometry (Al, Ga, In), 11: Total Atoms
    feature_names = [
        "lattice_len_1",
        "lattice_len_2",
        "lattice_len_3",
        "lattice_ang_a",
        "lattice_ang_b",
        "lattice_ang_g",
        "volume",
        "atomic_density",
        "stoich_Al",
        "stoich_Ga",
        "stoich_In",
        "total_atoms",
    ]

    analysis_df = pd.DataFrame(all_global_feats, columns=feature_names)
    analysis_df["error_magnitude"] = errors

    print("\nFailure Analysis (Correlation with Error Magnitude):")
    # Compute correlation between features and error
    correlations = analysis_df.corr()["error_magnitude"].drop("error_magnitude")
    # Sort by absolute correlation
    sorted_corrs = correlations.abs().sort_values(ascending=False)

    for feat in sorted_corrs.index:
        print(f"  {feat:<20}: {correlations[feat]:.4f}")

    # 3. Submission Generation
    THRESHOLD = 0.05479004207787702

    if final_metric < THRESHOLD:
        print(f"\nValidation metric ({final_metric}) meets threshold ({THRESHOLD}).")
        print("Generating submission file...")
        training.generate_submission(
            trained_model, test_loader, config.DEVICE, config.SUBMISSION_PATH
        )
    else:
        print(
            f"\nValidation metric ({final_metric}) does NOT meet threshold ({THRESHOLD})."
        )
        print("Submission skipped.")


if __name__ == "__main__":
    main()
