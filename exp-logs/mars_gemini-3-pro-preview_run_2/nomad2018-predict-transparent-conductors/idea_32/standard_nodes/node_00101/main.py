import os
import torch
import numpy as np
import pandas as pd
import sys

# Import from the provided library
from library.config import Config
from library.train import Trainer
from library.data import get_dataloaders
from library.utils import set_seed, rmsle
from library.model import MS_RA_CGN


def main():
    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    # Override Config for a fast baseline execution within time limits
    Config.MAX_EPOCHS = 25  # Reduced epochs for speed
    Config.NUM_WORKERS = 2  # Optimize for available vCPUs

    # Set seeds for reproducibility
    set_seed(Config.SEED)

    print(f"Configuration:")
    print(f"  Device: {Config.DEVICE}")
    print(f"  Max Epochs: {Config.MAX_EPOCHS}")
    print(f"  Batch Size: {Config.BATCH_SIZE}")

    # -------------------------------------------------------------------------
    # 2. Model Training
    # -------------------------------------------------------------------------
    # Initialize Trainer (builds model, optimizer, scaler)
    trainer = Trainer()

    # Execute training loop
    # This will train the model, save checkpoints, and cache preprocessed data
    print("\n--- Starting Training ---")
    trainer.run(load_cached_data=True)

    # -------------------------------------------------------------------------
    # 3. Validation & Metric Assessment
    # -------------------------------------------------------------------------
    print("\n--- Starting Validation Assessment ---")

    # Load the best model checkpoint
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    if not os.path.exists(best_model_path):
        print("Error: Best model checkpoint not found.")
        sys.exit(1)

    print(f"Loading best model from {best_model_path}...")
    trainer.model.load_state_dict(
        torch.load(best_model_path, map_location=Config.DEVICE)
    )
    trainer.model.eval()

    # Get dataloaders (re-using cached data)
    _, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # Run inference on validation set
    val_preds_list = []
    val_targets_list = []
    val_ids_list = []

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(Config.DEVICE)

            # Forward pass
            out_scaled = trainer.model(batch)

            # Inverse transform to original scale
            out = trainer.scaler.inverse_transform(out_scaled)

            val_preds_list.append(out.cpu().numpy())
            val_targets_list.append(batch.y.cpu().numpy())
            val_ids_list.extend(batch.id.cpu().numpy())

    # Concatenate results
    val_preds = np.concatenate(val_preds_list, axis=0)
    val_targets = np.concatenate(val_targets_list, axis=0)

    # Clip negative predictions to 0 for RMSLE calculation safety
    val_preds = np.maximum(val_preds, 0)
    val_targets = np.maximum(val_targets, 0)

    # Compute Final Validation Metric
    final_metric = rmsle(val_targets, val_preds)
    print(f"Final Validation Metric: {final_metric}")

    # -------------------------------------------------------------------------
    # 4. Failure Analysis
    # -------------------------------------------------------------------------
    print("\n--- Performing Failure Analysis ---")

    # Calculate error magnitude per sample (Mean Absolute Error across targets)
    # shape: (n_samples, n_targets) -> (n_samples,)
    errors = np.mean(np.abs(val_targets - val_preds), axis=1)

    # Load validation metadata to get features
    val_meta_df = pd.read_csv(Config.VAL_METADATA_PATH)

    # Create a DataFrame for analysis
    # Ensure alignment by merging on ID
    error_df = pd.DataFrame({"id": val_ids_list, "error_magnitude": errors})
    analysis_df = val_meta_df.merge(error_df, on="id")

    # Select numerical features for correlation analysis
    feature_cols = [
        "number_of_total_atoms",
        "percent_atom_al",
        "percent_atom_ga",
        "percent_atom_in",
        "lattice_vector_1_ang",
        "lattice_vector_2_ang",
        "lattice_vector_3_ang",
        "lattice_angle_alpha_degree",
        "lattice_angle_beta_degree",
        "lattice_angle_gamma_degree",
    ]

    # Compute correlations
    correlations = (
        analysis_df[feature_cols]
        .corrwith(analysis_df["error_magnitude"])
        .sort_values(ascending=False)
    )

    print("Correlation between Error Magnitude and Input Features:")
    print(correlations)

    # -------------------------------------------------------------------------
    # 5. Submission Generation
    # -------------------------------------------------------------------------
    THRESHOLD = 0.049412816762924194

    if final_metric < THRESHOLD:
        print(f"\nMetric {final_metric} < {THRESHOLD}. Generating submission...")

        test_preds_list = []
        test_ids_list = []

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(Config.DEVICE)

                # Forward pass
                out_scaled = trainer.model(batch)

                # Inverse transform
                out = trainer.scaler.inverse_transform(out_scaled)

                test_preds_list.append(out.cpu().numpy())
                test_ids_list.extend(batch.id.cpu().numpy())

        test_preds = np.concatenate(test_preds_list, axis=0)
        # Clip negative predictions
        test_preds = np.maximum(test_preds, 0)

        # Create submission DataFrame
        submission_df = pd.DataFrame(
            {
                "id": test_ids_list,
                "formation_energy_ev_natom": test_preds[:, 0],
                "bandgap_energy_ev": test_preds[:, 1],
            }
        )

        # Sort by ID to ensure consistent order
        submission_df.sort_values("id", inplace=True)

        # Save
        submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
        submission_df.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")

    else:
        print(f"\nMetric {final_metric} >= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
