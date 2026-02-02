import os
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr

# Import from the provided library files
from library.config import Config
from library.data import get_loaders
from library.model import RAGLUNet
from library.train import Trainer
from library.utils import set_seed, compute_metrics


def main():
    # -------------------------------------------------------------------------
    # 1. Setup & Configuration
    # -------------------------------------------------------------------------
    # Override max epochs for a fast baseline execution as required
    Config.MAX_EPOCHS = 50

    # Setup directories and seeds
    Config.setup()
    set_seed(Config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    print("Loading and processing data...")
    # load_cached_data=True allows utilizing preprocessed .npz files if they exist
    train_loader, val_loader, test_loader, scaler = get_loaders(
        load_cached_data=True, batch_size=Config.BATCH_SIZE
    )

    # -------------------------------------------------------------------------
    # 3. Model Initialization
    # -------------------------------------------------------------------------
    print("Initializing RA-GLU-Net model...")
    model = RAGLUNet(config=Config)

    # -------------------------------------------------------------------------
    # 4. Training
    # -------------------------------------------------------------------------
    print("Starting training...")
    trainer = Trainer(model, scaler, device)

    # Explicitly pass max_epochs to ensure the override is respected
    trainer.fit(train_loader, val_loader, max_epochs=Config.MAX_EPOCHS)

    # -------------------------------------------------------------------------
    # 5. Validation Assessment
    # -------------------------------------------------------------------------
    print("Performing validation assessment...")

    # Generate predictions on the validation set using the best saved model
    # Trainer.predict loads the best checkpoint automatically
    val_preds = trainer.predict(val_loader)

    # Extract ground truth targets from the validation loader
    val_targets_list = []
    # Note: val_loader is not shuffled, so order is preserved
    for batch in val_loader:
        val_targets_list.append(batch.y.numpy())

    # Concatenate and inverse transform targets to original scale
    val_targets_scaled = np.concatenate(val_targets_list, axis=0)
    val_targets = scaler.inverse_transform(val_targets_scaled)

    # Compute the official metric (Column-wise RMSLE)
    final_metric = compute_metrics(val_preds, val_targets)

    # Print the metric in the required format
    print(f"Final Validation Metric: {final_metric}")

    # -------------------------------------------------------------------------
    # 6. Failure Analysis
    # -------------------------------------------------------------------------
    print("\nPerforming failure analysis...")

    # Calculate error magnitude per sample
    # Using the same logic as RMSLE: error = sqrt(mean((log(p+1) - log(t+1))^2))
    vp = np.maximum(val_preds, 0)
    vt = np.maximum(val_targets, 0)
    log_diff_sq = (np.log1p(vp) - np.log1p(vt)) ** 2
    sample_msle = np.mean(log_diff_sq, axis=1)
    sample_error = np.sqrt(sample_msle)

    # Load validation metadata to correlate error with features
    val_meta = pd.read_csv(Config.VAL_METADATA_PATH)

    if len(val_meta) != len(sample_error):
        print(
            f"Warning: Metadata length ({len(val_meta)}) does not match prediction length ({len(sample_error)})"
        )
    else:
        features_to_check = [
            "number_of_total_atoms",
            "lattice_vector_1_ang",
            "lattice_vector_2_ang",
            "lattice_vector_3_ang",
            "lattice_angle_alpha_degree",
            "lattice_angle_beta_degree",
            "lattice_angle_gamma_degree",
            "percent_atom_al",
            "percent_atom_ga",
            "percent_atom_in",
        ]

        print(f"{'Feature':<30} {'Correlation':<15} {'P-value':<10}")
        print("-" * 60)

        for feat in features_to_check:
            if feat in val_meta.columns:
                feat_vals = val_meta[feat].values
                # Handle potential NaNs
                valid_mask = ~np.isnan(feat_vals)
                if np.sum(valid_mask) > 1:
                    corr, p_val = pearsonr(
                        feat_vals[valid_mask], sample_error[valid_mask]
                    )
                    print(f"{feat:<30} {corr:+.4f}          {p_val:.4f}")

    # -------------------------------------------------------------------------
    # 7. Submission Generation
    # -------------------------------------------------------------------------
    threshold = 0.049412816762924194

    if final_metric < threshold:
        print(
            f"\nValidation metric ({final_metric}) is better than threshold ({threshold}). Generating submission..."
        )

        # Generate predictions on the test set
        test_preds = trainer.predict(test_loader)

        # Load test metadata to get IDs
        test_meta = pd.read_csv(Config.TEST_METADATA_PATH)

        if len(test_preds) != len(test_meta):
            print(
                f"Error: Number of predictions ({len(test_preds)}) does not match test metadata ({len(test_meta)})"
            )
        else:
            # Create submission DataFrame
            submission_df = pd.DataFrame(
                {
                    "id": test_meta["id"],
                    "formation_energy_ev_natom": test_preds[:, 0],
                    "bandgap_energy_ev": test_preds[:, 1],
                }
            )

            # Save submission
            os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
            submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
            print(f"Submission saved to {Config.SUBMISSION_PATH}")
            print(submission_df.head())

    else:
        print(
            f"\nValidation metric ({final_metric}) is NOT better than threshold ({threshold}). Skipping submission."
        )


if __name__ == "__main__":
    main()
