import os
import torch
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

from library.config import Config
from library.model import SPRACGN
from library.data import get_dataloaders
from library.train import Trainer
from library.utils import set_seed, compute_metric, StandardScaler


def main():
    # 1. Setup
    print("Setting up environment...")
    # Override Config for fast baseline
    Config.NUM_EPOCHS = 100  # Increased to allow full convergence
    Config.BATCH_SIZE = 48

    set_seed(Config.SEED)
    Config.setup()

    device = Config.DEVICE
    print(f"Using device: {device}")

    # 2. Load Data
    print("Loading data...")
    # Load all data (cached if available)
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Initialize Model and Trainer
    print("Initializing model...")
    model = SPRACGN(config=Config)
    trainer = Trainer(model, device=device)

    # 4. Train
    print("Starting training...")
    # Fit scaler on training data
    trainer.fit_scaler(train_loader)

    # Train the model
    trainer.fit(train_loader, val_loader, epochs=Config.NUM_EPOCHS)

    # 5. Validation Assessment
    print("Performing validation assessment...")
    # Load best model
    if not os.path.exists(Config.MODEL_CHECKPOINT_PATH):
        print("No checkpoint found. Training might have failed.")
        return

    best_model = SPRACGN(config=Config)
    best_model.load_state_dict(
        torch.load(Config.MODEL_CHECKPOINT_PATH, map_location=device)
    )
    best_model.to(device)
    best_model.eval()

    # Load scaler for inverse transform
    scaler = StandardScaler(device=device)
    scaler.load(Config.TARGET_SCALER_PATH)

    val_preds = []
    val_targets = []

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            # Predict
            pred_scaled = best_model(batch)
            pred = scaler.inverse_transform(pred_scaled)

            val_preds.append(pred.cpu().numpy())
            val_targets.append(batch.y.cpu().numpy())

    val_preds = np.concatenate(val_preds, axis=0)
    val_targets = np.concatenate(val_targets, axis=0)

    # Compute Metric
    final_metric = compute_metric(val_targets, val_preds)
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("Performing failure analysis...")
    # Calculate errors (Mean Absolute Error per sample)
    # We look at the average error across the two targets for correlation analysis
    errors = np.mean(np.abs(val_preds - val_targets), axis=1)

    # Load validation metadata to correlate with features
    val_meta_df = pd.read_csv(Config.VAL_METADATA_PATH)

    # Ensure lengths match
    if len(errors) != len(val_meta_df):
        print(
            f"Warning: Mismatch in validation set size. Preds: {len(errors)}, Meta: {len(val_meta_df)}"
        )
    else:
        # Select numerical features for correlation
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

        print("Correlation between MAE and features:")
        for col in feature_cols:
            if col in val_meta_df.columns:
                vals = val_meta_df[col].values
                # Handle potential NaNs or constant values
                if np.std(vals) == 0:
                    corr = 0.0
                else:
                    corr, _ = pearsonr(errors, vals)
                print(f"  {col}: {corr:.4f}")

    # 7. Submission Generation
    # Threshold check
    threshold = 0.049412816762924194
    if final_metric < threshold:
        print(
            f"Validation metric ({final_metric}) is better than threshold ({threshold}). Generating submission..."
        )

        test_preds = []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                pred_scaled = best_model(batch)
                pred = scaler.inverse_transform(pred_scaled)
                test_preds.append(pred.cpu().numpy())

        test_predictions = np.concatenate(test_preds, axis=0)

        # Load test metadata for IDs
        test_meta_df = pd.read_csv(Config.TEST_METADATA_PATH)

        submission_df = pd.DataFrame(
            {
                "id": test_meta_df["id"],
                "formation_energy_ev_natom": test_predictions[:, 0],
                "bandgap_energy_ev": test_predictions[:, 1],
            }
        )

        os.makedirs(os.path.dirname(Config.SUBMISSION_FILE_PATH), exist_ok=True)
        submission_df.to_csv(Config.SUBMISSION_FILE_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_FILE_PATH}")
    else:
        print(
            f"Validation metric ({final_metric}) did not meet threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
