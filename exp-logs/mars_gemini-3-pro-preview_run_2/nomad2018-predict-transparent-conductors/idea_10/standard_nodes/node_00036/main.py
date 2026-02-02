import torch
import numpy as np
import pandas as pd
import os
import sys

# Add current directory to path to ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.engine import run_training, generate_submission, calculate_rmsle
from library.dataset import CrystalGraphDataset, collate_graphs
from library.model import CGCNN
from torch.utils.data import DataLoader


def main():
    # Set seeds for reproducibility to ensure consistent results
    torch.manual_seed(Config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Config.SEED)
    np.random.seed(Config.SEED)

    print(
        "Starting execution of Dual-Stream Geometric-Compositional Network pipeline..."
    )

    # -------------------------------------------------------------------------
    # 1. Model Training
    # -------------------------------------------------------------------------
    # We run training for 50 epochs. This provides a good balance between
    # computational speed (fast baseline) and model convergence for this
    # dataset size (~1.7k samples).
    print("\n--- Phase 1: Training ---")
    global_scaler, target_scaler = run_training(epochs=50)

    # -------------------------------------------------------------------------
    # 2. Validation & Metric Calculation
    # -------------------------------------------------------------------------
    print("\n--- Phase 2: Validation ---")

    # Load the validation dataset using the metadata
    # We reuse the scalers fitted on the training data to ensure consistent normalization
    val_dataset = CrystalGraphDataset(
        metadata_path=Config.VAL_METADATA_PATH,
        cache_prefix="val",
        load_cached_data=True,
        global_scaler=global_scaler,
        target_scaler=target_scaler,
        fit_scalers=False,
    )

    # Create DataLoader for validation
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_graphs,
    )

    # Load the best model checkpoint saved during training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inference Device: {device}")

    model = CGCNN().to(device)
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    if not os.path.exists(checkpoint_path):
        print("Error: Checkpoint not found. Training might have failed.")
        return

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()  # Set model to evaluation mode

    all_preds = []
    all_targets = []

    # Run inference without gradient computation for speed and memory efficiency
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)

            # Forward pass
            preds_scaled = model(batch)

            # Inverse transform predictions and targets to original scale (eV)
            preds_real = target_scaler.inverse_transform(preds_scaled)
            targets_real = target_scaler.inverse_transform(batch.y)

            # Apply physical constraint: Energies cannot be negative
            preds_real = torch.clamp(preds_real, min=0.0)

            all_preds.append(preds_real.cpu().numpy())
            all_targets.append(targets_real.cpu().numpy())

    # Concatenate all batches
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate the official metric: Column-wise Root Mean Squared Logarithmic Error
    val_metric = calculate_rmsle(all_preds, all_targets)
    print(f"Final Validation Metric: {val_metric}")

    # -------------------------------------------------------------------------
    # 3. Failure Analysis
    # -------------------------------------------------------------------------
    print("\n--- Phase 3: Failure Analysis ---")

    # Calculate Mean Absolute Error (MAE) per sample across the two targets
    sample_errors = np.mean(np.abs(all_preds - all_targets), axis=1)

    # Retrieve raw global features from the dataset for correlation analysis
    # Features: Lattice parameters (6) + Composition fractions (4)
    feature_names = [
        "lattice_a",
        "lattice_b",
        "lattice_c",
        "lattice_alpha",
        "lattice_beta",
        "lattice_gamma",
        "frac_Al",
        "frac_Ga",
        "frac_In",
        "frac_O",
    ]

    # Ensure data alignment
    if len(sample_errors) == len(val_dataset.global_features):
        df_analysis = pd.DataFrame(val_dataset.global_features, columns=feature_names)
        df_analysis["error_magnitude"] = sample_errors

        # Compute correlation between error magnitude and input features
        correlations = df_analysis.corr()["error_magnitude"].drop("error_magnitude")
        print("Correlation between Error Magnitude and Input Features:")
        print(correlations.sort_values(ascending=False))
    else:
        print(
            "Warning: Mismatch in validation set size for analysis. Skipping correlation calculation."
        )

    # -------------------------------------------------------------------------
    # 4. Submission Generation
    # -------------------------------------------------------------------------
    print("\n--- Phase 4: Submission ---")
    TARGET_THRESHOLD = 0.05085437756413089

    if val_metric < TARGET_THRESHOLD:
        print(
            f"Validation metric {val_metric} is lower than threshold {TARGET_THRESHOLD}."
        )
        generate_submission(global_scaler=global_scaler, target_scaler=target_scaler)
    else:
        print(
            f"Validation metric {val_metric} is NOT lower than threshold {TARGET_THRESHOLD}."
        )
        print("Submission file will NOT be generated.")


if __name__ == "__main__":
    main()
