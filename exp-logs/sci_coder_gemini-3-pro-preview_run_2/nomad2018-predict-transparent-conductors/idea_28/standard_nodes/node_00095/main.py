import os
import torch
import pandas as pd
import numpy as np

# Import library modules
from library.config import Config
from library.data import get_dataloaders
from library.model import SRACGN
from library.utils import get_scaler, compute_metric
from library.train import run_training


def main():
    # -------------------------------------------------------------------------
    # 1. Configuration and Setup
    # -------------------------------------------------------------------------
    # Set random seeds for reproducibility
    seed = Config.SEED
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Use Config values directly
    print(f"Configured for training: {Config.NUM_EPOCHS} epochs.")

    # -------------------------------------------------------------------------
    # 2. Model Training
    # -------------------------------------------------------------------------
    print("\n--- Starting Training Phase ---")
    # run_training encapsulates the training loop, validation, and checkpointing
    run_training(
        epochs=Config.NUM_EPOCHS,
        batch_size=Config.BATCH_SIZE,
        learning_rate=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
        patience=Config.EARLY_STOPPING_PATIENCE,
        load_cached_data=True,
    )

    # -------------------------------------------------------------------------
    # 3. Validation Assessment & Failure Analysis
    # -------------------------------------------------------------------------
    print("\n--- Starting Validation Assessment ---")

    # Load the best model
    if not os.path.exists(Config.MODEL_CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Checkpoint not found at {Config.MODEL_CHECKPOINT_PATH}"
        )

    model = SRACGN().to(device)
    model.load_state_dict(torch.load(Config.MODEL_CHECKPOINT_PATH, map_location=device))
    model.eval()

    # Load validation data
    # We use the same loader function but only need the val_loader
    _, val_loader, _ = get_dataloaders(load_cached_data=True)

    # Load Scaler (it was fit during training)
    scaler = get_scaler(None, load_cached_data=True)

    # Inference on Validation Set
    val_preds = []
    val_targets = []
    val_ids = []

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            outputs = model(batch)

            # Inverse transform predictions
            preds_np = scaler.inverse_transform(outputs.cpu().numpy())
            targets_np = batch.y.cpu().numpy()
            ids_np = batch.id.cpu().numpy()

            val_preds.append(preds_np)
            val_targets.append(targets_np)
            val_ids.append(ids_np)

    val_preds = np.concatenate(val_preds, axis=0)
    val_targets = np.concatenate(val_targets, axis=0)
    val_ids = np.concatenate(val_ids, axis=0)

    # Compute Metric
    final_metric = compute_metric(val_targets, val_preds)
    print(f"Final Validation Metric: {final_metric}")

    # --- Failure Analysis ---
    print("\n--- Performing Failure Analysis ---")

    # Calculate error magnitude (Mean Absolute Error per sample across targets)
    # Target 0: Formation Energy, Target 1: Bandgap
    errors = np.abs(val_targets - val_preds)
    mean_errors = np.mean(errors, axis=1)  # Shape: (N,)

    # Load metadata to correlate with features
    val_metadata = pd.read_csv(Config.VAL_METADATA_PATH)

    # Create a dataframe for analysis
    analysis_df = pd.DataFrame({"id": val_ids, "error_magnitude": mean_errors})

    # Merge with metadata
    analysis_df = analysis_df.merge(val_metadata, on="id", how="left")

    # Select numerical columns for correlation
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

    print("Correlation between Error Magnitude and Input Features:")
    correlations = (
        analysis_df[feature_cols]
        .corrwith(analysis_df["error_magnitude"])
        .sort_values(ascending=False)
    )
    print(correlations)

    # -------------------------------------------------------------------------
    # 4. Submission Generation
    # -------------------------------------------------------------------------
    THRESHOLD = 0.049412816762924194

    if final_metric < THRESHOLD:
        print(
            f"\nValidation metric ({final_metric}) is better than threshold ({THRESHOLD}). Generating submission..."
        )

        # Load Test Data
        _, _, test_loader = get_dataloaders(load_cached_data=True)

        test_preds = []
        test_ids = []

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                outputs = model(batch)

                # Inverse transform
                preds_np = scaler.inverse_transform(outputs.cpu().numpy())
                ids_np = batch.id.cpu().numpy()

                test_preds.append(preds_np)
                test_ids.append(ids_np)

        test_preds = np.concatenate(test_preds, axis=0)
        test_ids = np.concatenate(test_ids, axis=0)

        # Create Submission DataFrame
        submission_df = pd.DataFrame(
            {
                "id": test_ids,
                "formation_energy_ev_natom": test_preds[:, 0],
                "bandgap_energy_ev": test_preds[:, 1],
            }
        )

        # Sort by ID
        submission_df = submission_df.sort_values("id")

        # Save
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nValidation metric ({final_metric}) did not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
