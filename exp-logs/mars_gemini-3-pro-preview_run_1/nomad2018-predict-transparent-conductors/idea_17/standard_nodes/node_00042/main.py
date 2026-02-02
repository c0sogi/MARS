import os
import torch
import numpy as np
import pandas as pd
import random
from library.config import (
    SEED,
    DEVICE,
    MODEL_SAVE_PATH,
    SUBMISSION_PATH,
    TARGET_COLS,
    WORKING_DIR,
)
from library.engine import run_training, generate_submission
from library.dataset import get_dataloaders
from library.model import PCWDSModel


def set_seed(seed):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    # 1. Setup
    set_seed(SEED)
    print(f"Orchestrating PC-WDS pipeline on {DEVICE}...")

    # 2. Training
    # This handles data preprocessing (if not cached), training loop,
    # validation monitoring, and saving the best model checkpoint.
    print("\n" + "=" * 40)
    print(" PHASE 1: TRAINING")
    print("=" * 40)
    run_training(load_cached_data=True)

    # 3. Validation & Failure Analysis
    print("\n" + "=" * 40)
    print(" PHASE 2: VALIDATION & FAILURE ANALYSIS")
    print("=" * 40)

    # Load validation data
    # We use the same get_dataloaders function to ensure consistent preprocessing
    _, val_loader, _ = get_dataloaders(load_cached_data=True)

    # Load the best trained model
    model = PCWDSModel().to(DEVICE)
    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"Critical Error: Model checkpoint not found at {MODEL_SAVE_PATH}")
        return

    print(f"Loading best model from {MODEL_SAVE_PATH}...")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    model.eval()

    all_preds_log = []
    all_targets_log = []
    all_global_feats = []

    # Run inference on validation set
    # We disable gradient calculation for speed and memory efficiency
    with torch.no_grad():
        for atom_x, batch_indices, global_x, targets, _ in val_loader:
            # Move batch to device
            atom_x = atom_x.to(DEVICE)
            batch_indices = batch_indices.to(DEVICE)
            global_x_gpu = global_x.to(DEVICE)
            targets = targets.to(DEVICE)

            # Forward pass
            outputs = model(atom_x, batch_indices, global_x_gpu)

            # Store results (keep in log space for metric calculation)
            all_preds_log.append(outputs.cpu().numpy())
            all_targets_log.append(targets.cpu().numpy())
            all_global_feats.append(global_x.numpy())  # Keep on CPU

    # Concatenate all batches
    preds_log = np.concatenate(all_preds_log, axis=0)
    targets_log = np.concatenate(all_targets_log, axis=0)
    global_feats = np.concatenate(all_global_feats, axis=0)

    # Calculate Metric: Column-wise Root Mean Squared Logarithmic Error (RMSLE)
    # Note: The model was trained on log1p(y), so the predictions and targets
    # are already in the logarithmic space. RMSE on these values IS the RMSLE.
    # Metric = mean( sqrt( mean( (log(1+p) - log(1+t))^2 ) ) ) over columns
    mse_per_col = np.mean((preds_log - targets_log) ** 2, axis=0)
    rmsle_per_col = np.sqrt(mse_per_col)
    final_metric = np.mean(rmsle_per_col)

    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    # We calculate the error magnitude per sample (Euclidean distance in log space)
    # and correlate it with global features to find sources of error.
    sample_errors = np.sqrt(np.mean((preds_log - targets_log) ** 2, axis=1))

    # Feature names corresponding to get_global_features in data_utils.py
    # Order: lv1, lv2, lv3, alpha, beta, gamma, volume, density, Al, Ga, In, total_atoms
    feature_names = [
        "lattice_vector_1_ang",
        "lattice_vector_2_ang",
        "lattice_vector_3_ang",
        "lattice_angle_alpha_degree",
        "lattice_angle_beta_degree",
        "lattice_angle_gamma_degree",
        "cell_volume",
        "atomic_density",
        "percent_atom_al",
        "percent_atom_ga",
        "percent_atom_in",
        "number_of_total_atoms",
    ]

    analysis_df = pd.DataFrame(global_feats, columns=feature_names)
    analysis_df["error_magnitude"] = sample_errors

    # Compute correlations
    correlations = analysis_df.corr()["error_magnitude"].drop("error_magnitude")
    # Sort by absolute correlation strength
    correlations = correlations.iloc[correlations.abs().argsort()[::-1]]

    print("\nCorrelation between Error Magnitude and Input Features:")
    print(correlations)

    # 4. Submission
    print("\n" + "=" * 40)
    print(" PHASE 3: SUBMISSION GENERATION")
    print("=" * 40)

    THRESHOLD = 0.05479004207787702

    if final_metric < THRESHOLD:
        print(f"Validation metric {final_metric} meets threshold ({THRESHOLD}).")
        generate_submission(load_cached_data=True)
    else:
        print(
            f"Validation metric {final_metric} does NOT meet threshold ({THRESHOLD})."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
