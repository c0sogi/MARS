import os
import sys
import torch
import numpy as np
import pandas as pd
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import from the provided library files
from library.config import (
    MODEL_SAVE_PATH,
    SEED,
    SUBMISSION_PATH,
    ATOM_INPUT_DIM,
    GLOBAL_INPUT_DIM,
    ATOM_HIDDEN_DIM,
    GLOBAL_HIDDEN_DIM,
    DROPOUT_RATE,
    NUM_EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    EARLY_STOPPING_PATIENCE,
)
from library.data_processing import get_dataloaders
from library.model import GPIMSDS
from library.training import Trainer
from library.inference import run_inference


def main():
    # 1. Setup Environment
    # Set random seeds for reproducibility
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)

    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # 2. Load Data
    # Load cached data if available to speed up the process
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Initialize Model
    print("Initializing GPI-MS-DS model...")
    model = GPIMSDS().to(device)

    # 4. Train Model
    # We use the Trainer class which handles the training loop and early stopping
    print("Starting training...")
    trainer = Trainer(model, device)
    trainer.fit(train_loader, val_loader)

    # 5. Validation Assessment
    print("\nPerforming validation assessment...")

    # Load the best model saved during training
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"Loading best model from {MODEL_SAVE_PATH}")
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    else:
        print("Warning: Best model checkpoint not found. Using current model state.")

    model.eval()

    all_preds_log = []
    all_targets_log = []
    all_global_feats = []

    # Inference on validation set (no gradient computation)
    with torch.no_grad():
        for batch in val_loader:
            # Move data to device
            atom_feats = batch["atom_features"].to(device)
            global_feats = batch["global_features"].to(device)
            batch_idx = batch["batch_index"].to(device)
            targets = batch["targets"].to(device)

            # Forward pass
            outputs = model(atom_feats, global_feats, batch_idx)

            # The model is trained on log(1+y), so outputs are in log space.
            # We compare against log(1+y_true) to calculate RMSLE.
            targets_log = torch.log1p(targets)

            all_preds_log.append(outputs.cpu().numpy())
            all_targets_log.append(targets_log.cpu().numpy())

            # Store global features for failure analysis (keep on CPU)
            all_global_feats.append(batch["global_features"].numpy())

    # Concatenate batches
    all_preds_log = np.concatenate(all_preds_log, axis=0)
    all_targets_log = np.concatenate(all_targets_log, axis=0)
    all_global_feats = np.concatenate(all_global_feats, axis=0)

    # Calculate Column-wise RMSLE
    # Metric: Column-wise root mean squared logarithmic error.
    # Since our predictions and targets are already in log space (log1p),
    # RMSLE is simply the RMSE of these log values.
    # We calculate MSE per column, take sqrt, then mean across columns.
    mse_per_col = np.mean((all_preds_log - all_targets_log) ** 2, axis=0)
    rmsle_per_col = np.sqrt(mse_per_col)
    final_metric = np.mean(rmsle_per_col)

    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("\nPerforming failure analysis...")

    # Calculate error magnitude per sample (MSE in log space)
    # This represents how far off the prediction is from the truth
    error_per_sample = np.mean((all_preds_log - all_targets_log) ** 2, axis=1)

    # Define feature names based on the order in data_processing.py
    # 3 (Lat Len) + 3 (Lat Ang) + 1 (Vol) + 1 (Dens) + 4 (Stoich) + 1 (Total) + 3 (Aspect) + 3 (Phys) + 1 (Dist) = 20
    global_feat_names = [
        "lat_len_a",
        "lat_len_b",
        "lat_len_c",
        "lat_ang_alpha",
        "lat_ang_beta",
        "lat_ang_gamma",
        "volume",
        "density",
        "stoich_Al",
        "stoich_Ga",
        "stoich_In",
        "stoich_O",
        "total_atoms",
        "aspect_ab",
        "aspect_bc",
        "aspect_ca",
        "mean_mass",
        "mean_radius",
        "mean_en",
        "ang_distortion",
    ]

    # Create DataFrame for analysis
    analysis_df = pd.DataFrame(all_global_feats, columns=global_feat_names)
    analysis_df["error_magnitude"] = error_per_sample

    # Calculate correlation between features and error magnitude
    correlations = analysis_df.corr()["error_magnitude"].drop("error_magnitude")

    # Sort by absolute correlation
    top_correlations = correlations.abs().sort_values(ascending=False).head(5)

    print("Top 5 features correlated with model error:")
    for feat, corr in top_correlations.items():
        print(f"  {feat}: {correlations[feat]:.4f}")

    # 7. Submission Generation
    THRESHOLD = 0.04819517582654953

    if final_metric < THRESHOLD:
        print(f"\nValidation metric {final_metric} meets threshold ({THRESHOLD}).")
        print("Generating submission for test set...")

        # run_inference loads the best model from disk and generates the csv
        run_inference(load_cached_data=True)

    else:
        print(
            f"\nValidation metric {final_metric} does NOT meet threshold ({THRESHOLD})."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
