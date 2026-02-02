import os
import sys
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_log_error
import warnings

# Add current directory to sys.path
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed, StandardScaler
from library.data import get_dataloaders
from library.model import RA_CGN_AR
from library.engine import run_training

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def compute_rmsle(y_true, y_pred):
    """
    Computes Column-wise Root Mean Squared Logarithmic Error.
    y_true, y_pred: numpy arrays of shape (N, 2)
    """
    # Ensure non-negative predictions for log
    y_pred = np.maximum(y_pred, 0)

    # Calculate MSLE for each column
    # multioutput='raw_values' returns an array of errors, one for each output
    msle = mean_squared_log_error(y_true, y_pred, multioutput="raw_values")

    # Take sqrt to get RMSLE per column
    rmsle = np.sqrt(msle)

    # Return the mean of column-wise RMSLEs
    return np.mean(rmsle)


def main():
    # 1. Setup and Configuration
    print("Initializing RA-CGN-AR Pipeline...")
    set_seed(Config.SEED)

    # Override Config for Optimized Run
    Config.NUM_EPOCHS = 120  # Increased epochs for convergence
    Config.BATCH_SIZE = (
        48  # Reduced batch size for generalization (Cite solution_lesson_node_00005)
    )

    device = torch.device(Config.DEVICE)

    # 2. Train Model
    print("Starting Training...")
    # Using full dataset (subset_size=None) but fewer epochs
    run_training(subset_size=None, load_cached_data=True)

    # 3. Load Best Model and Scaler
    print("Loading best model and scaler...")
    model = RA_CGN_AR().to(device)
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # Load scaler
    scaler = StandardScaler(device=device)
    scaler.load(os.path.join(Config.CACHE_DIR, "target_scaler.npz"))

    # 4. Validation & Metric Calculation
    print("Running Validation Inference...")
    _, val_loader, _ = get_dataloaders(load_cached_data=True)

    val_ids = []
    val_preds = []
    val_targets = []

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            outputs = model(batch)

            # Inverse transform predictions
            outputs_original = scaler.inverse_transform(outputs)

            val_preds.append(outputs_original.cpu().numpy())
            val_targets.append(batch.y.cpu().numpy())
            val_ids.extend(batch.id.cpu().numpy())

    val_preds = np.concatenate(val_preds, axis=0)
    val_targets = np.concatenate(val_targets, axis=0)

    # Compute Metric
    final_metric = compute_rmsle(val_targets, val_preds)
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Load metadata to get features
    val_meta_df = pd.read_csv(Config.VAL_METADATA_PATH)

    # Map predictions/targets to metadata using IDs
    # Create a dataframe for results
    results_df = pd.DataFrame(
        {
            "id": val_ids,
            "pred_formation": val_preds[:, 0],
            "pred_bandgap": val_preds[:, 1],
            "true_formation": val_targets[:, 0],
            "true_bandgap": val_targets[:, 1],
        }
    )

    # Calculate errors
    results_df["error_formation"] = np.abs(
        results_df["pred_formation"] - results_df["true_formation"]
    )
    results_df["error_bandgap"] = np.abs(
        results_df["pred_bandgap"] - results_df["true_bandgap"]
    )
    results_df["mean_error"] = (
        results_df["error_formation"] + results_df["error_bandgap"]
    ) / 2

    # Merge with metadata features
    analysis_df = pd.merge(results_df, val_meta_df, on="id", how="left")

    # Features to analyze
    features = [
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

    print("Correlation between Mean Absolute Error and Features:")
    correlations = {}
    for feat in features:
        if feat in analysis_df.columns:
            corr = analysis_df["mean_error"].corr(analysis_df[feat])
            correlations[feat] = corr

    # Sort and print correlations
    sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    for feat, corr in sorted_corrs:
        print(f"  {feat:<30}: {corr:.4f}")

    # 6. Submission Generation
    threshold = 0.049412816762924194
    if final_metric < threshold:
        print(
            f"\nValidation metric ({final_metric}) is better than threshold ({threshold}). Generating submission..."
        )

        _, _, test_loader = get_dataloaders(load_cached_data=True)
        test_ids = []
        test_preds = []

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                outputs = model(batch)
                outputs_original = scaler.inverse_transform(outputs)

                test_preds.append(outputs_original.cpu().numpy())
                test_ids.extend(batch.id.cpu().numpy())

        test_preds = np.concatenate(test_preds, axis=0)

        # Ensure non-negative predictions (physics constraint)
        test_preds = np.maximum(test_preds, 0)

        submission_df = pd.DataFrame(
            {
                "id": test_ids,
                "formation_energy_ev_natom": test_preds[:, 0],
                "bandgap_energy_ev": test_preds[:, 1],
            }
        )

        # Sort by ID to match sample submission format usually
        submission_df = submission_df.sort_values("id")

        # Save
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

        # Preview
        print(submission_df.head())

    else:
        print(
            f"\nValidation metric ({final_metric}) did not meet threshold ({threshold}). Skipping submission."
        )


if __name__ == "__main__":
    main()
