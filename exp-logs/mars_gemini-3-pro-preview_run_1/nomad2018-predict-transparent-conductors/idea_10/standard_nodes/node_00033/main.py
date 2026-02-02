import os
import torch
import pandas as pd
import numpy as np
import warnings

# Import from library
from library.config import Config
from library.train import run_training, generate_submission
from library.data import get_dataloaders
from library.model import RTDSModel
from library.utils import seed_everything, rmsle

# Suppress warnings
warnings.filterwarnings("ignore")


def main():
    # 1. Setup and Configuration
    Config.setup()

    # Adjust hyperparameters for a fast baseline execution
    # 15 epochs is sufficient for convergence on this dataset size
    Config.NUM_EPOCHS = 15

    # 2. Train the Model
    # process_dataset will cache data to Config.WORKING_DIR (./working/idea_10)
    print("--- Starting Training ---")
    run_training(load_cached_data=True, debug_size=None)

    # 3. Load the best model for validation and analysis
    device = torch.device(Config.DEVICE)
    model = RTDSModel().to(device)

    if not os.path.exists(Config.MODEL_SAVE_PATH):
        print(f"Error: Model file not found at {Config.MODEL_SAVE_PATH}")
        return

    print(f"Loading best model from {Config.MODEL_SAVE_PATH}...")
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    # 4. Perform Validation on the Hold-out Set
    print("--- Starting Validation Inference ---")
    # We retrieve the validation loader. load_cached_data=True ensures we use the data processed during training.
    _, val_loader, _ = get_dataloaders(load_cached_data=True, debug_size=None)

    all_preds_log = []
    all_targets_log = []
    all_ids = []

    with torch.no_grad():
        for atom_x, glob_x, targets_log, batch_indices, ids in val_loader:
            atom_x = atom_x.to(device)
            glob_x = glob_x.to(device)
            batch_indices = batch_indices.to(device)

            # Forward pass
            outputs_log = model(atom_x, glob_x, batch_indices)

            # Collect log-scale predictions and targets
            all_preds_log.append(outputs_log.cpu().numpy())
            all_targets_log.append(targets_log.cpu().numpy())
            all_ids.extend(ids)

    all_preds_log = np.concatenate(all_preds_log, axis=0)
    all_targets_log = np.concatenate(all_targets_log, axis=0)

    # Convert to original scale for metric calculation
    # The model predicts log1p(y), so we apply expm1(y) to get back to original units
    # Clip to 0 to ensure valid physical values
    all_preds_orig = np.maximum(np.expm1(all_preds_log), 0)
    all_targets_orig = np.maximum(np.expm1(all_targets_log), 0)

    # Calculate Final Validation Metric
    metric = rmsle(all_targets_orig, all_preds_orig)
    print(f"Final Validation Metric: {metric}")

    # 5. Failure Analysis
    print("--- Performing Failure Analysis ---")
    # Calculate per-sample error (mean absolute error in log space serves as a good proxy for relative error)
    # We average the error across the two targets
    sample_errors = np.mean(np.abs(all_preds_log - all_targets_log), axis=1)

    # Load metadata to correlate errors with physical features
    val_metadata = pd.read_csv(Config.VAL_METADATA_PATH)

    # Create analysis dataframe
    error_df = pd.DataFrame({"id": all_ids, "error": sample_errors})

    # Merge with metadata
    val_metadata["id"] = val_metadata["id"].astype(int)
    error_df["id"] = error_df["id"].astype(int)
    analysis_df = pd.merge(error_df, val_metadata, on="id", how="inner")

    # Calculate correlations between error magnitude and features
    numeric_cols = analysis_df.select_dtypes(include=[np.number]).columns.tolist()
    targets_to_exclude = ["formation_energy_ev_natom", "bandgap_energy_ev"]
    features_to_check = [
        c for c in numeric_cols if c not in ["id", "error"] + targets_to_exclude
    ]

    if features_to_check:
        correlations = (
            analysis_df[features_to_check]
            .corrwith(analysis_df["error"])
            .sort_values(key=abs, ascending=False)
        )
        print("Top correlations between Error and Features:")
        print(correlations.head(10))
    else:
        print("No numeric features found for correlation analysis.")

    # 6. Submission Generation
    THRESHOLD = 0.05479004207787702

    if metric < THRESHOLD:
        print(
            f"\nMetric ({metric}) is below threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission(load_cached_data=True)
    else:
        print(
            f"\nMetric ({metric}) is NOT below threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
