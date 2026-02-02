import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import from the provided library
from library.config import Config
from library.engine import run_training, generate_submission
from library.data import get_data, CrystalGraphDataset, collate_graphs
from library.model import DBGT
from library.utils import TargetScaler, compute_rmsle


def set_seed(seed):
    """Sets random seed for reproducibility."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_and_analyze():
    """
    Loads the best model, performs inference on the validation set,
    computes the official metric, and runs failure analysis.
    """
    device = torch.device(Config.DEVICE)

    # 1. Load Validation Data
    # We use load_cached_data=True because run_training has likely already cached it
    val_graphs, val_targets, val_ids = get_data(
        Config.VAL_METADATA_PATH,
        "val",
        load_cached_data=True,
        debug=Config.DEBUG,
        debug_size=Config.DEBUG_SIZE,
    )

    val_dataset = CrystalGraphDataset(val_graphs, val_targets, val_ids)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_graphs,
        num_workers=Config.NUM_WORKERS,
    )

    # 2. Load Best Model and Scaler
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model_runfile.pth")
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Checkpoint not found at {best_model_path}")

    print(f"Loading best model from {best_model_path} for analysis...")
    # Set weights_only=False to support numpy arrays in scaler state
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)

    model = DBGT(config=Config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    scaler = TargetScaler()
    scaler.load_state_dict(checkpoint["scaler_state_dict"])

    # 3. Inference
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            x = batch["x"].to(device)
            edge_index = batch["edge_index"].to(device)
            edge_attr = batch["edge_attr"].to(device)
            batch_idx = batch["batch"].to(device)
            targets = batch["y"].to(device)

            outputs = model(
                {
                    "x": x,
                    "edge_index": edge_index,
                    "edge_attr": edge_attr,
                    "batch": batch_idx,
                }
            )

            # Inverse transform predictions to original scale
            preds_orig = scaler.inverse_transform(outputs)

            # Clip negative predictions to 0 (physical constraint)
            preds_orig = torch.clamp(preds_orig, min=0.0)

            all_preds.append(preds_orig.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # 4. Compute Metric
    # Metric is Column-wise Root Mean Squared Logarithmic Error
    rmsle = compute_rmsle(all_targets, all_preds)
    print(f"Final Validation Metric: {rmsle}")

    # 5. Failure Analysis
    print("\nRunning Failure Analysis...")

    # Calculate error per sample: Mean Absolute Log Error
    # log1p(pred) - log1p(true)
    log_diff = np.log1p(all_preds) - np.log1p(all_targets)
    # Mean absolute error across the 2 targets for each sample
    sample_errors = np.mean(np.abs(log_diff), axis=1)

    # Load metadata to get features
    val_meta_df = pd.read_csv(Config.VAL_METADATA_PATH)

    # Filter metadata to match the current validation set (in case of debug mode)
    # We map using 'id'
    val_ids_np = np.array(val_ids)

    # Create a dataframe for analysis
    analysis_df = pd.DataFrame({"id": val_ids_np, "error": sample_errors})

    # Merge with metadata features
    # Select numerical columns from metadata
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

    # Ensure we only merge on available columns
    available_cols = [c for c in feature_cols if c in val_meta_df.columns]

    merged_df = analysis_df.merge(
        val_meta_df[["id"] + available_cols], on="id", how="left"
    )

    # Compute correlations
    print("Correlation between Error Magnitude and Features:")
    correlations = (
        merged_df[available_cols]
        .corrwith(merged_df["error"])
        .sort_values(ascending=False, key=abs)
    )
    print(correlations)

    return rmsle


def main():
    # 1. Setup
    set_seed(Config.SEED)

    # 2. Configure for Full Training
    # We use the full epoch count defined in Config to ensure convergence
    print(
        f"Configuration: Running for {Config.NUM_EPOCHS} epochs on device {Config.DEVICE}"
    )

    # 3. Run Training
    # This will train the model and save the best checkpoint to Config.CHECKPOINT_DIR
    run_training(load_cached_data=True)

    # 4. Validation and Analysis
    val_metric = validate_and_analyze()

    # 5. Conditional Submission
    threshold = 0.053007537912991315
    if val_metric < threshold:
        print(f"\nValidation metric {val_metric} is better than threshold {threshold}.")
        print("Generating submission...")
        generate_submission(load_cached_data=True)
    else:
        print(
            f"\nValidation metric {val_metric} is NOT better than threshold {threshold}."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
