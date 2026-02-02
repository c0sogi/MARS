import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

# Import from provided libraries
from library.config import Config
from library.train import train_model, set_seed
from library.dataset import MaterialsDataset, collate_fn
from library.model import GDCC_WDS
from library.inference import generate_submission


def run_validation_and_analysis(model, device, val_loader):
    """
    Runs inference on the validation set, computes metrics, and performs failure analysis.
    """
    model.eval()
    all_preds_log = []
    all_targets_log = []
    all_global_feats = []

    print("Running validation inference...")
    with torch.no_grad():
        for batch in val_loader:
            atomic_features = batch["atomic_features"].to(device)
            global_features = batch["global_features"].to(device)
            batch_indices = batch["batch_indices"].to(device)
            targets = batch["targets"].to(device)

            num_graphs = len(batch["ids"])

            # Forward pass
            outputs = model(atomic_features, global_features, batch_indices, num_graphs)

            all_preds_log.append(outputs.cpu().numpy())
            all_targets_log.append(targets.cpu().numpy())
            all_global_feats.append(batch["global_features"].cpu().numpy())

    preds_log = np.concatenate(all_preds_log, axis=0)
    targets_log = np.concatenate(all_targets_log, axis=0)
    global_feats = np.concatenate(all_global_feats, axis=0)

    # --- Compute Metric ---
    # Metric: Column-wise Root Mean Squared Logarithmic Error
    # Since the model predicts log(1+y) and targets are log(1+y),
    # we calculate RMSE on these values directly.
    mse_per_col = np.mean((preds_log - targets_log) ** 2, axis=0)
    rmsle_per_col = np.sqrt(mse_per_col)

    # The final metric is typically the mean of the column-wise metrics for ranking
    final_metric = np.mean(rmsle_per_col)

    print(
        f"Validation RMSLE per column: Formation Energy={rmsle_per_col[0]:.6f}, Bandgap={rmsle_per_col[1]:.6f}"
    )
    print(f"Final Validation Metric: {final_metric}")

    # --- Failure Analysis ---
    print("\nPerforming Failure Analysis...")
    # Calculate error magnitude (L1 norm of log-error per sample)
    # error_magnitude shape: (N_samples,)
    error_magnitude = np.mean(np.abs(preds_log - targets_log), axis=1)

    # Feature names based on Data Processing (Global Stream)
    # 3 (Lattice Lens) + 3 (Lattice Angs) + 1 (Volume) + 1 (Density) + 1 (Total Atoms) + 3 (Stoichiometry)
    feature_names = [
        "Lattice_A",
        "Lattice_B",
        "Lattice_C",
        "Angle_Alpha",
        "Angle_Beta",
        "Angle_Gamma",
        "Volume",
        "Density",
        "Total_Atoms",
        "Stoich_Al",
        "Stoich_Ga",
        "Stoich_In",
    ]

    # We are using scaled features from the loader, but correlations work fine on scaled data (linear transform)
    # Compute correlation between each global feature and the error magnitude
    print("Correlation between Error Magnitude and Global Features:")
    correlations = []
    for i, name in enumerate(feature_names):
        if i < global_feats.shape[1]:
            feat_values = global_feats[:, i]
            # Handle potential constant features (std=0)
            if np.std(feat_values) > 1e-9:
                corr, _ = pearsonr(feat_values, error_magnitude)
                correlations.append((name, corr))
            else:
                correlations.append((name, 0.0))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    for name, corr in correlations:
        print(f"  {name:<15}: {corr:.4f}")

    return final_metric


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. Train Model
    # Using 100 epochs for a fast but effective baseline.
    # The dataset is small (~1.7k samples), so 100 epochs is very quick.
    print("Starting Model Training...")
    train_model(epochs=100, batch_size=Config.BATCH_SIZE, load_cached_data=True)

    # 3. Load Best Model
    print("\nLoading Best Model for Validation...")
    if not os.path.exists(Config.BEST_MODEL_PATH):
        print("Error: Best model file not found.")
        sys.exit(1)

    model = GDCC_WDS()
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    model.to(device)

    # 4. Prepare Validation Data
    # mode='val' ensures we use the scalers fitted on training data
    val_dataset = MaterialsDataset(
        metadata_path=Config.VAL_METADATA_PATH, mode="val", load_cached_data=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
    )

    # 5. Run Validation & Analysis
    metric = run_validation_and_analysis(model, device, val_loader)

    # 6. Submission Logic
    # Threshold from instructions: 0.05479004207787702
    THRESHOLD = 0.05479004207787702

    if metric < THRESHOLD:
        print(
            f"\nMetric {metric} is below threshold {THRESHOLD}. Generating submission..."
        )
        submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
        generate_submission(
            model_path=Config.BEST_MODEL_PATH,
            output_path=submission_path,
            load_cached_data=True,
        )
    else:
        print(
            f"\nMetric {metric} is NOT below threshold {THRESHOLD}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
