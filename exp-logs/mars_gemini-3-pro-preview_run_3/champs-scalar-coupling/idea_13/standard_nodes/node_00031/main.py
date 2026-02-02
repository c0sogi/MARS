import os
import torch
import numpy as np
import pandas as pd
import sys

# Import library modules
from library.config import Config
from library.data_processor import DataProcessor
from library.trainer import Trainer
from library.inference import Predictor
from library.losses import LossComputer


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    print("Setting up configuration for fast baseline...")

    # Patch Config for runtime constraints
    # Increased to 50 epochs for convergence (Cite solution_lesson_node_00020)
    Config.MAX_EPOCHS = 50
    # Config.BATCH_SIZE is set in config.py (512)
    Config.DEBUG = False  # Must use full data to have a chance at the metric

    # Ensure reproducibility
    Config.set_seed(Config.SEED)

    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # ==========================================
    # 2. Data Processing
    # ==========================================
    # Ensure data is processed and cached
    dp = DataProcessor()
    dp.run(load_cached_data=True)

    # ==========================================
    # 3. Training
    # ==========================================
    print("\nStarting Training Phase...")
    trainer = Trainer()
    trainer.train()

    # ==========================================
    # 4. Validation & Metric Calculation
    # ==========================================
    print("\nStarting Validation & Failure Analysis...")

    # Load best model for validation
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    if not os.path.exists(best_model_path):
        print("Error: Best model not found. Training may have failed.")
        sys.exit(1)

    model = trainer.model
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # We need the LossComputer for unstandardization
    loss_computer = trainer.loss_computer

    # Containers for analysis
    all_preds = []
    all_targets = []
    all_types = []
    all_dists = []

    val_loader = trainer.val_loader

    with torch.no_grad():
        for batch in val_loader:
            # Move batch to device
            batch = trainer._to_device(batch)

            # Inference
            with torch.cuda.amp.autocast():
                preds = model(batch)
                pred_coupling_std = preds["coupling"]

            # Unstandardize
            coupling_types = batch["coupling_type"]
            pred_coupling_phys = loss_computer.unstandardize(
                pred_coupling_std, coupling_types
            )
            target_coupling = batch["coupling_value"]

            # Store data for metric calculation and failure analysis
            all_preds.append(pred_coupling_phys.cpu().numpy())
            all_targets.append(target_coupling.cpu().numpy())
            all_types.append(coupling_types.cpu().numpy())

            # Calculate distances for failure analysis
            # pos: (N, 3), coupling_atom_index: (2, C)
            pos = batch["pos"]
            c_idx = batch["coupling_atom_index"]
            p0 = pos[c_idx[0]]
            p1 = pos[c_idx[1]]
            dists = (p0 - p1).norm(dim=-1)
            all_dists.append(dists.cpu().numpy())

    # Concatenate all batches
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    types = np.concatenate(all_types)
    dists = np.concatenate(all_dists)

    # --- Compute Competition Metric ---
    # Log of the Mean Absolute Error, calculated for each scalar coupling type, and then averaged across types.
    abs_diff = np.abs(y_pred - y_true)

    log_maes = []
    unique_types = np.unique(types)

    print("\nPer-type Performance:")
    for t in unique_types:
        mask = types == t
        mae = np.mean(abs_diff[mask])
        log_mae = np.log(mae + 1e-9)

        # Get type name for display
        t_name = Config.COUPLING_TYPES[t]
        print(f"  {t_name}: MAE={mae:.4f}, LogMAE={log_mae:.4f}")

        log_maes.append(log_mae)

    final_metric = np.mean(log_maes)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_metric}")

    # ==========================================
    # 5. Failure Analysis
    # ==========================================
    print("\n--- Failure Analysis ---")

    # Correlation between Error Magnitude and Distance
    corr_dist = np.corrcoef(abs_diff, dists)[0, 1]
    print(f"Correlation between Absolute Error and Distance: {corr_dist:.4f}")

    # Correlation between Error Magnitude and Coupling Type (using integer encoding)
    corr_type = np.corrcoef(abs_diff, types)[0, 1]
    print(
        f"Correlation between Absolute Error and Coupling Type Index: {corr_type:.4f}"
    )

    # ==========================================
    # 6. Conditional Submission
    # ==========================================
    threshold = -1.2761284112930298

    if final_metric < threshold:
        print(
            f"\nMetric ({final_metric}) is better than threshold ({threshold}). Generating submission..."
        )
        predictor = Predictor()
        predictor.predict()
    else:
        print(
            f"\nMetric ({final_metric}) did not meet threshold ({threshold}). Skipping submission."
        )


if __name__ == "__main__":
    main()
