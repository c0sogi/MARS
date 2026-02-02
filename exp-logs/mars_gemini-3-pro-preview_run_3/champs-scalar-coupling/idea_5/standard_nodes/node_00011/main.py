import os
import sys
import torch
import numpy as np
import pandas as pd
import time

# Import library modules
from library.config import Config
from library.data import get_dataloaders
from library.engine import Trainer
from library.utils import TargetScaler


def pearson_corr(x, y):
    """
    Calculates Pearson correlation coefficient using NumPy.
    """
    if len(x) < 2:
        return 0.0
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    num = np.sum((x - x_mean) * (y - y_mean))
    den = np.sqrt(np.sum((x - x_mean) ** 2) * np.sum((y - y_mean) ** 2))
    if den == 0:
        return 0.0
    return num / den


def run_pipeline():
    # ==========================================
    # 1. Configuration for Fast Baseline
    # ==========================================
    print("Configuring for fast baseline execution...")
    # Modify Config for a fast, representative run
    Config.DEBUG = True
    # data.py divides this by 10 to get num_molecules.
    # 20000 -> 2000 molecules -> approx 30k coupling samples.
    Config.DEBUG_SAMPLE_SIZE = 20000
    Config.MAX_EPOCHS = 3
    Config.NUM_WORKERS = 4

    # Setup environment
    Config.setup_environment()

    # Clear debug cache to ensure we process the correct number of samples
    if Config.DEBUG:
        processed_dir = Config.PROCESSED_DATA_DIR
        if os.path.exists(processed_dir):
            for f in os.listdir(processed_dir):
                if "_debug.pt" in f:
                    try:
                        os.remove(os.path.join(processed_dir, f))
                        print(f"Removed old cache file: {f}")
                    except OSError as e:
                        print(f"Error removing cache file {f}: {e}")

    # ==========================================
    # 2. Data Loading
    # ==========================================
    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(debug=Config.DEBUG)

    # ==========================================
    # 3. Training
    # ==========================================
    print("Initializing Trainer...")
    trainer = Trainer()

    print("Starting Training Loop...")
    trainer.fit(train_loader, val_loader)

    # ==========================================
    # 4. Validation & Failure Analysis
    # ==========================================
    print("Starting Validation Analysis...")

    # Load the best model saved during training
    if os.path.exists(Config.MODEL_SAVE_PATH):
        print(f"Loading best model from {Config.MODEL_SAVE_PATH}")
        trainer.model.load_state_dict(
            torch.load(Config.MODEL_SAVE_PATH, map_location=trainer.device)
        )
    else:
        print("Warning: No model checkpoint found. Using current model state.")

    trainer.model.eval()

    # Containers for analysis
    all_preds = []
    all_targets = []
    all_types = []
    all_dists = []

    # Inference loop on Validation Set
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(trainer.device)

            # Forward pass
            pred_std, _, _ = trainer.model(batch)

            # Inverse transform to physical scale
            pred_phys = trainer.scaler.inverse_transform(pred_std, batch.type_coupling)

            # Store results
            all_preds.append(pred_phys.cpu().numpy())
            all_targets.append(batch.y_coupling.cpu().numpy())
            all_types.append(batch.type_coupling.cpu().numpy())

            # Calculate distances for the coupling pairs for analysis
            # batch.pos: (N, 3), batch.edge_index_coupling: (2, E)
            row = batch.edge_index_coupling[0]
            col = batch.edge_index_coupling[1]
            pos = batch.pos
            # Euclidean distance
            dists = torch.norm(pos[row] - pos[col], p=2, dim=-1)
            all_dists.append(dists.cpu().numpy())

    # Concatenate all batches
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_types = np.concatenate(all_types)
    all_dists = np.concatenate(all_dists)

    # --- Compute Metric (LogMAE) ---
    unique_types = np.unique(all_types)
    type_log_maes = []

    print("\n--- Performance per Type ---")
    for t_idx in unique_types:
        t_name = Config.COUPLING_TYPES[t_idx]
        mask = all_types == t_idx

        if np.sum(mask) > 0:
            mae = np.mean(np.abs(all_preds[mask] - all_targets[mask]))
            log_mae = np.log(mae)
            type_log_maes.append(log_mae)
            print(f"Type {t_name}: MAE={mae:.4f}, LogMAE={log_mae:.4f}")

    final_metric = np.mean(type_log_maes) if type_log_maes else 0.0

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_metric}")

    # --- Failure Analysis ---
    print("\n--- Failure Analysis ---")
    errors = np.abs(all_preds - all_targets)

    if len(errors) > 1:
        # Correlation: Error vs Distance
        corr_dist = pearson_corr(errors, all_dists)
        print(f"Correlation between Error and Distance: {corr_dist:.4f}")

        # Correlation: Error vs Target Magnitude
        corr_mag = pearson_corr(errors, np.abs(all_targets))
        print(f"Correlation between Error and Target Magnitude: {corr_mag:.4f}")
    else:
        print("Not enough samples for correlation analysis.")

    # ==========================================
    # 5. Submission
    # ==========================================
    # Threshold check
    THRESHOLD = -1.2761284112930298

    # LogMAE is better when lower.
    if final_metric < THRESHOLD:
        print(
            f"\nMetric {final_metric:.4f} is better than threshold {THRESHOLD:.4f}. Generating submission..."
        )
        trainer.predict(test_loader)
    else:
        print(
            f"\nMetric {final_metric:.4f} did not meet threshold {THRESHOLD:.4f}. Skipping submission."
        )


if __name__ == "__main__":
    run_pipeline()
