import sys
import os
import torch
import numpy as np
import pandas as pd
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Ensure library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.data import get_dataloaders
from library.trainer import Trainer
from library.utils import seed_everything, MetricLogger


def run_pipeline():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    config = Config()

    # Optimize for "Fast Baseline" execution
    config.MAX_EPOCHS = 5
    config.BATCH_SIZE = 128

    # Set random seeds
    seed_everything(config.SEED)

    print(f"Configuration: Epochs={config.MAX_EPOCHS}, Batch Size={config.BATCH_SIZE}")

    # ==========================================
    # 2. Data Loading
    # ==========================================
    print("Loading Data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=config.BATCH_SIZE, device=config.DEVICE, load_cached_data=True
    )

    # Cite solution_lesson_node_00014: Architecture cannot overcome a massive data deficit.
    # Training on the full dataset to maximize performance.
    pass

    # ==========================================
    # 3. Model Training
    # ==========================================
    print("Initializing Trainer...")
    trainer = Trainer(config)

    print("Starting Training Loop...")
    trainer.train(train_loader, val_loader)

    # ==========================================
    # 4. Final Validation & Metric Calculation
    # ==========================================
    print("\nPerforming Final Validation...")

    # Load the best model checkpoint
    if os.path.exists(config.MODEL_SAVE_PATH):
        trainer.model.load_state_dict(
            torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE)
        )
    else:
        print("Warning: No model checkpoint found. Using current model state.")

    trainer.model.eval()
    logger = MetricLogger()

    # Containers for Failure Analysis
    all_errors = []
    all_dists = []
    all_types = []

    with torch.no_grad():
        for batch in val_loader:
            # Forward Pass
            out = trainer.model(
                batch["atom_types"],
                batch["edge_index"],
                batch["edge_dist"],
                batch["triplet_index"],
                batch["triplet_angle"],
                batch["coupling_node_indices"],
                batch["coupling_edge_indices"],
                batch["coupling_types"],
            )

            # Extract Predictions (Scaled)
            preds_scaled = out["coupling"].squeeze()

            # Prepare Targets (Scaled) for Metric Calculation
            means = torch.tensor(trainer.scaler.mean_arr, device=config.DEVICE)[
                batch["coupling_types"]
            ]
            stds = torch.tensor(trainer.scaler.std_arr, device=config.DEVICE)[
                batch["coupling_types"]
            ]
            targets_scaled = (batch["coupling_values"] - means) / stds

            # Update Metric Logger
            logger.update(preds_scaled, targets_scaled, batch["coupling_types"])

            # --- Collect Data for Failure Analysis ---
            # Inverse transform to get physical units
            preds_phys = trainer.scaler.inverse_transform(
                preds_scaled, batch["coupling_types"]
            )
            targets_phys = batch["coupling_values"].cpu().numpy()

            # Calculate Absolute Error
            errors = np.abs(preds_phys - targets_phys)
            all_errors.append(errors)

            # Get Distances for the specific coupling edges
            # batch['edge_dist'] has all edge distances.
            # batch['coupling_edge_indices'] maps couplings to edges.
            dists = batch["edge_dist"][batch["coupling_edge_indices"]].cpu().numpy()
            all_dists.append(dists)

            # Get Coupling Types
            all_types.append(batch["coupling_types"].cpu().numpy())

    # Compute and Print Final Metric
    final_metric = logger.compute_metric(trainer.scaler)
    # REQUIRED FORMAT
    print(f"Final Validation Metric: {final_metric}")

    # ==========================================
    # 5. Failure Analysis
    # ==========================================
    print("\n--- Failure Analysis ---")
    all_errors = np.concatenate(all_errors)
    all_dists = np.concatenate(all_dists)
    all_types = np.concatenate(all_types)

    # Global Correlation
    valid_mask = np.isfinite(all_errors) & np.isfinite(all_dists)
    if np.sum(valid_mask) > 1:
        corr_dist = np.corrcoef(all_errors[valid_mask], all_dists[valid_mask])[0, 1]
        print(f"Correlation between Error and Distance: {corr_dist:.4f}")
    else:
        print("Insufficient data for correlation analysis.")

    # Per-Type Correlation
    print("Error vs Distance Correlation by Coupling Type:")
    df_analysis = pd.DataFrame(
        {"error": all_errors, "dist": all_dists, "type_idx": all_types}
    )

    type_map = {v: k for k, v in Config.COUPLING_MAP.items()}
    df_analysis["type"] = df_analysis["type_idx"].map(type_map)

    for t in Config.COUPLING_TYPES:
        subset = df_analysis[df_analysis["type"] == t]
        if len(subset) > 10:
            c = subset["error"].corr(subset["dist"])
            print(f"  {t}: {c:.4f}")

    # ==========================================
    # 6. Submission Generation
    # ==========================================
    THRESHOLD = -1.2761284112930298

    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric:.6f}) meets threshold ({THRESHOLD}). Generating submission..."
        )
        trainer.generate_submission(test_loader)
    else:
        print(
            f"\nMetric ({final_metric:.6f}) does not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    run_pipeline()
