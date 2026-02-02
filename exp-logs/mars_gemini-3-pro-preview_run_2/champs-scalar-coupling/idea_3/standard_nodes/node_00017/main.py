import sys
import os
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# 1. Override Configuration for Fast Baseline
# We modify the config module before other libraries import from it
import library.config

library.config.DEBUG = False

# 2. Import Library Modules
from library.engine import Trainer
from library.utils import set_seed, log_mae
from library.config import DEVICE


def main():
    # Ensure reproducible results
    set_seed(42)

    # Initialize the Trainer
    print("Initializing Trainer...")
    trainer = Trainer()

    # Execute Training Pipeline
    print("Starting Training...")
    trainer.run()

    # ---------------------------------------------------------
    # Validation & Failure Analysis
    # ---------------------------------------------------------
    print("Performing Final Validation and Failure Analysis...")

    # Load the best model weights
    if os.path.exists(trainer.best_model_path):
        print(f"Loading best model from {trainer.best_model_path}")
        trainer.model.load_state_dict(
            torch.load(trainer.best_model_path, map_location=DEVICE)
        )
    else:
        print("Warning: Best model not found. Using current model state.")

    trainer.model.eval()
    val_loader = trainer.get_dataloader("val")

    all_preds = []
    all_targets = []
    all_types = []
    all_dists = []

    # Inference Loop
    with torch.no_grad():
        for batch in val_loader:
            # Move batch to device
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(DEVICE)

            # Forward pass
            preds_norm = trainer.model(batch)

            # Inverse transform targets to original scale
            preds = trainer.scaler.inverse_transform(preds_norm, batch["target_type"])
            targets = batch["y"]

            # Flatten tensors
            preds = preds.view(-1)
            targets = targets.view(-1)

            # Collect results
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())
            all_types.append(batch["target_type"].cpu())

            # Calculate inter-atomic distances for failure analysis
            # batch['target_edge_index'] contains indices of edges that are coupling targets
            target_indices = batch["target_edge_index"]
            src = batch["edge_index"][0, target_indices]
            dst = batch["edge_index"][1, target_indices]

            pos = batch["pos"]
            # Euclidean distance: ||pos_src - pos_dst||
            dists = (pos[src] - pos[dst]).norm(dim=1)
            all_dists.append(dists.cpu())

    # Concatenate all batches
    y_pred = torch.cat(all_preds)
    y_true = torch.cat(all_targets)
    types = torch.cat(all_types)
    dists = torch.cat(all_dists)

    # Compute Final Metric (Log MAE)
    metric = log_mae(y_true, y_pred, types)
    print(f"Final Validation Metric: {metric}")

    # ---------------------------------------------------------
    # Failure Analysis
    # ---------------------------------------------------------
    errors = torch.abs(y_true - y_pred).numpy()
    dists_np = dists.numpy()
    types_np = types.numpy()

    print("-" * 30)
    print("Failure Analysis Report")
    print("-" * 30)

    if len(errors) > 1:
        # Correlation between Error and Distance
        corr_dist, _ = pearsonr(errors, dists_np)
        print(f"Correlation (Error vs Distance): {corr_dist:.4f}")

        # Correlation between Error and Coupling Type (Encoded)
        corr_type, _ = pearsonr(errors, types_np)
        print(f"Correlation (Error vs Type Index): {corr_type:.4f}")
    else:
        print("Insufficient data for correlation analysis.")

    # ---------------------------------------------------------
    # Submission Generation
    # ---------------------------------------------------------
    # Generate submission only if metric is better (lower) than -1.407172441
    if metric < -1.407172441:
        print("\nMetric threshold met (< -1.407172441). Generating submission...")
        trainer.predict()
    else:
        print(f"\nMetric {metric} is not lower than -1.407172441. Submission skipped.")


if __name__ == "__main__":
    main()
