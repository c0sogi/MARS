import sys
import os
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from library.config import Config
from library.data import get_dataloaders, COUPLING_TYPES
from library.model import HGANet
from library.train import Trainer, generate_submission
from library.utils import set_seed


def main():
    # 1. Configuration
    # We use fewer epochs for a fast baseline, but enough to learn.
    # A100 allows larger batch size.
    config = Config(epochs=4, batch_size=192, debug=False, hidden_dim=192)

    set_seed(config.SEED)
    print("Configuration configured. Starting pipeline...")

    # 2. Data Loading
    print("Loading DataLoaders...")
    train_loader, val_loader, _, standardizer = get_dataloaders(
        config, load_cached_data=True
    )

    # 3. Model Initialization
    print("Initializing HGA-Net Model...")
    model = HGANet(config)

    # 4. Training
    print("Starting Training...")
    trainer = Trainer(config, model, train_loader, val_loader, standardizer)
    trainer.fit()

    # 5. Metric Reporting
    final_metric = trainer.best_metric
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("\nStarting Failure Analysis on Validation Set...")

    # Load best model for analysis
    model.load_state_dict(
        torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE)
    )
    model.eval()

    val_abs_errors = []
    val_targets = []
    val_mol_sizes = []

    with torch.no_grad():
        for batch in val_loader:
            batch = {
                k: v.to(config.DEVICE) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # Predict
            preds = model(batch)

            # Get targets and types for inverse transform
            targets = batch["y"]
            types = batch["coupling_type"]

            # Move to CPU
            preds_np = preds.cpu().numpy()
            targets_np = targets.cpu().numpy()
            types_np = types.cpu().numpy()

            # Inverse Transform
            type_str_map = np.array(COUPLING_TYPES)
            types_str = type_str_map[types_np]

            orig_preds = standardizer.inverse_transform(preds_np, types_str)
            orig_targets = standardizer.inverse_transform(targets_np, types_str)

            # Calculate Errors
            abs_err = np.abs(orig_preds - orig_targets)
            val_abs_errors.append(abs_err)
            val_targets.append(np.abs(orig_targets))  # Magnitude of target

            # Calculate Molecule Sizes (Num atoms per coupling)
            # batch['batch'] maps nodes to graph index.
            # We need to map coupling index to graph index to get num atoms for that coupling's molecule.
            # batch['coupling_index'] refers to nodes. We can look up the graph index of the first atom in the coupling.
            # node_graph_indices = batch['batch']
            # coupling_node_0 = batch['coupling_index'][:, 0]
            # coupling_graph_indices = node_graph_indices[coupling_node_0]
            # Now count atoms per graph
            # graph_sizes = torch.bincount(batch['batch'])
            # coupling_mol_sizes = graph_sizes[coupling_graph_indices]

            node_graph_indices = batch["batch"]
            coupling_node_0 = batch["coupling_index"][:, 0]
            coupling_graph_indices = node_graph_indices[coupling_node_0]

            # bincount gives size of each graph 0..B-1
            graph_sizes = torch.bincount(batch["batch"])

            # Map back to couplings
            sizes = graph_sizes[coupling_graph_indices]
            val_mol_sizes.append(sizes.cpu().numpy())

    # Concatenate
    all_abs_errors = np.concatenate(val_abs_errors)
    all_targets = np.concatenate(val_targets)
    all_mol_sizes = np.concatenate(val_mol_sizes)

    # Correlations
    corr_target, _ = pearsonr(all_abs_errors, all_targets)
    corr_size, _ = pearsonr(all_abs_errors, all_mol_sizes)

    print(f"Correlation (Error vs Target Magnitude): {corr_target:.4f}")
    print(f"Correlation (Error vs Molecule Size): {corr_size:.4f}")

    # 7. Submission
    THRESHOLD = -1.407172441
    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric}) meets threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission(config)
    else:
        print(
            f"\nMetric ({final_metric}) does not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
