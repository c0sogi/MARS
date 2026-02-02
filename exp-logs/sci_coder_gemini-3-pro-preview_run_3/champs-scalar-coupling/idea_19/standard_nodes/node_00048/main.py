import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from library.config import Config
from library.train import Trainer, predict_submission
from library.utils import seed_everything, calc_log_mae
from library.data import MoleculeDataset, collate_molecules


def main():
    # 1. Configure for Fast Execution
    # We use debug=True and a small sample size to ensure completion within 3 minutes.
    config = Config(
        debug=True,
        debug_samples=2000,
        epochs=1,
        batch_size=32,
        num_workers=2,
        patience=1,
    )

    # Set random seeds for reproducibility
    seed_everything(config.seed)

    # 2. Train the Model
    trainer = Trainer(config)
    trainer.fit()

    # 3. Validation and Failure Analysis
    print("Starting Validation and Failure Analysis...")

    # Load the best model checkpoint
    device = torch.device(config.device)
    model = trainer.model
    if os.path.exists(config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    # Load Validation Dataset (uses cached debug data)
    val_dataset = MoleculeDataset(config, split="val", load_cached_data=True)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_molecules,
    )

    # Containers for analysis
    all_preds = []
    all_targets = []
    all_types = []
    all_dists = []

    standardizer = trainer.standardizer

    # Inference Loop
    with torch.no_grad():
        for batch in val_loader:
            # Move batch to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            # Forward Pass
            preds_std = model(batch)

            # Inverse Transform Predictions
            types = batch["coupling_type"]
            preds = standardizer.inverse_transform(preds_std.squeeze(), types)
            targets = batch["coupling_value"]

            # Calculate Inter-atomic Distances for Analysis
            coords = batch["atom_coords"]
            idx0 = batch["coupling_atom_index_0"]
            idx1 = batch["coupling_atom_index_1"]
            dists = (coords[idx0] - coords[idx1]).norm(dim=-1)

            # Store results
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())
            all_types.append(types.cpu())
            all_dists.append(dists.cpu())

    # Concatenate results
    y_pred = torch.cat(all_preds)
    y_true = torch.cat(all_targets)
    t_types = torch.cat(all_types)
    dists = torch.cat(all_dists)

    # Calculate and Print Final Metric
    metric = calc_log_mae(y_true, y_pred, t_types)
    print(f"Final Validation Metric: {metric}")

    # Failure Analysis: Correlations
    errors = torch.abs(y_true - y_pred)

    if len(errors) > 1:
        # Correlation between Error and Distance
        corr_dist = np.corrcoef(errors.numpy(), dists.numpy())[0, 1]
        print(f"Correlation between Error and Distance: {corr_dist:.4f}")

        # Correlation between Error and Target Magnitude
        corr_mag = np.corrcoef(errors.numpy(), torch.abs(y_true).numpy())[0, 1]
        print(f"Correlation between Error and Target Magnitude: {corr_mag:.4f}")
    else:
        print("Insufficient data for correlation analysis.")

    # 4. Submission Logic
    # Strict threshold check as per instructions
    THRESHOLD = -1.2761284112930298

    if metric < THRESHOLD:
        print(f"Metric {metric} is lower than {THRESHOLD}. Generating submission...")
        predict_submission(config)
    else:
        print(f"Metric {metric} is not lower than {THRESHOLD}. Skipping submission.")


if __name__ == "__main__":
    main()
