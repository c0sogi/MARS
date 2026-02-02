import sys
import os
import numpy as np
import pandas as pd
import torch
import warnings
from scipy.stats import pearsonr

# Import provided library modules
from library.config import Config
from library.train import Trainer
from library.utils import set_seed

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_failure_analysis(trainer):
    """
    Performs failure analysis on the validation set.
    Calculates correlations between error magnitude and input features.
    """
    print("\nStarting Failure Analysis on Validation Set...")

    trainer.model.eval()
    device = trainer.device

    all_errors = []
    all_dists = []
    all_targets = []
    all_types = []

    # Disable gradients for inference
    with torch.no_grad():
        for batch in trainer.val_loader:
            # Move batch to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)

            # Forward pass
            pred_coupling_std, _, _ = trainer.model(batch)

            # Inverse transform predictions
            coupling_types = batch["coupling_type"]
            pred_coupling = trainer.standardizer.inverse_transform(
                pred_coupling_std, coupling_types
            )

            # Get targets
            targets = batch["y"]

            # Calculate Absolute Error
            errors = torch.abs(pred_coupling - targets)

            # Calculate Coupling Distances (Feature)
            # Indices in the batch
            idx_0, idx_1 = batch["coupling_index"]
            pos = batch["pos"]
            pos_0 = pos[idx_0]
            pos_1 = pos[idx_1]
            dists = torch.norm(pos_0 - pos_1, dim=-1)

            # Collect data
            all_errors.append(errors.cpu().numpy())
            all_dists.append(dists.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_types.append(coupling_types.cpu().numpy())

    # Concatenate all batches
    errors = np.concatenate(all_errors)
    dists = np.concatenate(all_dists)
    targets = np.concatenate(all_targets)
    types = np.concatenate(all_types)

    # 1. Correlation with Distance
    corr_dist, _ = pearsonr(errors, dists)
    print(f"Correlation between Error and Inter-atomic Distance: {corr_dist:.4f}")

    # 2. Correlation with Target Magnitude
    corr_mag, _ = pearsonr(errors, np.abs(targets))
    print(f"Correlation between Error and Target Magnitude: {corr_mag:.4f}")

    # 3. Error by Type
    df_analysis = pd.DataFrame({"error": errors, "type": types})
    # Map int types back to string for display if needed, but using raw ints is fine for grouping
    # We use the Config.COUPLING_TYPES list to map back
    type_map = {v: k for k, v in Config.TYPE_MAP.items()}
    df_analysis["type_name"] = df_analysis["type"].map(type_map)

    print("\nMean Absolute Error by Coupling Type:")
    print(df_analysis.groupby("type_name")["error"].mean().sort_values(ascending=False))


def main():
    # 1. Setup
    # Set seeds for reproducibility
    set_seed(Config.SEED)

    print(f"Initializing Trainer (Device: {Config.DEVICE})...")

    # 2. Initialize Trainer
    # load_cached_data=True ensures we use preprocessed .npy files if they exist
    trainer = Trainer(load_cached_data=True)

    # 3. Train
    trainer.fit(max_epochs=Config.MAX_EPOCHS)

    # 4. Validate
    print("Computing Final Validation Metric...")
    val_score = trainer.validate()

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {val_score}")

    # 5. Failure Analysis
    run_failure_analysis(trainer)

    # 6. Submission
    # Threshold defined in task description
    THRESHOLD = -1.2761284112930298

    if val_score < THRESHOLD:
        print(
            f"\nValidation score ({val_score}) meets threshold ({THRESHOLD}). Generating submission..."
        )
        trainer.predict()
    else:
        print(
            f"\nValidation score ({val_score}) does not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
