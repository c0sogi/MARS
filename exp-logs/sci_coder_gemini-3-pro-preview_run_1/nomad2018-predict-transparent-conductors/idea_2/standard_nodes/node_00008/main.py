import warnings

warnings.filterwarnings("ignore")

import torch
import pandas as pd
import numpy as np
import os
import sys

# Import from provided libraries
from library.config import Config
from library.data import CrystalDataset, collate_batch
from library.model import IALCDS
from library.train import train_model, set_seed
from library.utils import inverse_log_transform, compute_rmsle


def predict(model, dataloader, device):
    """
    Runs inference on a dataloader and returns IDs, predictions, and targets (if available).
    """
    model.eval()
    ids_list = []
    preds_list = []
    targets_list = []

    with torch.no_grad():
        for batch in dataloader:
            # Move data to device
            atom_types = batch["atom_types"].to(device)
            dist_matrix = batch["dist_matrix"].to(device)
            lattice_features = batch["lattice_features"].to(device)
            mask = batch["mask"].to(device)

            # Forward pass
            outputs = model(atom_types, dist_matrix, lattice_features, mask)

            # Inverse transform (log1p -> expm1) to get original scale
            preds_original = inverse_log_transform(outputs)

            preds_list.append(preds_original.cpu().numpy())
            ids_list.append(batch["id"].numpy())

            # Handle targets if present
            if "target" in batch:
                targets_original = inverse_log_transform(batch["target"])
                targets_list.append(targets_original.cpu().numpy())

    preds = np.concatenate(preds_list, axis=0)
    ids = np.concatenate(ids_list, axis=0)

    if targets_list:
        targets = np.concatenate(targets_list, axis=0)
    else:
        targets = None

    return ids, preds, targets


def run():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Train Model
    # We use 50 epochs to ensure fast execution as a baseline.
    # The dataset size is small (~1.7k), so we use the full dataset.
    print(f"Starting training on {device}...")
    model = train_model(num_epochs=50, limit_data=None)

    # 3. Validation Assessment
    print("Performing validation inference...")
    val_dataset = CrystalDataset(mode="val", load_cached_data=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=2,
    )

    val_ids, val_preds, val_targets = predict(model, val_loader, device)

    # Compute Metric
    # Clip predictions to be non-negative (physical constraint)
    val_preds = np.maximum(val_preds, 0)
    metric = compute_rmsle(val_targets, val_preds)

    print(f"Final Validation Metric: {metric}")

    # 4. Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Calculate error magnitude per sample.
    # We use Mean Absolute Error in log space as a proxy for "error magnitude".
    # This aligns with the RMSLE objective.
    log_preds = np.log1p(val_preds)
    log_targets = np.log1p(val_targets)
    sample_errors = np.mean(np.abs(log_preds - log_targets), axis=1)

    # Load validation metadata to get features
    val_df = pd.read_csv(Config.VAL_CSV)

    # Create analysis dataframe
    analysis_df = pd.DataFrame({"id": val_ids, "error": sample_errors})

    # Merge with metadata to link errors to features
    # Ensure IDs match types
    val_df["id"] = val_df["id"].astype(int)
    analysis_df["id"] = analysis_df["id"].astype(int)
    merged_df = pd.merge(val_df, analysis_df, on="id")

    # Select numerical columns for correlation analysis
    # Exclude targets and ID
    exclude_cols = ["id", "formation_energy_ev_natom", "bandgap_energy_ev", "error"]
    numeric_cols = [
        c
        for c in merged_df.select_dtypes(include=[np.number]).columns
        if c not in exclude_cols
    ]

    if numeric_cols:
        correlations = (
            merged_df[numeric_cols]
            .corrwith(merged_df["error"])
            .abs()
            .sort_values(ascending=False)
        )
        print("Top correlations with error magnitude:")
        print(correlations.head(10))
    else:
        print("No numerical features found for correlation analysis.")

    # 5. Submission Generation
    THRESHOLD = 0.05781995991591556

    if metric < THRESHOLD:
        print(
            f"\nValidation metric {metric} is below threshold {THRESHOLD}. Generating submission..."
        )

        test_dataset = CrystalDataset(mode="test", load_cached_data=True)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_batch,
            num_workers=2,
        )

        test_ids, test_preds, _ = predict(model, test_loader, device)

        # Clip negative predictions
        test_preds = np.maximum(test_preds, 0)

        submission_df = pd.DataFrame(
            {
                "id": test_ids,
                "formation_energy_ev_natom": test_preds[:, 0],
                "bandgap_energy_ev": test_preds[:, 1],
            }
        )

        # Ensure correct sorting or order if necessary (though ID matching is explicit)
        submission_df = submission_df.sort_values("id")

        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"\nValidation metric {metric} is NOT below threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    run()
