import os
import sys
import torch
import numpy as np
import pandas as pd
import random
import time
from torch.utils.data import DataLoader

# Import from provided libraries
from library.config import (
    DEVICE,
    SEED,
    BATCH_SIZE,
    NUM_WORKERS,
    WORKING_DIR,
    SUBMISSION_PATH,
    STATS_PATH,
    MAX_EPOCHS,
)
from library.data import get_train_val_datasets, get_test_dataset, collate_molecules
from library.model import MPIN
from library.engine import train_model, evaluate
from library.utils import Standardizer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_failure_analysis(model, val_loader, standardizer, device):
    """
    Analyzes model performance on the validation set to identify error correlations.
    """
    print("\n--- Failure Analysis ---")
    model.eval()

    all_errors = []
    all_types = []
    all_targets = []
    all_mol_sizes = []

    with torch.no_grad():
        for batch in val_loader:
            # Move to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            # Predict
            preds_norm = model(batch)

            # Metadata
            types = batch["coupling_type"]
            targets = batch["coupling_value"]

            # Inverse transform
            preds = standardizer.inverse_transform(preds_norm, types)

            # Calculate Absolute Error
            errors = torch.abs(preds - targets)

            # Collect data
            all_errors.append(errors.cpu().numpy())
            all_types.append(types.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

            # Molecule size (num_nodes) repeated for each coupling in the molecule
            # batch['batch'] maps nodes to graph index. We need mapping for couplings.
            # However, simpler proxy: just use batch-level stats isn't granular enough.
            # We can get num_nodes from the batch structure if needed, but let's stick
            # to coupling-level features for speed.

    # Concatenate
    errors_np = np.concatenate(all_errors)
    types_np = np.concatenate(all_types)
    targets_np = np.concatenate(all_targets)

    # Create DataFrame for correlation
    df_analysis = pd.DataFrame(
        {"error": errors_np, "type": types_np, "target_magnitude": np.abs(targets_np)}
    )

    # Calculate correlations
    corr_type = df_analysis["error"].corr(df_analysis["type"])
    corr_mag = df_analysis["error"].corr(df_analysis["target_magnitude"])

    print(f"Correlation (Error vs Coupling Type): {corr_type:.4f}")
    print(f"Correlation (Error vs Target Magnitude): {corr_mag:.4f}")

    # Grouped analysis
    print("\nMean Absolute Error by Coupling Type:")
    print(df_analysis.groupby("type")["error"].mean())


def main():
    # 1. Setup
    set_seed(SEED)
    print(f"Running on device: {DEVICE}")

    # Override Max Epochs for Fast Baseline
    # The config has 50, but we limit to 10 to ensure completion within 2 hours
    # while utilizing the full dataset.
    FAST_EPOCHS = 10
    print(f"Limiting training to {FAST_EPOCHS} epochs for fast baseline.")

    # 2. Data Loading
    print("Loading datasets...")
    # Load datasets (cached if available)
    train_dataset, val_dataset = get_train_val_datasets(load_cached=True)

    # Initialize Standardizer (fits on training data implicitly via process_data or we ensure it loads)
    # Note: process_data in library.data saves the stats. Standardizer loads them.
    standardizer = Standardizer(STATS_PATH)
    # Ensure stats are loaded/available
    if not os.path.exists(STATS_PATH):
        # Fallback: manually fit if for some reason process_data didn't (unlikely)
        print("Fitting standardizer manually...")
        standardizer.fit(train_dataset.coupling_types, train_dataset.coupling_values)
    else:
        standardizer.load()

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_molecules,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_molecules,
        pin_memory=True,
    )

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # 3. Model Initialization
    model = MPIN().to(DEVICE)

    # 4. Training
    print("Starting training...")
    best_metric = train_model(
        model,
        train_loader,
        val_loader,
        standardizer,
        device=DEVICE,
        epochs=FAST_EPOCHS,
        save_path=os.path.join(WORKING_DIR, "best_model.pth"),
    )

    # 5. Validation & Metrics
    print("\n--- Final Evaluation ---")
    # Load best model
    model.load_state_dict(
        torch.load(os.path.join(WORKING_DIR, "best_model.pth"), map_location=DEVICE)
    )

    # Compute Final Metric
    val_score, _ = evaluate(model, val_loader, DEVICE, standardizer)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {val_score}")

    # Failure Analysis
    run_failure_analysis(model, val_loader, standardizer, DEVICE)

    # 6. Submission
    THRESHOLD = -1.2761284112930298

    if val_score < THRESHOLD:
        print(
            f"\nValidation score ({val_score}) meets threshold ({THRESHOLD}). Generating submission..."
        )

        # Load Test Data
        test_dataset = get_test_dataset(load_cached=True)
        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            collate_fn=collate_molecules,
        )

        model.eval()
        ids_list = []
        preds_list = []

        with torch.no_grad():
            for batch in test_loader:
                # Move to device
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(DEVICE)

                # Predict
                preds_norm = model(batch)

                # Inverse Transform
                types = batch["coupling_type"]
                preds_phys = standardizer.inverse_transform(preds_norm, types)

                # Store
                ids_list.append(batch["coupling_id"].cpu().numpy())
                preds_list.append(preds_phys.cpu().numpy())

        # Concatenate
        all_ids = np.concatenate(ids_list)
        all_preds = np.concatenate(preds_list)

        # Create Submission DataFrame
        df_sub = pd.DataFrame({"id": all_ids, "scalar_coupling_constant": all_preds})

        # Sort by ID to match sample submission format (optional but good practice)
        df_sub = df_sub.sort_values("id")

        # Save
        df_sub.to_csv(SUBMISSION_PATH, index=False)
        print(f"Submission saved to {SUBMISSION_PATH}")

    else:
        print(
            f"\nValidation score ({val_score}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
