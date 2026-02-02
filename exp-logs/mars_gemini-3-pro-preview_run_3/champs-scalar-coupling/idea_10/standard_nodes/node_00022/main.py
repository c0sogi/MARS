import sys
import os
import torch
import pandas as pd
import numpy as np
import time

# Add current directory to path to ensure imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.engine import Trainer
from library.utils import seed_everything, calculate_log_mae
from library.data_factory import DataFactory
from library.loader import FlattenedGraphDataset, GraphCollator
from torch.utils.data import DataLoader


def main():
    # ==========================================
    # 1. Configuration for Fast Baseline
    # ==========================================
    # Override Config defaults to ensure execution within time limits
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50000  # Train on ~50k molecules (approx 15% of data)
    Config.MAX_EPOCHS = 5  # Limit epochs for speed
    Config.BATCH_SIZE = 128
    Config.NUM_WORKERS = 4

    # Ensure fully reproducible results
    seed_everything(Config.SEED)

    print("Starting Fast Baseline Run...")
    print(f"Debug Mode: {Config.DEBUG} (Sample Size: {Config.DEBUG_SAMPLE_SIZE})")
    print(f"Max Epochs: {Config.MAX_EPOCHS}")

    # ==========================================
    # 2. Training
    # ==========================================
    # Initialize Trainer (handles data loading and model init)
    trainer = Trainer(load_cached_data=True)

    # Execute training loop
    trainer.fit()

    # ==========================================
    # 3. Final Validation & Failure Analysis
    # ==========================================
    print("\nRunning Final Validation and Failure Analysis...")

    # Load the best model checkpoint
    if os.path.exists(Config.MODEL_SAVE_PATH):
        trainer.model.load_state_dict(
            torch.load(Config.MODEL_SAVE_PATH, map_location=trainer.device)
        )
        print("Loaded best model checkpoint.")
    else:
        print("Warning: No checkpoint found. Using current model state.")

    trainer.model.eval()

    # Prepare for analysis
    val_loader = trainer.val_loader
    standardizer = trainer.standardizer
    device = trainer.device

    all_preds = []
    all_targets = []
    all_types = []
    all_dists = []
    all_errors = []

    # Inference loop on validation set
    with torch.no_grad():
        for batch in val_loader:
            # Move batch to device
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(device)

            # Forward pass
            preds = trainer.model(batch)
            pred_z = preds["coupling"].view(-1)

            # Metadata
            types = batch["coupling_type"]
            targets = batch["coupling_value"]

            if len(types) == 0:
                continue

            # Inverse transform predictions to original scale
            pred_raw = standardizer.inverse_transform(pred_z, types)

            # Calculate absolute errors
            errors = torch.abs(pred_raw - targets)

            # Extract distances for the coupling edges
            # coupling_edge_index maps to the global edge array
            c_edge_idx = batch["coupling_edge_index"]
            dists = batch["edge_attr"][c_edge_idx]

            # Collect results
            all_preds.append(pred_raw.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_types.append(types.cpu().numpy())
            all_dists.append(dists.cpu().numpy())
            all_errors.append(errors.cpu().numpy())

    # Concatenate all batches
    if len(all_preds) > 0:
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        all_types = np.concatenate(all_types)
        all_dists = np.concatenate(all_dists)
        all_errors = np.concatenate(all_errors)

        # Calculate Final Metric
        avg_log_mae, _ = calculate_log_mae(all_preds, all_targets, all_types)
        print(f"Final Validation Metric: {avg_log_mae}")

        # Failure Analysis: Correlations
        df_analysis = pd.DataFrame(
            {
                "error": all_errors,
                "dist": all_dists,
                "target": all_targets,
                "type": all_types,
            }
        )

        corr_dist = df_analysis["error"].corr(df_analysis["dist"])
        corr_mag = df_analysis["error"].corr(df_analysis["target"].abs())

        print("\nFailure Analysis (Correlations with Error):")
        print(f"  Distance: {corr_dist:.4f}")
        print(f"  Target Magnitude: {corr_mag:.4f}")
    else:
        print("Final Validation Metric: 0.0")
        avg_log_mae = 0.0

    # ==========================================
    # 4. Submission
    # ==========================================
    # Threshold check
    threshold = -1.2761284112930298

    if avg_log_mae < threshold:
        print(
            f"\nMetric ({avg_log_mae}) is better than threshold ({threshold}). Generating submission..."
        )
        generate_submission(trainer)
    else:
        print(
            f"\nMetric ({avg_log_mae}) did not meet threshold ({threshold}). Skipping submission."
        )


def generate_submission(trainer):
    """
    Generates predictions for the entire test set and saves to submission.csv.
    """
    # Disable DEBUG to ensure full test set processing
    Config.DEBUG = False
    print("Processing Test Data (Full Set)...")

    # Re-initialize dataset for test split
    # Force load_cached_data=False to ensure we don't load a partial debug cache if it exists
    test_dataset = FlattenedGraphDataset(split="test", load_cached_data=False)

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=GraphCollator(),
    )

    ids = []
    preds = []

    trainer.model.eval()
    device = trainer.device

    print("Running Inference on Test Set...")
    with torch.no_grad():
        for batch in test_loader:
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(device)

            # Forward pass
            out = trainer.model(batch)
            pred_z = out["coupling"].view(-1)

            c_types = batch["coupling_type"]
            c_ids = batch["coupling_id"]

            if len(c_types) == 0:
                continue

            # Inverse transform
            pred_raw = trainer.standardizer.inverse_transform(pred_z, c_types)

            ids.append(c_ids.cpu().numpy())
            preds.append(pred_raw.cpu().numpy())

    # Concatenate results
    ids = np.concatenate(ids)
    preds = np.concatenate(preds)

    # Create DataFrame and Save
    df_sub = pd.DataFrame({"id": ids, "scalar_coupling_constant": preds})

    # Ensure submission directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(f"Submission shape: {df_sub.shape}")


if __name__ == "__main__":
    main()
