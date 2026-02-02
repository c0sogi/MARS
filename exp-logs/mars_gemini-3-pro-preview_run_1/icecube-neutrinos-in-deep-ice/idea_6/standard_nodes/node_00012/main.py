import os
import shutil
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.utils import set_seed, spherical_to_cartesian, cartesian_to_spherical
from library.data_processing import IceCubeDataset, collate_fn
from library.model_architecture import DynGTNet
from library.training_engine import Trainer
from library.inference_engine import generate_submission


def analyze_failures(model, val_loader):
    """
    Performs failure analysis on the validation set.
    Calculates correlations between angular error and input features (n_pulses, total_charge).
    """
    model.eval()
    device = next(model.parameters()).device

    errors = []
    n_pulses_list = []
    total_charge_list = []

    print("Running failure analysis on validation set...")

    with torch.no_grad():
        for batch in val_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)  # (azimuth, zenith)

            # Predict
            pred_vecs = model(x)

            # Convert predictions to spherical (azimuth, zenith)
            pred_x, pred_y, pred_z = pred_vecs[:, 0], pred_vecs[:, 1], pred_vecs[:, 2]
            pred_az, pred_zen = cartesian_to_spherical(pred_x, pred_y, pred_z)

            # Calculate Angular Error
            # Convert targets to cartesian
            true_az, true_zen = y[:, 0], y[:, 1]
            t_x, t_y, t_z = spherical_to_cartesian(true_az, true_zen)

            # Convert preds to cartesian (re-using computation or from spherical)
            p_x, p_y, p_z = spherical_to_cartesian(pred_az, pred_zen)

            # Dot product for cosine similarity
            dot = t_x * p_x + t_y * p_y + t_z * p_z
            dot = torch.clamp(dot, -1.0, 1.0)

            # Angular error in radians
            batch_errors = torch.acos(dot).cpu().numpy()
            errors.extend(batch_errors)

            # --- Extract Meta Features from Input x ---
            # x shape: (Batch, N_Pulses, 6)
            # Feature 4: log_charge (padded with -5.0)
            # Feature 5: auxiliary

            log_q = x[:, :, 4]

            # Mask for valid pulses (log_q > -4.0, since padding is -5.0)
            mask = log_q > -4.0

            # 1. Number of Pulses
            n_pulses = mask.sum(dim=1).cpu().numpy()
            n_pulses_list.extend(n_pulses)

            # 2. Total Charge
            # Recover charge: q approx 10^log_q
            q_recovered = torch.pow(10, log_q)
            # Zero out padding
            q_recovered = q_recovered * mask.float()
            total_q = q_recovered.sum(dim=1).cpu().numpy()
            total_charge_list.extend(total_q)

    # Compute Correlations
    errors = np.array(errors)
    n_pulses = np.array(n_pulses_list)
    total_charge = np.array(total_charge_list)

    if len(errors) > 1:
        corr_n, _ = pearsonr(errors, n_pulses)
        corr_q, _ = pearsonr(errors, total_charge)

        print(f"Correlation (Error vs n_pulses): {corr_n:.10f}")
        print(f"Correlation (Error vs total_charge): {corr_q:.10f}")
    else:
        print("Insufficient data for correlation analysis.")


def main():
    # 1. Setup
    set_seed(Config.SEED)
    Config.setup()

    # --- Configure for Fast Baseline ---
    # Adjust hyperparameters to ensure completion within 2 hours
    Config.MAX_EPOCHS = 2
    Config.BATCH_SIZE = 512  # Increased for A100 efficiency
    Config.NUM_WORKERS = 12  # Utilize available vCPUs

    print("Initializing Datasets...")
    train_dataset = IceCubeDataset(mode="train")
    val_dataset = IceCubeDataset(mode="val")

    # --- Subsample Data ---
    # To meet time constraints, we train on a subset of batches (e.g., 5 batches ~1M events)
    # and validate on a subset (e.g., 1 batch ~200k events).

    # Select first 5 unique batches for training
    train_batches = train_dataset.meta["batch_id"].unique()[:5]
    train_dataset.meta = train_dataset.meta[
        train_dataset.meta["batch_id"].isin(train_batches)
    ].copy()
    # Reset index to ensure correct access
    train_dataset.meta = train_dataset.meta.reset_index(drop=True)
    print(
        f"Subsampled Training Data: {len(train_dataset)} events from batches {train_batches}"
    )

    # Select first 1 unique batch for validation
    val_batches = val_dataset.meta["batch_id"].unique()[:1]
    val_dataset.meta = val_dataset.meta[
        val_dataset.meta["batch_id"].isin(val_batches)
    ].copy()
    val_dataset.meta = val_dataset.meta.reset_index(drop=True)
    print(
        f"Subsampled Validation Data: {len(val_dataset)} events from batches {val_batches}"
    )

    # 2. Data Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,  # Can use same or larger batch size
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # 3. Model Initialization
    print("Initializing DynGTNet Model...")
    model = DynGTNet()

    # 4. Training
    print("Starting Training...")
    trainer = Trainer(model, train_loader, val_loader, Config)
    trainer.fit()

    # 5. Final Validation
    print("Performing Final Validation on Hold-out Set...")
    # Using the subsampled val_loader which acts as our hold-out set for this run
    final_metric = trainer.validate()
    print(f"Final Validation Metric: {final_metric:.15f}")

    # 6. Failure Analysis
    analyze_failures(model, val_loader)

    # 7. Submission Logic
    THRESHOLD = 1.106787505105717

    if final_metric < THRESHOLD:
        print(
            f"\nMetric {final_metric} is lower than threshold {THRESHOLD}. Generating submission..."
        )

        # Generate submission using large batch size for speed
        # Config.SUBMISSION_DIR is set to ./working/idea_6/submission in Config
        df = generate_submission(batch_size=2048, num_workers=Config.NUM_WORKERS)

        # Move submission to the required location: ./submission/submission.csv
        target_dir = "./submission"
        os.makedirs(target_dir, exist_ok=True)
        target_path = os.path.join(target_dir, "submission.csv")

        # Source path from Config
        source_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

        if os.path.exists(source_path):
            shutil.copy(source_path, target_path)
            print(f"Final submission saved to {target_path}")
        else:
            print(f"Error: Generated submission not found at {source_path}")

    else:
        print(
            f"\nMetric {final_metric} is not lower than threshold {THRESHOLD}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
