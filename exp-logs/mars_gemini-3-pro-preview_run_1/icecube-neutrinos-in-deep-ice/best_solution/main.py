import sys
import os
import torch
import numpy as np
import pandas as pd
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_add_pool
from scipy.stats import pearsonr

# Ensure local library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.model import IceCubeDGCN
from library.data import IceCubeGraphDataset
from library.engine import Engine
from library.predict import generate_submission


def main():
    # 1. Initialize Configuration and Environment
    Config.initialize()
    device = Config.DEVICE
    print(f"Running on device: {device}")

    # 2. Data Selection for Fast Baseline
    # Load metadata to select specific batch IDs
    train_meta_full = pd.read_parquet(Config.TRAIN_META_PATH)
    val_meta_full = pd.read_parquet(Config.VAL_META_PATH)

    # Select a small subset of batches to ensure execution within time limits
    # 2 batches for training (~400k events), 1 batch for validation (~200k events)
    train_batch_ids = train_meta_full["batch_id"].unique()[:2]
    val_batch_ids = val_meta_full["batch_id"].unique()[:1]

    print(f"Selected Training Batches: {train_batch_ids}")
    print(f"Selected Validation Batches: {val_batch_ids}")

    # Initialize Datasets
    train_dataset = IceCubeGraphDataset(mode="train", batch_ids=train_batch_ids)
    val_dataset = IceCubeGraphDataset(mode="val", batch_ids=val_batch_ids)

    # Initialize DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model, Optimizer, and Scheduler Setup
    model = IceCubeDGCN().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Calculate steps for OneCycleLR
    # Approximate length based on metadata count for selected batches
    n_train_samples = len(
        train_meta_full[train_meta_full["batch_id"].isin(train_batch_ids)]
    )
    # Add a buffer to steps_per_epoch to prevent OneCycleLR from overflowing
    # due to slight mismatches in DataLoader batch counts with IterableDataset
    steps_per_epoch = (n_train_samples // Config.BATCH_SIZE) + 10

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        epochs=Config.NUM_EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3,
    )

    # 4. Training
    engine = Engine(model, device, optimizer, scheduler)
    save_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print("Starting training...")
    engine.fit(
        train_loader,
        val_loader,
        epochs=Config.NUM_EPOCHS,
        patience=Config.PATIENCE,
        save_path=save_path,
    )

    # 5. Final Validation Assessment
    print("Loading best model for final validation...")
    model.load_state_dict(torch.load(save_path, map_location=device))

    val_loss, val_mae = engine.validate(val_loader)
    print(f"Final Validation Metric: {val_mae}")

    # 6. Failure Analysis
    print("Performing Failure Analysis...")
    model.eval()

    all_errors = []
    all_n_pulses = []
    all_total_charges = []

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)

            # Predict
            pred = model(batch)

            # Calculate angular error per event
            pred_norm = torch.nn.functional.normalize(pred, p=2, dim=1)
            target_norm = torch.nn.functional.normalize(batch.y, p=2, dim=1)
            dot_prod = torch.sum(pred_norm * target_norm, dim=1).clamp(-1.0, 1.0)
            angles = torch.acos(dot_prod).cpu().numpy()
            all_errors.extend(angles)

            # Extract Features
            # 1. Number of pulses: Count nodes per graph in the batch
            # batch.batch is a tensor of graph indices [0, 0, ..., 1, 1, ...]
            counts = torch.bincount(batch.batch).cpu().numpy()
            all_n_pulses.extend(counts)

            # 2. Total Charge: Sum of 10^log_charge per graph
            # Feature index 4 is log10(charge)
            log_charges = batch.x[:, 4]
            charges = torch.pow(10, log_charges)
            # Sum charges based on batch index
            sum_charges = (
                global_add_pool(charges.unsqueeze(-1), batch.batch)
                .squeeze(-1)
                .cpu()
                .numpy()
            )
            all_total_charges.extend(sum_charges)

    # Compute Correlations
    all_errors = np.array(all_errors)
    all_n_pulses = np.array(all_n_pulses)
    all_total_charges = np.array(all_total_charges)

    if len(all_errors) > 1:
        corr_pulses, _ = pearsonr(all_errors, all_n_pulses)
        corr_charge, _ = pearsonr(all_errors, all_total_charges)
        print(f"Correlation (Error vs N_Pulses): {corr_pulses:.6f}")
        print(f"Correlation (Error vs Total_Charge): {corr_charge:.6f}")

    # 7. Submission Generation
    THRESHOLD = 1.5417
    if val_mae < THRESHOLD:
        print(
            f"Validation metric {val_mae} is below threshold {THRESHOLD}. Generating submission..."
        )
        generate_submission(
            model_weights_path=save_path,
            output_csv_path=Config.SUBMISSION_PATH,
            batch_ids=None,  # Process all test batches
            num_workers=Config.NUM_WORKERS,
        )
    else:
        print(
            f"Validation metric {val_mae} is above threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
