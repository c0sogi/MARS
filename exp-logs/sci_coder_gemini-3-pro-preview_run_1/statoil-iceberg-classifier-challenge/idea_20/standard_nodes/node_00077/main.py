import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import log_loss
from scipy.stats import pearsonr

from library.config import Config, set_seed
from library.data import (
    get_loaders,
    get_test_loader,
    process_and_cache_data,
    IcebergDataset,
    get_transforms,
)
from library.model import IcebergResNet18
from library.engine import train_one_epoch, evaluate, predict


# Custom update_bn to handle model(x, angle) signature
def custom_update_bn(loader, model, device=None):
    momenta = {}
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)
            momenta[module] = module.momentum
            module.momentum = None
            module.num_batches_tracked *= 0

    model.train()
    with torch.no_grad():
        for batch in loader:
            # Loader yields (images, angles, labels) or (images, angles)
            images = batch[0].to(device)
            angles = batch[1].to(device)
            model(images, angles)

    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.momentum = momenta[module]


def run_training(train_loader, val_loader, device, epochs, swa_epochs, seed):
    set_seed(seed)

    model = IcebergResNet18().to(device)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
    )

    # Loss
    criterion = nn.BCEWithLogitsLoss()

    # SWA Setup
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=Config.SWA_LR)

    start_swa = epochs - swa_epochs
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        # Train
        loss_train = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )

        if epoch > start_swa:
            # SWA Phase
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            # Standard Phase
            if val_loader:
                loss_val = evaluate(model, val_loader, criterion, device, epoch)
                scheduler.step(loss_val)
            else:
                # If no validation set (full training), step based on train loss
                scheduler.step(loss_train)

    # Update BN statistics for SWA model
    custom_update_bn(train_loader, swa_model, device=device)

    return swa_model


def main():
    device = Config.DEVICE
    print(f"Using device: {device}")

    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------
    # Increased epochs to ensure full convergence before SWA
    TOTAL_EPOCHS = 60
    SWA_EPOCHS = 12
    NUM_SEEDS = 5
    THRESHOLD = 0.16918645240183008

    # -------------------------------------------------------------------------
    # Data Loading
    # -------------------------------------------------------------------------
    print("Loading data...")
    train_loader, val_loader = get_loaders(load_cached_data=True)
    test_loader, test_ids = get_test_loader(load_cached_data=True)

    # -------------------------------------------------------------------------
    # Phase 1: Validation Ensemble
    # -------------------------------------------------------------------------
    print("\n=== Phase 1: Validation Ensemble ===")
    val_probs_ensemble = []

    # Extract targets and angles for analysis
    val_targets = []
    val_angles = []
    for _, angles, labels in val_loader:
        val_targets.extend(labels.numpy().flatten())
        val_angles.extend(angles.numpy().flatten())
    val_targets = np.array(val_targets)
    val_angles = np.array(val_angles)

    seeds = [Config.SEED + i for i in range(NUM_SEEDS)]

    for seed in seeds:
        print(f"\nTraining Validation Model (Seed {seed})...")
        model = run_training(
            train_loader, val_loader, device, TOTAL_EPOCHS, SWA_EPOCHS, seed
        )

        # Predict on Validation Set
        model.eval()
        probs = predict(model, val_loader, device)
        val_probs_ensemble.append(probs)

    # Average Predictions
    avg_val_probs = np.mean(val_probs_ensemble, axis=0)

    # Compute Metric
    final_metric = log_loss(val_targets, avg_val_probs)
    print(f"\nFinal Validation Metric: {final_metric}")

    # -------------------------------------------------------------------------
    # Failure Analysis
    # -------------------------------------------------------------------------
    print("\n=== Failure Analysis ===")
    errors = np.abs(val_targets - avg_val_probs)

    if np.isnan(val_angles).any():
        print("Warning: NaNs in incidence angles. Skipping correlation analysis.")
    else:
        corr, _ = pearsonr(errors, val_angles)
        print(f"Correlation between Error and Incidence Angle: {corr:.4f}")

    # -------------------------------------------------------------------------
    # Phase 2: Submission
    # -------------------------------------------------------------------------
    if final_metric < THRESHOLD:
        print(f"\nMetric {final_metric} < {THRESHOLD}. Proceeding to submission...")

        # Prepare Full Dataset (Train + Val)
        data_cache = process_and_cache_data(load_cached_data=True)

        all_images = np.concatenate(
            [data_cache["train"]["images"], data_cache["val"]["images"]]
        )
        all_angles = np.concatenate(
            [data_cache["train"]["angles"], data_cache["val"]["angles"]]
        )
        all_labels = np.concatenate(
            [data_cache["train"]["labels"], data_cache["val"]["labels"]]
        )

        full_dataset = IcebergDataset(
            images=all_images,
            angles=all_angles,
            labels=all_labels,
            transform=get_transforms("train"),  # Use training transforms (augmentation)
        )

        full_loader = DataLoader(
            full_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        test_probs_ensemble = []

        for seed in seeds:
            print(f"\nTraining Full Model (Seed {seed})...")
            # Pass None as val_loader
            model = run_training(
                full_loader, None, device, TOTAL_EPOCHS, SWA_EPOCHS, seed
            )

            # Predict on Test Set
            probs = predict(model, test_loader, device)
            test_probs_ensemble.append(probs)

        # Average Predictions
        avg_test_probs = np.mean(test_probs_ensemble, axis=0)

        # Save Submission
        df = pd.DataFrame({"id": test_ids, "is_iceberg": avg_test_probs})
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(f"\nMetric {final_metric} >= {THRESHOLD}. Skipping submission.")


if __name__ == "__main__":
    main()
